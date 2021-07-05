#!/home/class239/anaconda3/bin/python3
import sys
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

# ========== DRIVING METHOD ==========

def main():
    """
    Generate list of interferograms to process and baseline plot given baseline table and processing parameters
    
    Usage A - generate default prm_file
        get_baseline.py default
    
    Usage B -  Make interferogram selections
        get_baseline.py prm_file baseline_file

    INPUT:
    prm_file - parameter (PRM) file containing date and baseline values for interferogram selection
    baseline_file - GMTSAR baseline table file
    
    OUTPUT:
    short.dat - list of interferograms in YYYYMMDD_YYYYMMDD format
    intf.in   - list of interferogram pairs in SLC naming convention (e.g. S1A20150126_ALL_F1:S1A20150607_ALL_F1), for input into GMTSAR interferogram scripts 
        Note: subsets of these lists will be generated which correspond to the selection parameters provided in the prm_file
        Ex:
        intf.in.SEQ for SEQ = True
        intf.in.ORD_2 for ORD = 2
        intf.in.LONG for LONG_INTFS  = True

    baseline_plot.eps - plot of interferograms selected
    """

    # If specified, write default parameter file
    if (len(sys.argv) == 2) and sys.argv[1] == 'default':
        print('Writing default.PRM')

        with open('default.PRM', 'w') as file:
            file.write(default_prm_file.__doc__)
        sys.exit()

    # Return docstring if arguments unspecified
    elif len(sys.argv) < 3:
        print(main.__doc__)
        sys.exit()

    # Get arguments
    prm_file      = sys.argv[1];
    baseline_file = sys.argv[2];

    # Read in baseline table
    baseline_table = load_baseline_table(baseline_file) 


    # Get pairs
    intf_inputs, intf_dates, subset_inputs, subset_dates, supermaster = select_pairs(baseline_table, prm_file)
    

    # Write intferferogram list to use with GMTSAR scripts
    write_intf_list('intf.in', intf_inputs)


    # Write dates to list of interferogram directories to be generate=d
    write_intf_list('short.dat', [dates[0].strftime('%Y%m%d') + '_' + dates[1].strftime('%Y%m%d') for dates in intf_dates])


    # Also write interferogram subset lists
    for key in subset_inputs:
        write_intf_list('intf.in.' + key, subset_inputs[key])
        write_intf_list('short.dat.' + key, [dates[0].strftime('%Y%m%d') + '_' + dates[1].strftime('%Y%m%d') for dates in subset_dates[key]])


    # Make baseline plot 
    baseline_plot(intf_dates, baseline_table, supermaster=supermaster)


# ========== FUNCTIONS ==========

def load_PRM(prm_file, var_in):
    """
    Read GMTASAR-style PRM file
    """

    # Intialize dictionary
    prm = {}

    # Set date format
    date_format = '%Y%m%d'

    # Set everything uppercase just in case
    var_in = var_in.upper()

    # Read in line by line
    with open(prm_file, 'r') as f:
        for line in f:
            # Split into row elements
            item = line.split()
            
            # Catch empty lines
            if not item:
                continue

            # Catch comments
            elif (item[0] == '#') or ('#' in item[2]):
                continue
            else:
                # Use first and last elements of split line (excluding '=') to generate dictionaries for each line in PRM
                var = item[0]
                # Handle different types of variable values
                # Check date first
                if 'DATE' in var: 
                    try: # Only accepts dates of specified date_format
                        val = dt.datetime.strptime(item[2], date_format)
                    except ValueError:
                        try: # Handle numbers
                            val = float(item[2])
                        except ValueError: # Handle anything else
                            val = item[2]

                else: # Handle numbers
                    try:
                        val = float(item[2])
                    except ValueError: # Handle anything else
                        val = item[2]


                # Append to dictionary
                prm[var] = val

    # Check for variable
    if var_in not in prm:
        val_out = None
    else:
        # Extract parameter value
        val_out = prm[var_in]

    return val_out


def load_baseline_table(file_name):
    """
    Load GMTSAR baseline table. 
    """

    baseline_table = pd.read_csv(file_name, header=None, delim_whitespace=True)  # Read table
    baseline_table.columns = ['scene_id', 'sar_time', 'sar_day', 'B_para', 'Bp']

    dates = []

    for scene in baseline_table['scene_id']:

        # Handle Sentinel-1 IDs
        if 'S1A' in scene or 'S1B' in scene or 's1a' in scene or 's1b' in scene:
            for i in range(len(scene) - 8):
                tmp_str = scene[i:i + 8]
                if tmp_str.isdigit():
                    try:
                        dates.append(dt.datetime.strptime(tmp_str, '%Y%m%d'))
                        break
                    except ValueError:
                        continue
            print(dates)

        # Handle ALOS-2 IDs
        elif 'ALOS2' in scene:
            tmp_str = scene.split('-')[3]
            try:
                dates.append(dt.datetime.strptime(tmp_str, '%y%m%d'))
            except ValueError:
                print('Date not identified in {}'.format(tmp_str))
                continue

        else:
            print('Error: Satellite name not identified in {}'.format(file_name))
            print('(Currently only compatible with ALOS-2 and Sentinel-1)')
            sys.exit()


    # Append datetime objects and sort dataframe before returning
    baseline_table['date'] = dates
    baseline_table = baseline_table.sort_values(by='sar_time')
    baseline_table = baseline_table.reset_index(drop=True)

    return baseline_table


def write_intf_list(file_name, intf_list):
    """
    Write list of interferograms to specified file_name.
    """

    with open(file_name, 'w') as file:
        for intf in intf_list:
            file.write(intf + '\n')


def select_pairs(baseline_table, prm_file):
    """
    Select interferogmetric pairs based off of parameters specified in prm_file
    """

    # ---------- SET THINGS UP ----------
    # Get number of aquisitions
    N = len(baseline_table)  
    print()
    print('Number of SAR scenes =', N)

    # Check pair selection parameters
    SEQ  = load_PRM(prm_file, 'SEQ')
    ORD  = load_PRM(prm_file, 'ORD')
    LONG = load_PRM(prm_file, 'LONG')

    # Load baseline parameters
    defaults = [0, 0, 0, 0] # Default values
    BL_MODE  = load_PRM(prm_file, 'BL_MODE')
    BP_MAX   = load_PRM(prm_file, 'BP_MAX')
    DT_MIN   = load_PRM(prm_file, 'DT_MIN')
    DT_MAX   = load_PRM(prm_file, 'DT_MAX')

    # If any parameter is unspecified, instate default values
    for param, value, default in zip(['BP_MAX', 'DT_MIN', 'DT_MAX'], [BL_MODE, BP_MAX, DT_MIN, DT_MAX], defaults):
        if value == None:
            print('{} not specified, default = {}'.format(param, default))
            param = default

    # Compute mean baseline
    Bp_mean = baseline_table['Bp'].mean()

    # Get supermaster scene
    DATE_MASTER = load_PRM(prm_file, 'DATE_MASTER');

    if DATE_MASTER not in baseline_table['date']:
        # Find scene with baseline closest to mean if no date is specified in PRM file
        supermaster_tmp = baseline_table[abs(baseline_table['Bp'] - Bp_mean) == min(abs(baseline_table['Bp'] - Bp_mean)) ]
        print()
        print('DATE_MASTER = {} is not found in dataset'.format(DATE_MASTER))
        print('Using scene with baseline closest to stack mean ({} m):'.format(np.round(Bp_mean, 2)))
        print('Master date = {} '.format(pd.to_datetime(supermaster_tmp['date'].values[0]).strftime('%Y/%m/%d')))
        print('Baseline    = {} m'.format(np.round(supermaster_tmp['Bp'].values[0], 2)))
    
    else:
        print('DATE_MASTER = {}'.format(DATE_MASTER))
        supermaster_tmp = baseline_table[baseline_table['date'] == DATE_MASTER]

    # Convert to dictionary
    supermaster = {}

    for col in zip(supermaster_tmp.columns):
        supermaster[col[0]] = supermaster_tmp[col[0]].values[0]
        

    # ---------- INTERFEROGRAM SELECTION ----------
    # This portion of the code operates by 'turning on' elements of a NxN network matrix corresponding to all possible interferometric pairs
    # All values start 'off'

     # Initialize dictionary to contain a network matrix for each subset of interferograms to be made
    subset_IDs = {}

    # If SEQ is specified, select every sequential interferogram
    if bool(SEQ) == True:
        print()
        print('Making sequential interferograms')

        ID_SEQ = np.zeros((N, N)) 

        for i in range(N):
            for j in range(N):
                # if np.mod(i, 1) == 0:    
                if abs(j - i) == 1:
                    ID_SEQ[i, j] = 1

        subset_IDs['SEQ'] = ID_SEQ

    # If ORD is specified, select all n-order pairs
    if ORD > 0:
        print('Making all order-{} interferograms'.format(int(ORD)))
        ID_ORD = np.zeros((N, N)) 

        for i in range(N):
            for j in range(N):
                # if np.mod(i, ORD + 1) == 0:    
                if abs(j - i) == ORD:
                    ID_ORD[i, j] = 1

        subset_IDs['ORD_{}'.format(int(ORD))] = ID_ORD

    # If LONG is specified, identify scenes which fit date range provided by LONG_START and LONG_END
    if bool(LONG) == True:
        print('Making long interferograms')

        ID_LONG = np.zeros((N, N)) 

        # Read dates 
        LONG_START = load_PRM(prm_file,'LONG_START');
        LONG_END   = load_PRM(prm_file,'LONG_END');

        # Or set defaults
        for param, value, default in zip(['LONG_START'], [LONG_START, LONG_END], [150, 270]):
            if value == None:
                print('{} not specified, default = {}'.format(param, default))
                param = default

        # Identify all dates in stack that fall within Julian day range
        long_scenes = baseline_table[[(date.timetuple().tm_yday >= LONG_START) & (date.timetuple().tm_yday <= LONG_END) for date in baseline_table['date']]]
        # years       = np.unique([date.year for date in baseline_table['date']])
        
        # Initialize while loop
        i = 0
        complete = False
        jday0 = baseline_table['date'][i].timetuple().tm_yday # Julian day of first aquisition
        year0 = baseline_table['date'][i].year # year of first aquisition

        # If first scene precedes window, make first pair in same year. Otherwise, make first pair in next year 
        if jday0 < LONG_START:
            year = year0
        else:
            year = year0 + 1

        # Pair current scene with baseline-minimizing scene in next available window
        while complete == False:

            # From 'long_scenes', identify first window
            long_scenes0 = []

            while len(long_scenes0) == 0:
                start0       = dt.datetime.strptime(str(int(year*1000 + LONG_START)), '%Y%j') # Convert dates from ints to str so datetime can use them
                end0         = dt.datetime.strptime(str(int(year*1000 + LONG_END)), '%Y%j')
                long_scenes0 = long_scenes[(long_scenes['date'] >= start0) & (long_scenes['date'] <= end0)]
                year += 1

                # Once the year of the final scene is reached, if the scene is in or before the window, set it to be the reference image scene
                if (year == baseline_table['date'].iloc[-1].year) and (baseline_table['date'][i].timetuple().tm_yday <= LONG_END):
                    long_scenes0 = baseline_table[baseline_table['date'] == baseline_table['date'].iloc[-1]] # This is silly indexing, but it must be to work.
                    complete = True
                    # Otherwise, continue for one more pair

                # If the later case is not triggered then the this one will be.
                if year > baseline_table['date'].iloc[-1].year: 
                    long_scenes0 = baseline_table.iloc[-1, :]
                    complete = True

            # Find index of scene within window that minimizes the perpendicular baseline with respect to the initial scene
            j = (abs(baseline_table['Bp'][i] - long_scenes0['Bp'])).idxmin()

            # Turn element on in subset array
            ID_LONG[i, j] = 1

            # Reset index
            i = j

        subset_IDs['LONG'] = ID_LONG

    # If BL_MODE is nonzero, use baseline constraints
    if BL_MODE > 0:
        print()
        print('Max. perpendicular baseline = {:.0f} m'.format(BP_MAX))
        print('Min. epoch length           = {:.0f} days'.format(DT_MIN))
        print('Max. epoch length           = {:.0f} days'.format(DT_MAX))

        ID_BASELINE = np.zeros((N, N)) 

        # Loop over all pairs
        for i in range(N):
            for j in range(N):
                # Perpendicular baseline
                dp = baseline_table['Bp'][i] - baseline_table['Bp'][j]

                # Epoch length
                dT = (baseline_table['date'][j] - baseline_table['date'][i]).days

                # Select if all three limits are satisfied
                if (abs(dp) < BP_MAX) and (dT >= DT_MIN) and (dT <= DT_MAX):
                    ID_BASELINE[i, j] = 1

        if BL_MODE == 1:
            # Include all interferograms satisfying baseline constraints
            print('Including all intereferograms satisfying baseline constraints')
            subset_IDs['BL'] = ID_BASELINE

        elif BL_MODE == 2:
            # Select interferograms satisfying baseline constraints from previous selectionss
            print('Enforcing baseline constraints on selections from SEQ, ORD, and/or LONG')
            for key in subset_IDs.keys():
                subset_IDs[key] *= ID_BASELINE


    # ---------- PREPARE OUTPUT LISTS ----------
    # Create initial and repeat matricies of dimension N x N
    initials = np.array(list(baseline_table['date'])).repeat(N).reshape(N, N)
    repeats  = np.array(list(baseline_table['date'])).repeat(N).reshape(N, N).T

    # Loop through subset dictionary to make individual subset interferograms] lists
    subset_inputs = {}
    subset_dates = {}

    for key in subset_IDs.keys():
        inputs = []
        dates  = []

        for i in range(len(subset_IDs[key])):
            for j in range(len(subset_IDs[key][0])):
                if subset_IDs[key][i, j] == 1 and initials[i, j] < repeats[i, j]:  # We only want the upper half of the matrix, so ignore intf pairs where 'initial' comes after 'repeat'
                    inputs.append(baseline_table['scene_id'][i] + ':' + baseline_table['scene_id'][j])
                    dates.append([baseline_table['date'][i],baseline_table['date'][j]])

        subset_inputs[key] = inputs 
        subset_dates[key]  = dates


    # Aggregate to master lists
    intf_inputs = []
    intf_dates  = []

    for key in subset_inputs:
        intf_inputs.extend(subset_inputs[key])

    for key in subset_dates:
        intf_dates.extend(subset_dates[key])

    # Get number of interferogams to make
    n = len(intf_inputs)
    print()
    print('Total number of interferograms = {}'.format(n))


    return intf_inputs, intf_dates, subset_inputs, subset_dates, supermaster


def baseline_plot(intf_dates, baseline_table, supermaster={}):

    """
    Make baseline netwwork plot for given set of interferograms

    INPUT:
    intf_dates     - list containing n interferogram date pairs (n, 2)
    baseline_table - Dataframe containing appended GMTSAR baseline info table
    (supermaster   - supply dictionary containing info for the supermaster scene; will be plotted in red)
    """

    # Check for supermaster; set to empty if none is provided
    if len(supermaster) == 0:
        supermaster['dates'] = None
        supermaster['Bp'] = None

    # Initialize plot
    fig, ax = plt.subplots(figsize=(10,6))

    # Plot pairs
    for date_pair in intf_dates:
        # Get corresponding baselines
        Bp_pair = [baseline_table[baseline_table['date'] == date]['Bp'].values for date in date_pair]

        # Plot
        # print(date_pair, Bp_pair)

        ax.plot(date_pair, Bp_pair, c='k', linewidth=1, zorder=0)


    # Plot nodes
    for i in range(len(baseline_table)):

        # Change settings if master
        if baseline_table['date'][i] == supermaster['date']:
            c = 'r'
            c_text = 'r'
            s = 30
        else:
            c = 'C0'
            c_text = 'k'
            s = 20

        ax.scatter(baseline_table['date'][i], baseline_table['Bp'][i], marker='o', c=c, s=20)

        # Offset by 10 days/5 m for readability
        ax.text(baseline_table['date'][i] + 0.005*(baseline_table['date'].iloc[-1] - baseline_table['date'].iloc[0]), 
                baseline_table['Bp'][i]   + 0.01*(baseline_table['Bp'].iloc[-1] - baseline_table['Bp'].iloc[0]), 
                baseline_table['date'][i].strftime('%Y/%m/%d'), 
                size=8, color=c_text, 
                # bbox={'facecolor': 'w', 'pad': 0, 'edgecolor': 'w', 'alpha': 0.7}
                )
    
    ax.set_ylabel('Perpendicular baseline (m)')
    ax.set_xlabel('Date')
    ax.tick_params(direction='in')
    plt.savefig('baseline_plot.eps')
    plt.show()


def default_prm_file():
    '''
# ---------- Dates ----------
DATE_START  = 1900/01/01  # Lower bound on scene dates to use (YYYY/MM/DD)
DATE_END    = 2100/01/01  # Upper bound on scene dates to use (YYYY/MM/DD)
DATE_MASTER = None        # Date of master scene for image alignment (if not specified, scene closest to perpendicular baseline mean is used)

# ---------- Pair types ----------
# For all options, set to 0 to not include in selection process

SEQ        = 1    # Generate sequential pairs, starting from initial scene
ORD        = 2    # Generate nth-order pairs (e.g. ORD = 1 is equivalent to SEQ, ORD = 2 skips one scene, etc.)
LONG       = 1    # Generate chain of 6-18 month pairs that connect the first and last dates
LONG_START = 150  # Earliest day-of-year to use in possible pairs (i.e. 1 for Jan. 1, 152 for June 1, etc)
LONG_END   = 270  # Latest dat-of-year to use in possible pairs


# ---------- Baseline constraints ---------- 
# Temporal and perpendicular baseline limits may be used in the following ways:
# 1 - Make all interferograms which satisfy give constraints regardless of specification from SEQ, N, or LONG
# 2 - Use baseline constraints as a filter on previously specified pairs from SEQ, N, or LONG

BL_MODE = 0    # Choose baseline constraint mode (1, 2, or 0 to not use) 
BP_MAX  = 100  # Maximum perpendicular baseline (m)
DT_MIN  = 0    # Minimum interferogram epoch length (days)
DT_MAX  = 600  # Maximums interferogram epoch length (days) 
    '''
    return




if __name__ == '__main__':
    main()