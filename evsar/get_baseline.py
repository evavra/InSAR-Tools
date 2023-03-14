# ##!/home/class239/anaconda3/bin/python3
import sys
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# ========== DRIVING METHOD ==========
def main():
    """
    -------------------------------------- OPTION A -------------------------------------- 
    Generate parameter file using specified baseline constraints.

    Usage:
        python get_baseline.py DT_MIN DT_MAX BP_MAX file_name [DATE_REF]

    Input:
        DT_MIN    - minimum interferogram epoch length (days)
        DT_MAX    - maximum interferogram epoch length (days)
        BP_MAX    - maximum perpendicular baseline (m)
        file_name - name of parameter file to create (i.e. file_name.PRM)
        DATE_REF  - [optional] reference date (YYYY/MM/DD)

    Output:
        file_name.PRM - interferogram selection parameter file (saved to disk)

    -------------------------------------- OPTION B -------------------------------------- 
    Generate lists of interferograms to process and make interferogram baseline plot.
    
    Usage:
        python get_baseline.py prm_file baseline_file [modes]

        Examples:
        python get_baseline.py prm_file baseline_file 
        python get_baseline.py prm_file baseline_file REF
        python get_baseline.py prm_file baseline_file SEQ SKIP LONG

    Input:
        prm_file      - parameter file for interferogram selection
        baseline_file - GMTSAR baseline table file
        modes         - [optional] list of modes to use. 
                          SEQ   - sequential pairs
                          SKIP  - skip pairs
                          FIRST - pairs with respect to first date
                          LAST  - pairs with respect to last date
                          REF   - pairs with respect to reference date
                          LONG  - long epoch pairs 
                          BL    - baseline-constrained pairs
                        Default is to use specification in PRM file.
    OUTPUT:
        baseline_plot.eps - plot of interferograms pairs (saved to disk)

        The following files are generated in for both YYYYMMDD_YYYYMMDD (file = dates) and 
        SLC (file intf) naming conventions:

        file.ALL   - list of all pairs formed
        file.SEQ   - dates.ALL subset for sequential pairs
        file.SKIP  - dates.ALL subset for skip pairs
        file.FIRST - dates.ALL subset for pairs with respect to first date
        file.LAST  - dates.ALL subset for pairs with respect to last date
        file.REF   - dates.ALL subset for pairs with respect to reference date
        file.LONG  - dates.ALL subset for long epoch pairs 
        file.BL    - dates.ALL subset for baseline-constrained pairs

    ------------------------------------------------------------------------------------ 
    """

    # For Option A, write parameter file
    if (len(sys.argv[1:4]) > 0) and all([arg.isdigit() for arg in sys.argv[1:4]]):

        # Get parameters
        DT_MIN    = sys.argv[1]
        DT_MAX    = sys.argv[2]
        BP_MAX    = sys.argv[3]
        file_name = sys.argv[4]

        if len(sys.argv) > 5: 
           DATE_REF = sys.argv[5]
        else:
           DATE_REF = 'None'

        # Get file text
        prm_text = get_prm_file(DT_MIN, DT_MAX, BP_MAX, DATE_REF)

        # Save to disk
        with open(f'{file_name}.PRM', 'w') as file:
            file.write(prm_text)
        print(f'{file_name}.PRM saved')
        sys.exit()

    # Option B, select interferograms 
    elif len(sys.argv) > 1:
        
        # Get arguments
        prm_file      = sys.argv[1];
        baseline_file = sys.argv[2];
        modes         = sys.argv[3:]

        # Read in baseline table
        baseline_table = load_baseline_table(baseline_file) 

        # Get pairs
        intf_inputs, intf_dates, subset_inputs, subset_dates, supermaster = select_pairs(baseline_table, prm_file, modes)

        # Write intferferogram list to use with GMTSAR scripts
        write_intf_list('intf.ALL', intf_inputs)

        # Write dates to list of interferogram directories to be generate=d
        write_intf_list('dates.ALL', [dates[0].strftime('%Y%m%d') + '_' + dates[1].strftime('%Y%m%d') for dates in intf_dates])

        # Also write interferogram subset lists
        for key in subset_inputs:
            write_intf_list('intf.' + key, subset_inputs[key])
            write_intf_list('dates.' + key, [dates[0].strftime('%Y%m%d') + '_' + dates[1].strftime('%Y%m%d') for dates in subset_dates[key]])

        # Make baseline plot 
        baseline_plot(subset_dates, baseline_table, supermaster=supermaster)

    # Return docstring otherwise
    else:
        print(main.__doc__)
        sys.exit()


# ========== FUNCTIONS ==========
def load_PRM(prm_file, var_in):
    """
    Read GMTASAR-style PRM file
    """

    # Intialize dictionary
    prm = {}

    # Set date format
    # date_format = '%Y/%m/%d'

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
                    try: # Try YYYY/MM/DD
                        val = dt.datetime.strptime(item[2], '%Y/%m/%d')
                    except ValueError:
                        try: # Try YYYYMMDD
                            val = dt.datetime.strptime(item[2], '%Y%m%d')                       
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
        if 'S1' in scene or 's1' in scene:
            for i in range(len(scene) - 8):
                tmp_str = scene[i:i + 8]
                if tmp_str.isdigit():
                    try:
                        dates.append(dt.datetime.strptime(tmp_str, '%Y%m%d'))
                        break
                    except ValueError:
                        continue
            # print(dates)

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


def select_pairs(baseline_table, prm_file, modes):
    """
    Select interferogmetric pairs based off of parameters specified in prm_file
    """

    # ---------- SET THINGS UP ----------
    # Get number of aquisitions
    N = len(baseline_table)  
    print()
    print('Number of SAR scenes =', N)

    # Get pair selection parameters
    selected_modes = {}
    default_modes  = ['SEQ', 'SKIP', 'FIRST', 'LAST', 'REF', 'LONG', 'BL',]

   
    for mode in default_modes:
        if len(modes) == 0:
            selected_modes[mode] = load_PRM(prm_file, mode)
        elif mode in modes:
            selected_modes[mode] = 1
        else: 
            selected_modes[mode] = 0

    # SEQ   = load_PRM(prm_file, 'SEQ')
    # SKIP  = load_PRM(prm_file, 'SKIP')
    # FIRST = load_PRM(prm_file, 'FIRST')
    # LAST  = load_PRM(prm_file, 'LAST')
    # REF   = load_PRM(prm_file, 'REF')
    # LONG  = load_PRM(prm_file, 'LONG')
    # BL    = load_PRM(prm_file, 'BL')

    # Load baseline parameters
    defaults = [0, 0, 0] # Default values
    BP_MAX   = load_PRM(prm_file, 'BP_MAX')
    DT_MIN   = load_PRM(prm_file, 'DT_MIN')
    DT_MAX   = load_PRM(prm_file, 'DT_MAX')

    # If any parameter is unspecified, instate default values
    for param, value, default in zip(['BP_MAX', 'DT_MIN', 'DT_MAX'], [BP_MAX, DT_MIN, DT_MAX], defaults):
        if value == None:
            print('{} not specified, default = {}'.format(param, default))
            param = default

    # Compute mean baseline
    Bp_mean = baseline_table['Bp'].mean()

    # Get supermaster scene
    DATE_REF = load_PRM(prm_file, 'DATE_REF');

    if DATE_REF == 'None':
        # Find scene with baseline closest to mean if no date is specified in PRM file
        supermaster_tmp = baseline_table[abs(baseline_table['Bp'] - Bp_mean) == min(abs(baseline_table['Bp'] - Bp_mean)) ]
        print()
        print('Using scene with baseline closest to stack mean ({} m):'.format(np.round(Bp_mean, 2)))
        print('Master date = {} '.format(pd.to_datetime(supermaster_tmp['date'].values[0]).strftime('%Y/%m/%d')))
        print('Baseline    = {} m'.format(np.round(supermaster_tmp['Bp'].values[0], 2)))
    
    elif pd.to_datetime(DATE_REF) in baseline_table['date'].values:
        print('DATE_REF = {}'.format(DATE_REF.strftime('%Y/%m/%d')))
        supermaster_tmp = baseline_table[baseline_table['date'].values == pd.to_datetime(DATE_REF)]

    else:
        print('Error! Reference date {} is not found in dataset'.format(DATE_REF))
        sys.exit()

    # Convert to dictionary
    supermaster   = {}
    i_supermaster = supermaster_tmp.index.values[0]

    for col in zip(supermaster_tmp.columns):
        supermaster[col[0]] = supermaster_tmp[col[0]].values[0]
            

    # ---------- INTERFEROGRAM SELECTION ----------
    # This portion of the code operates by 'turning on' elements of a NxN network matrix corresponding to all possible interferometric pairs
    # All values start 'off'

     # Initialize dictionary to contain a network matrix for each subset of interferograms to be made
    subset_IDs = {}

    print()
    print('Selected modes:')

    # 1) If SEQ is specified, select every sequential interferogram
    if bool(selected_modes['SEQ']) == True:
        print('SEQ   - sequential interferograms')

        ID_SEQ = np.zeros((N, N)) 

        for i in range(N):
            for j in range(N):
                # if np.mod(i, 1) == 0:    
                if abs(j - i) == 1:
                    ID_SEQ[i, j] = 1

        subset_IDs['SEQ'] = ID_SEQ

    # 2) If FIRST is specified, make all interferograms connecting to the 1st data take
    if bool(selected_modes['FIRST']) == True:
        print('FIRST - interferograms with respect to first date')

        ID_FIRST = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                if (i==0) or (j==0):
                    ID_FIRST[i, j] = 1

        subset_IDs['FIRST'] = ID_FIRST

    # 3) If LAST is specified, make all interferograms connecting to the 1st data take
    if bool(selected_modes['LAST']) == True:
        print('LAST  - interferograms with respect to last date')

        ID_LAST = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                if (i==N-1) or (j==N-1):
                    ID_LAST[i, j] = 1

        subset_IDs['LAST'] = ID_LAST

    # 4) If REF is specified, make all interferograms connecting to the 1st data take
    if bool(selected_modes['REF']) == True:
        print('REF   - interferograms with respect to reference date')

        ID_REF = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                if (i==i_supermaster) or (j==i_supermaster):
                    ID_REF[i, j] = 1

        subset_IDs['REF'] = ID_REF

    # 3) If SKIP is specified, make all 2nd-order pairs (skipping one scene)
    if selected_modes['SKIP'] > 0:
        print('SKIP  - all skip interferograms'.format(int(selected_modes['SKIP'])))
        ID_SKIP = np.zeros((N, N)) 

        for i in range(N):
            for j in range(N):
                # if np.mod(i, SKIP + 1) == 0:    
                if abs(j - i) == 2:
                    ID_SKIP[i, j] = 1

        subset_IDs['SKIP'] = ID_SKIP

        # # Generalized case:
        # # If SKIP is specified, select all n-order pairs
        # if SKIP > 0:
        #     print('all order-{} interferograms'.format(int(SKIP)))
        #     ID_SKIP = np.zeros((N, N)) 

        #     for i in range(N):
        #         for j in range(N):
        #             # if np.mod(i, SKIP + 1) == 0:    
        #             if abs(j - i) == SKIP:
        #                 ID_SKIP[i, j] = 1

        #     subset_IDs['SKIP_{}'.format(int(SKIP))] = ID_SKIP

    # 4) If LONG is specified, identify scenes which fit date range provided by LONG_START and LONG_END
    if bool(selected_modes['LONG']) == True:
        print('LONG  - long interferograms')

        ID_LONG = np.zeros((N, N)) 

        # Read dates 
        LONG_START = load_PRM(prm_file,'LONG_START');
        LONG_END   = load_PRM(prm_file,'LONG_END');

        # Or set defaults
        for param, value, default in zip(['LONG_START'], [LONG_START, LONG_END], [130, 280]):
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

                long_scenes_try = long_scenes[(long_scenes['date'] >= start0) & (long_scenes['date'] <= end0)]
                if len(long_scenes_try) > 2:
                  long_scenes0 = long_scenes_try
                year += 1

                # Once the year of the final scene is reached, if the scene is in or before the window, set it to be the reference image scene
                #if (year == baseline_table['date'].iloc[-1].year) and (baseline_table['date'][i].timetuple().tm_yday <= LONG_END):
                #    long_scenes0 = baseline_table[baseline_table['date'] == baseline_table['date'].iloc[-1]] # This is silly indexing, but it must be to work.

                #    #print(baseline_table['date'].iloc[-1])
                #    complete = True
                #    # Otherwise, continue for one more pair
                #print(long_scenes0)

                # If the later case is not triggered then the this one will be.
                if year > baseline_table['date'].iloc[-1].year: 
                    long_scenes0 = baseline_table.iloc[-1, :]
                    complete = True
            #print(long_scenes0)

            # Find index of scene within window that minimizes the perpendicular baseline with respect to the initial scene
            if type(long_scenes0['Bp']) is np.float64:
                j = long_scenes0.name
            else:
                j = (abs(baseline_table['Bp'][i] - long_scenes0['Bp'])).idxmin()

            # Turn element on in subset array
            ID_LONG[i, j] = 1

            # Reset index
            i = j

        subset_IDs['LONG'] = ID_LONG

    # 5) If BL is nonzero, use baseline constraints
    if selected_modes['BL'] > 0:

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

        if selected_modes['BL'] == 1:
            # Include all interferograms satisfying baseline constraints
            print('BL    - all intereferograms satisfying baseline constraints')
            subset_IDs['BL'] = ID_BASELINE

        elif selected_modes['BL'] == 2:
            # Select interferograms satisfying baseline constraints from previous selectionss
            print('BL    - Enforcing baseline constraints on previous selections')
            for key in subset_IDs.keys():
                subset_IDs[key] *= ID_BASELINE

        print('          Max. perp. baseline = {:.0f} m'.format(BP_MAX))
        print('          Min. epoch length   = {:.0f} days'.format(DT_MIN))
        print('          Max. epoch length   = {:.0f} days'.format(DT_MAX))
    
    # ---------- PREPARE OUTPUT LISTS ----------
    # Create initial and repeat matricies of dimension N x N
    initials = np.array(list(baseline_table['date'])).repeat(N).reshape(N, N)
    repeats  = np.array(list(baseline_table['date'])).repeat(N).reshape(N, N).T

    # Loop through subset dictionary to make individual subset interferograms] lists
    subset_inputs = {}
    subset_dates  = {}

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


def baseline_plot(subset_dates, baseline_table, supermaster={}, dates=True):

    """
    Make baseline netwwork plot for given set of interferograms

    INPUT:
    subset_dates   - 
    baseline_table - Dataframe containing appended GMTSAR baseline info table
    (supermaster   - supply dictionary containing info for the supermaster scene; will be plotted in red)
    """

    # Check for supermaster; set to empty if none is provided
    if len(supermaster) == 0:
        supermaster['dates'] = None
        supermaster['Bp']    = None

    # Initialize plot
    fig, ax = plt.subplots(figsize=(14, 8.2))

    # Plot pairs
    colors = ['k', 'steelblue', 'red', 'gold', 'green', 'mediumpurple']

    for i, key in enumerate(subset_dates.keys()):
        for j, date_pair in enumerate(subset_dates[key]):
            # Get corresponding baselines
            Bp_pair = [baseline_table[baseline_table['date'] == date]['Bp'].values for date in date_pair]

            if j == 0:
                label = key
            else:
                label = None

            ax.plot(date_pair, Bp_pair, c=colors[i], linewidth=1.5, zorder=0, label=label)


    # Plot nodes
    for i in range(len(baseline_table)):

        # Change settings if master
        if baseline_table['date'][i] == supermaster['date']:
            c = 'red'
            c_text = 'red'
            s = 30
        else:
            # c = 'C0'
            c = 'k'
            c_text = 'k'
            s = 20

        ax.scatter(baseline_table['date'][i], baseline_table['Bp'][i], marker='o', c=c, s=20)

        # Offset by 10 days/5 m for readability
        if dates:
            ax.text(baseline_table['date'][i] + 0.005*(baseline_table['date'].iloc[-1] - baseline_table['date'].iloc[0]), 
                    baseline_table['Bp'][i]   + 0.01*(baseline_table['Bp'].iloc[-1] - baseline_table['Bp'].iloc[0]), 
                    #baseline_table['date'][i].strftime('%Y/%m/%d'), 
                    baseline_table['date'][i].strftime('%m/%d'),
                    size=7, color=c_text, 
                    # bbox={'facecolor': 'w', 'pad': 0, 'edgecolor': 'w', 'alpha': 0.7}
                    )
    
    ax.legend()
    ax.set_ylabel('Perpendicular baseline (m)')
    ax.set_xlabel('Date')
    ax.tick_params(direction='in')
    plt.savefig('baseline_plot.eps')
    plt.show()


def get_prm_file(DT_MIN, DT_MAX, BP_MAX, DATE_REF):

    text  = '# ---------- Dates ---------- \n'
    text += 'DATE_START = 1900/01/01  # Lower bound on scene dates to use (YYYY/MM/DD) \n'
    text += 'DATE_END   = 2100/01/01  # Upper bound on scene dates to use (YYYY/MM/DD) \n'
    text += 'DATE_REF   = {}          # Date of master scene (YYYY/MM/DD) \n'.format(DATE_REF)
    text += '                         # Default: use scene closest to perpendicular baseline mean] \n'
    text += '\n'
    text += '# ---------- Pair types ---------- \n'
    text += '# For all options, set to 0 to not include in selection process \n'
    text += 'SEQ        = 0    # Generate sequential pairs, starting from initial scene \n'
    text += 'SKIP       = 0    # Generate 2nd-order pairs that skip one scene) \n'
    text += 'FIRST      = 0    # Generate pairs with respect to first acquisition \n'
    text += 'LAST       = 0    # Generate pairs with respect to last acquisition \n'
    text += 'REF        = 0    # Generate pairs with respect to reference acquisition \n'
    text += 'LONG       = 0    # Generate chain of 6-18 month pairs that connect the first and last dates \n' 
    text += 'LONG_START = 130  # Earliest Julian day to use in possible long pairs (1-366) \n'
    text += 'LONG_END   = 280  # Latest Julian day to use in possible long pairs (1-366) \n'
    text += '\n'
    text += '# ---------- Baseline constraints ---------- \n'
    text += '# Temporal and perpendicular baseline limits may be used in the following ways: \n'
    text += '# 1 - Make all interferograms which satisfy give constraints regardless of specification from SEQ, SKIP, or LONG \n'
    text += '# 2 - Use baseline constraints as a filter on previously specified pairs \n'
    text += '\n'
    text += 'BL      = 0   # Choose baseline constraint mode (1, 2, or 0 to not use) \n'
    text += 'BP_MAX  = {}  # Maximum perpendicular baseline (m) \n'.format(BP_MAX)
    text += 'DT_MIN  = {}  # Minimum interferogram epoch length (days) \n'.format(DT_MIN)
    text += 'DT_MAX  = {}  # Maximum interferogram epoch length (days) \n'.format(DT_MAX)

    return text



if __name__ == '__main__':
    main()
