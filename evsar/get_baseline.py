#!/home/class239/anaconda3/bin/python3
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import sys

# ========== SPECIFY FILEPATHS ==========

def main():
    """
    Generate list of interferograms to process and baseline plot given baseline table and processing parameters
    
    Usage: get_baseline.py prm_file baseline_file
    
    INPUT:
      prm_file - parameter (PRM) file containing date and baseline values for interferogram selection
      baseline_file - GMTSAR baseline table file
    
    OUTPUT:
      short.dat - list of interferograms in YYYYMMDD_YYYYMMDD format
        Ex: 20150126_20150607')
            20150126_20150701')
            20150126_20150725')
            20150126_20150818')
    
      intf.in - list of interferogram pairs in SLC naming convention, for input into GMTSAR interferogram scripts
        Ex: S1A20150126_ALL_F1:S1A20150607_ALL_F1
            S1A20150126_ALL_F1:S1A20150701_ALL_F1
    
      Note: subsets of these will be generated which correspond to the selection parameters provided in the prm_file
        Ex:
        intf.in.sequential for SEQUENTIAL = True
        intf.in.skip_2 for Skip = 2
        intf.in.y2y for Y2Y_INTFS  = True

      baseline_plot.eps - plot of interferograms satisfying baseline constraints
      """

    # Return docstring if arguments unspecified
    if len(sys.argv) < 3:
        print(main.__doc__)
        sys.exit()


    # Get arguments
    prm_file = sys.argv[1];
    baseline_file = sys.argv[2];
    print(baseline_file)

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
        # print('Error: {} not found in {}'.format(var_in, prm_file))
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
        if 'S1A' in scene or 'S1A' in scene:
            for i in range(len(scene) - 8):
                tmp_str = scene[i:i + 8]
                if tmp_str.isdigit():
                    try:
                        dates.append(dt.datetime.strptime(tmp_str, '%Y%m%d'))
                    except ValueError:
                        continue

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
    Write list of interferograms to file_name specified.
    """

    with open(file_name, 'w') as file:
        for intf in intf_list:
            file.write(intf + '\n')


def select_pairs(baseline_table, prm_file):

    # ---------- SET THINGS UP ----------
    # Get number of aquisitions
    N = len(baseline_table)  
    print()
    print('Number of SAR scenes:', N)

    # Check pair selection parameters
    SEQUENTIAL = load_PRM(prm_file, 'SEQUENTIAL')
    SKIP_N     = load_PRM(prm_file, 'SKIP_N')
    Y2Y_INTFS  = load_PRM(prm_file, 'Y2Y_INTFS')

    # Load baseline parameters
    defaults = [0, 0, 0] # Default values
    Bp_max   = load_PRM(prm_file,'BP_MAX');
    t_min    = load_PRM(prm_file, 'DT_MIN');
    t_max    = load_PRM(prm_file, 'DT_MAX');

    # If any parameter is unspecified, instate default values
    for param, value, default in zip(['BP_MAX', 'DT_MIN', 'DT_MAX'], [Bp_max, t_min, t_max], defaults):
        if value == None:
            print('{} not specified, default = {}'.format(param, default))
            param = default

    # Compute mean baseline
    Bp_mean = baseline_table['Bp'].mean()

    # Get supermaster scene
    DATE_MASTER = load_PRM(prm_file, 'DATE_MASTER');

    if DATE_MASTER == None:
        # Find scene with baseline closest to mean if no date is specified in PRM file
        supermaster_tmp = baseline_table[abs(baseline_table['Bp'] - Bp_mean) == min(abs(baseline_table['Bp'] - Bp_mean)) ]
        print('{} not specified, using scene with baseline closest to stack mean ({} m):'.format('DATE_MASTER', np.round(Bp_mean, 2)))
        # print(  supermaster_tmp['date'].values[0].strftime('%Y%m%d', supermaster_tmp['Bp'].values[0]))
        # print(  supermaster_tmp['date'], supermaster_tmp['Bp'])
        print( 'Date:', pd.to_datetime(supermaster_tmp['date'].values[0]).strftime('%Y/%m/%d'), '    Baseline: ', np.round(supermaster_tmp['Bp'].values[0], 2), 'm')
    else:
        print('Using supermaster date {}'.format(DATE_MASTER))
        supermaster_tmp = baseline_table[baseline_table['date'] == DATE_MASTER]

    # Convert to dictionary
    supermaster = {}
    for col in zip(supermaster_tmp.columns):
        supermaster[col[0]] = supermaster_tmp[col[0]].values[0]
        

    # ---------- ACTUAL INTERFEROGRAM SELECTION ----------
    # This portion of the code operates by 'turning on' elements of a NxN network matrix corresponding to all possible interferometric pairs
    # All values start 'off'

     # Initialize dictionary to contain a network matrix for each subset of interferograms to be made
    subset_IDs = {}

    # If Y2Y_INTFS is specified, identify scenes which fit date range provided by Y2Y_START and Y2Y_END
    if Y2Y_INTFS > 0:
        ID_Y2Y_INTFS = np.zeros((N, N)) 

        # Read dates 
        Y2Y_START = load_PRM(prm_file,'Y2Y_START');
        Y2Y_END = load_PRM(prm_file,'Y2Y_END');

        # Or set defaults
        for param, value, default in zip(['Y2Y_START'], [Y2Y_START, Y2Y_END], [dt.datetime(0,0,0,0,0,0), datetime.today()]):
            if value == None:
                print('{} not specified, default = {}'.format(param, default))
                param = default

        # Identify dates in stack that fall within range
        y2y_scenes = baseline_table[[(date.month >= Y2Y_START) & (date.month <= Y2Y_END) for date in baseline_table['date']]]

        # Calculate perpendicular baselines for all pairs
        for initial_id, date1 in zip(y2y_scenes.index, y2y_scenes['date']):
            for repeat_id, date2 in zip(y2y_scenes.index, y2y_scenes['date']):

                # Select year to year pairs
                if (date1.year != date2.year) and (date1 < date2):
                    continue
                    # ACTUALLY WRITE THIS SOMETIME ELLIS

    # OLD
    # # If SEQ_INTFS is specified, Select every nth pair using incremement specified by SEQ_INTFS
    #   if SEQ_INTFS > 0:
    #       print('Making sequential interferograms of order: {}'.format(np.arange(0, SEQ_INTFS+1)[1:]))
    #       for n in range(int(SEQ_INTFS)):
    #           inc = n + 1
    #           for i in range(N):
    #               for j in range(N):
    #                   if np.mod(i, inc) == 0:    
    #                       if abs(j - i) == inc:
    #                           ID[i, j] = 1

    # If SEQUENTIAL is specified, make every sequential interferogram
    if bool(SEQUENTIAL) == True:
        ID_SEQUENTIAL = np.zeros((N, N)) 
        print('Making sequential interferograms')
        for i in range(N):
            for j in range(N):
                # if np.mod(i, 1) == 0:    
                if abs(j - i) == 1:
                    ID_SEQUENTIAL[i, j] = 1

        subset_IDs['sequential'] = ID_SEQUENTIAL


    # If SKIP_N is specified, skip every nth pair and make sequential interferograms
    if SKIP_N > 0:
        print('Making sequential interferograms with skip = {}'.format(int(SKIP_N)))
        ID_SKIP_N = np.zeros((N, N)) 
        
        for i in range(N):
            for j in range(N):
                # if np.mod(i, SKIP_N + 1) == 0:    
                if abs(j - i) == SKIP_N + 1:
                    ID_SKIP_N[i, j] = 1

        subset_IDs['skip_{}'.format(int(SKIP_N))] = ID_SKIP_N


    # # If SKIP_N is specified, skip every nth pair and make sequential interferograms
    # if SKIP_N > 0:
    #     print('Making sequential interferograms with skip = {}'.format(int(SKIP_N)))
    #     ID_SKIP_N = np.zeros((N, N)) 
    #     for i in range(N):
    #         for j in range(N):
    #             if np.mod(i, SKIP_N + 1) == 0:    
    #                 if abs(j - i) == SKIP_N + 1:
    #                     ID_SKIP_N[i, j] = 1

    #         subset_IDs['skip_{}'.format(int(SKIP_N))] = ID_SKIP_N

    # # OLD
    # # Create initial and repeat matricies of dimension N x N
    # initials = np.array(list(baseline_table['date'])).repeat(N).reshape(N, N)
    # repeats = np.array(list(baseline_table['date'])).repeat(N).reshape(N, N).T


    # # Loop through indicies to get pair dates
    # intf_list = []
    # intf_dates = []

    # for i in range(len(ID)):
    #     for j in range(len(ID[0])):
    #         if ID[i, j] == 1 and initials[i, j] < repeats[i, j]:  # We only want the upper half of the matrix, so ignore intf pairs where 'initial' comes after 'repeat'
    #             # intf_list.append('S1_' + initials[i, j].strftime('%Y%m%d') + '_ALL_F2:S1_' + repeats[i, j].strftime('%Y%m%d') + '_ALL_F2')
    #             intf_list.append(baseline_table['scene_id'][i] + ':' + baseline_table['scene_id'][j])
    #             intf_dates.append([baseline_table['date'][i],baseline_table['date'][j]])


    # ---------- PREPARE OUTPUT LISTS ----------
    # Create initial and repeat matricies of dimension N x N
    initials = np.array(list(baseline_table['date'])).repeat(N).reshape(N, N)
    repeats = np.array(list(baseline_table['date'])).repeat(N).reshape(N, N).T

    # Loop through subset dictionary to make individual subset interferograms] lists
    subset_inputs = {}
    subset_dates = {}

    for key in subset_IDs.keys():
        inputs = []
        dates = []

        for i in range(len(subset_IDs[key])):
            for j in range(len(subset_IDs[key][0])):
                if subset_IDs[key][i, j] == 1 and initials[i, j] < repeats[i, j]:  # We only want the upper half of the matrix, so ignore intf pairs where 'initial' comes after 'repeat'
                    inputs.append(baseline_table['scene_id'][i] + ':' + baseline_table['scene_id'][j])
                    dates.append([baseline_table['date'][i],baseline_table['date'][j]])

        subset_inputs[key] = inputs 
        subset_dates[key]  = dates


    # Aggregate to master lists
    intf_inputs = []
    intf_dates = []

    for key in subset_inputs:
        intf_inputs.extend(subset_inputs[key])

    for key in subset_dates:
        intf_dates.extend(subset_dates[key])

    # Get number of interferogams to make
    n = len(intf_inputs)
    print('Number of interferograms: {}'.format(n))


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
        print(date_pair, Bp_pair)

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

        # Offset by 10 days/5 m for asthetics
        ax.text(baseline_table['date'][i] + dt.timedelta(days=10), 
                baseline_table['Bp'][i] + 5, 
                baseline_table['date'][i].strftime('%Y/%m/%d'), 
                size=8, color=c_text)
    
    ax.set_ylabel('Perpendicular baseline (m)')
    ax.set_xlabel('Date')
    plt.savefig('baseline_plot.eps')
    plt.show()


if __name__ == '__main__':
    main()

