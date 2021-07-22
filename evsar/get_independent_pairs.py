import sys
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

# ========== DRIVING METHOD ==========

def main():
    """
    ------------------------------------------------------------------------------------ 
    For given list of interferograms, generate baseline-minimizing/timespan maximizing 
    set of independent pairs.

    Usage:
        python get_baseline.py  

    Input:
        baseline_file - GMTSAR baseline table file

    Output:
        
    ------------------------------------------------------------------------------------ 
    """

    # Return docstring
    # if:
    #     print(main.__doc__)
    #     sys.exit()

    # intf_list     = sys.argv[1]
    baseline_file = sys.argv[1]

    DT_MIN = 300

    # Read in baseline table
    baseline_table = load_baseline_table(baseline_file) 
    scene_dates = [date.strftime('%Y%m%d') for date in baseline_table['date']]

    # Remove mean from scene baseline  and sort in descending order
    baseline_table['Bp']     -= baseline_table['Bp'].mean()
    baseline_table['Bp_abs']  = baseline_table['Bp'].abs()
    baseline_table = baseline_table.sort_values(by='Bp_abs', ascending=False)
    print(baseline_table)

    if (len(baseline_table) % 2) != 0:
        print(f'Odd number of scenes ({len(baseline_table)}), one scene will be skipped')


    # Starting from least-optimally oriented scene, make independent, baseline-minimizing pairs
    selected_dates = []
    selected_intfs = []

    baseline_table_avail = baseline_table

    for i, scene in baseline_table.iterrows():
        
        if i not in baseline_table_avail.axes[0]:
            continue
        else:

            # Identify possibly pair scenes based on epoch
            dt0  = np.array([dt.days for dt in baseline_table_avail['date'] - scene['date']])
            baseline_table0 = baseline_table_avail[np.abs(dt0) > DT_MIN]
            # print(baseline_table0.axes)

            if len(baseline_table0) == 0:
                print(scene['date'].strftime('%Y%m%d'), 'excluded') 
            else:   
                # Calulate baselines
                dBp0 = np.array(baseline_table0['Bp'] - scene['Bp'])

                # Select optimal pair date
                pair_date = baseline_table0['date'].iloc[np.argmin(np.abs(dBp0))]
                
                dt = (pair_date - scene['date']).days

                if dt > 0:
                    date1 = scene['date'].strftime('%Y%m%d')
                    idx1  = i
                    date2 = pair_date.strftime('%Y%m%d')
                    idx2  = baseline_table0.iloc[np.argmin(np.abs(dBp0))].name
                else:
                    date1 = pair_date.strftime('%Y%m%d')
                    idx1  = baseline_table0.iloc[np.argmin(np.abs(dBp0))].name
                    date2 = scene['date'].strftime('%Y%m%d')
                    idx2  = i

                intf_str = date1 + '_' + date2
                print(f'{intf_str}:    {np.min(np.abs(dBp0)):.1f} m     {abs(dt)} days')

                # Remove paired scenes from list
                baseline_table_avail = baseline_table_avail.drop(idx1)
                baseline_table_avail = baseline_table_avail.drop(idx2)

                selected_dates.append(intf_str)
                selected_intfs.append(baseline_table['scene_id'][idx1] + ':' + baseline_table['scene_id'][idx2])
    
    write_intf_list('intf.IND', selected_intfs)
    write_intf_list('dates.IND', selected_dates)

    baseline_plot('stack', {'IND': selected_dates}, baseline_table)

# inputs.append(baseline_table['scene_id'][i] + ':' + baseline_table['scene_id'][j]
    # intf_list     = sys.argv[1]
    # baseline_file = sys.argv[2]

    # # Get paths and dates
    # date_pairs = np.array(read_list(intf_list))
    # intf_dates = np.array([[date.strftime('%Y%m%d') for date in date_pairs[:, 0]], [date.strftime('%Y%m%d') for date in date_pairs[:, 1]]]).T

    # # Read in baseline table
    # baseline_table = load_baseline_table(baseline_file) 
    # scene_dates = [date.strftime('%Y%m%d') for date in baseline_table['date']]

    # # Get mean scene baseline
    # Bp_mean = baseline_table['Bp'].mean()
    # Bp_std  = baseline_table['Bp'].std()

    # # print(baseline_table['date'].strftime('%Y'))

    # # Calculate baselines
    # Bp = np.zeros(len(date_pairs))

    # for i, dates in enumerate(date_pairs):
    #     date0 = dates[0].strftime('%Y%m%d')
    #     date1 = dates[1].strftime('%Y%m%d')

    #     Bp[i] = baseline_table['Bp'].iloc[scene_dates.index(date1)] - baseline_table['Bp'].iloc[scene_dates.index(date0)]


    # # Get uses of each date
    # uses = np.zeros(len(scene_dates))

    # for i, date in enumerate(scene_dates):
    #     uses[i] = np.count_nonzero(intf_dates == date)

    # final_intfs = []





def read_list(intf_list):
    """
    Extract date pairs from list of interferograms.
    """

    date_pairs = []

    with open(intf_list, 'r') as file:
        for intf in file:
            # Slice lines to exclude '\n' statement
            date_pairs.append([dt.datetime.strptime(date[:8], '%Y%m%d') for date in intf.split('_')])

    return date_pairs


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


def baseline_plot(prm_file, subset_dates, baseline_table, supermaster={}):

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
    fig, ax = plt.subplots(figsize=(10,6))

    # Plot pairs
    colors = ['k', 'steelblue', 'tomato', 'gold']


    for i, key in enumerate(subset_dates.keys()):
        for j, date_pair in enumerate(subset_dates[key]):
            # Get corresponding baselines
            Bp_pair = [baseline_table[baseline_table['date'] == date]['Bp'].values for date in date_pair.split('_')]

            if j == 0:
                label = key
            else:
                label = None

            ax.plot([dt.datetime.strptime(date, '%Y%m%d') for date in date_pair.split('_')], Bp_pair, c=colors[i], linewidth=2, zorder=0, label=label)


    # Plot nodes
    for i in range(len(baseline_table)):

        # Change settings if master
        # if baseline_table['date'][i] == supermaster['date']:
        if baseline_table['date'][i] == 0:
            c = 'r'
            c_text = 'r'
            s = 30
        else:
            # c = 'C0'
            c = 'k'
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
    
    ax.legend()
    ax.set_ylabel('Perpendicular baseline (m)')
    ax.set_xlabel('Date')
    ax.tick_params(direction='in')
    plt.savefig(f'baseline_plot_{prm_file[:-4]}.eps')
    plt.show()


if __name__ == '__main__':
    main()