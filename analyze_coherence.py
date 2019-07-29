#!/bin/python
import numpy as np
import glob as glob
import sys
import datetime as dt
import subprocess
import netcdf_read_write
import readmytupledata
import matplotlib.pyplot as plt
import collections
import utilities_S1
import new_baseline_table
import math
from matplotlib import cm

"""
Written by Ellis Vavra, July 21, 2019

Want to be able to take a list of all interferogram directories and calculate the (1) mean coherence for each intf and (2) how many pixels are above/below a certain threshold (make a plot?)
"""

# -------------------------------- TOP LEVEL DRIVER -------------------------------- #


def driver():

    # homedir = '/Volumes/EV-Drive/Lorax/insar/des/f2/intf_all_*/20*_20*/'
    # homedir = '/Users/ellisvavra/Thesis/insar/des/f2/intf_all/20*_20*/'
    # homedir = '/Users/ellisvavra/Desktop/Thesis/S1_Processing'
    homedir = '/Users/ellisvavra/Thesis/insar/des/f2/intf_all/*/'
    filetype = 'corr.grd'
    skip_list = []
    area = [0, 20000, 0, 6000]
    baseline_table = 'baseline_table.dat'
    stage = 'original'
    intf_table = 'coherence_analysis_downsampled.dat'

    corr_min = 0.2
    corr_max = 1.0
    output_name = 'unwrap_intfs_0.2.in'

    # # Get paths to target .grds
    # path_list = getPaths(homedir, filetype)

    # # Calulate means
    # means = getMeans(path_list, area, filetype)

    # # Write data to interferogram tuple
    # iTuple = intfTuple(path_list, means, baseline_table, filetype, stage)

    # # Save tuple to .dat file for safe keeping
    # writeData(iTuple, intf_table)

    # # Make coherence plot
    # # iTuple = readIntfTable(intf_table)
    # plotIntfCoherence(iTuple, baseline_table, filetype, stage)

    # # Count number of times used
    # # iTuple = readIntfTable(intf_table)
    # scene_dates, date_sums = count(iTuple)

    # # Get mean coherence for each scene
    # # iTuple = readIntfTable(intf_table)
    # scene_dates, mean_scene_coherence = sceneCorr(iTuple)

    # # Plot mean scene coherence
    # plotSceneCoherence(scene_dates, mean_scene_coherence, date_sums)

    # # Plot interferogram coherence distribution
    # iTuple = readIntfTable(intf_table)
    # plotCorrHist(iTuple)

    # Filter intferogram list based off of coherence
    iTuple = readIntfTable(intf_table)
    intf_list = sortFromCorr(iTuple, corr_min, corr_max, output_name)
    writeIntfList(intf_list, output_name)

# -------------------------------- CONFIGURE -------------------------------- #


def getPaths(homedir, filetype):
        # Find all .grd files of given type
    print("Searching for: " + homedir + filetype)
    path_list = glob.glob(homedir + filetype)
    print(path_list)

    return path_list


def readIntfTable(intf_table):
    # Read intf_table.dat to intfTuple

    iTuple = collections.namedtuple('intf_data', ['paths', 'date_pairs', 'master', 'slave', 'temporal_baseline', 'orbital_baseline', 'mean_coherence'])

    paths = []
    date_pairs = []
    master = []
    slave = []
    temporal_baseline = []
    orbital_baseline = []
    mean_coherence = []

    with open(intf_table, "r") as intf_dat:
        for line in intf_dat:
            temparray = line.split()
            paths.append(temparray[0])
            date_pairs.append(temparray[1])
            master.append(dt.datetime.strptime(temparray[2], "%Y%m%d"))
            slave.append(dt.datetime.strptime(temparray[3], "%Y%m%d"))
            temporal_baseline.append(float(temparray[4]))
            orbital_baseline.append(float(temparray[5]))
            mean_coherence.append(float(temparray[6]))

        myData = iTuple(paths=paths, date_pairs=date_pairs, master=master, slave=slave, temporal_baseline=temporal_baseline, orbital_baseline=orbital_baseline, mean_coherence=mean_coherence)

    return myData


# -------------------------------- ANALYSIS -------------------------------- #

def getMeans(path_list, area, filetype):
    # Use GMT to calculate means of .grd files
    means = []

    for path in path_list:
        # Calulate mean and save to temporary file
        newFilePath = path[0:(len(path) - len(filetype))] + "mean_corr.grd"

        subprocess.call("gmt grdmath " + path + " MEAN = " + newFilePath, shell=True)

        # Extract mean
        mean_value = float(subprocess.check_output("gmt grdinfo " + newFilePath + " | grep z_min | awk '{print $3}'", shell=True))

        # Add to means list
        means.append(mean_value)

        print(path + ": " + str(mean_value))

    return means


def count(iTuple):
    # Count how many time each scene is used in interferograms
    print('Masters: ')
    for date in iTuple.master:
        print(date.strftime("%Y-%m-%d"))
    print()
    print('Slaves: ')
    for date in iTuple.slave:
        print(date.strftime("%Y-%m-%d"))
    print()

    # Initiate
    all_dates = iTuple.master + iTuple.slave
    print('All dates: ')
    for date in all_dates:
        print(date.strftime("%Y-%m-%d"))
    print()

    scene_dates = []
    date_sums = []

    # Get list of unique dates
    print('Unique dates:')
    for date in all_dates:
        if date not in scene_dates:
            scene_dates.append(date)
            print(scene_dates[-1].strftime("%Y-%m-%d"))

    scene_dates.sort()
    print()

    # Count number of times each date is used
    print('Number of times used:')
    for date in scene_dates:
        date_sums.append(all_dates.count(date))
        print(date.strftime("%Y-%m-%d") + ": " + str(date_sums[-1]))
    print()

    return scene_dates, date_sums


def sceneCorr(iTuple):
    # Determine mean coherence of all interferograms each scene is used in

    # Combine date lists
    all_dates = iTuple.master + iTuple.slave
    print('All dates: ')
    for date in all_dates:
        print(date.strftime("%Y-%m-%d"))
    print()

    # Get list of unique dates
    scene_dates = []
    print('Unique dates:')
    for date in all_dates:
        if date not in scene_dates:
            scene_dates.append(date)
            print(scene_dates[-1].strftime("%Y-%m-%d"))

    scene_dates.sort()
    print()

    # Make a corrTuple
    corrTuple = collections.namedtuple('coherence_data', ['dates', 'master_indicies', 'slave_indicies', 'mean_scene_coherence'])

    master_indicies = []
    slave_indicies = []

    print('Number of interferograms: ' + str(len(iTuple.master)))
    print('Number of scenes: ' + str(len(scene_dates)))
    print()

    # Find indicies of all usages of each SAR scene by querying the master and slave lists
    for date in scene_dates:
        print('Finding occurences of ' + date.strftime("%Y-%m-%d"))
        if date in iTuple.master:
            temp_master_index = [i for i, x in enumerate(iTuple.master) if x == date]
            master_indicies.append(temp_master_index)
            print(temp_master_index)
        else:
            master_indicies.append([])
            print(temp_master_index)

        if date in iTuple.slave:
            temp_slave_index = [i for i, x in enumerate(iTuple.slave) if x == date]
            slave_indicies.append(temp_slave_index)
            print([])
        else:
            slave_indicies.append([])
            print([])
    print()

    # Calulate mean coherence for each scene
    mean_scene_coherence = []

    print('Mean scene coherence: ')
    for i in range(len(scene_dates)):
        master_coherences = [iTuple.mean_coherence[j] for j in master_indicies[i]]
        slave_coherences = [iTuple.mean_coherence[j] for j in slave_indicies[i]]
        mean_coherence = sum(master_coherences + slave_coherences) / len(master_coherences + slave_coherences)
        mean_scene_coherence.append(mean_coherence)
        print(mean_coherence)
    print()

    return scene_dates, mean_scene_coherence


def sortFromCorr(iTuple, corr_min, corr_max, output_name):
    # Find indicies of all interferograms who meet the minimum coherence threshold
    intf_list = []

    print()
    print('Interferograms with coherence between ' + str(corr_min) + ' and ' + str(corr_max))

    for i in range(len(iTuple.paths)):
        if iTuple.mean_coherence[i] >= corr_min and iTuple.mean_coherence[i] < corr_max:
            intf_list.append(iTuple.date_pairs[i])
            print(iTuple.date_pairs[i] + ': ' + str(iTuple.mean_coherence[i]))

    # print(intf_list)
    return intf_list


# -------------------------------- OUTPUT -------------------------------- #

def intfTuple(path_list, means, baseline_table, filetype, stage):
    # Create intfTuple from scratch

    iTuple = collections.namedtuple('intf_data', ['paths', 'date_pairs', 'master', 'slave', 'temporal_baseline', 'orbital_baseline', 'mean_coherence'])

    date_pairs = []
    master = []
    slave = []

    # ASSUMES THAT GMTSAR INTERFEROGRAM DIRECTORIES ARE IN ORIGINAL NAMING CONVENTION
    if stage == 'original':
        for line in path_list:

            print('Searching: */' + line[(-16 - len(filetype)): (-len(filetype) - 1)] + "/*SLC")
            SLCs = glob.glob("*/" + line[(-16 - len(filetype)): (-len(filetype) - 1)] + "/*SLC")
            print(SLCs)

            date_pairs.append(line[(-16 - len(filetype)): (-len(filetype) - 1)])

            # Add their dates to the master and slave image lists
            # S1_20190122_ALL_F2.SLC
            len(line)
            master.append(dt.datetime.strptime(SLCs[0][-19: -11], "%Y%m%d"))
            slave.append(dt.datetime.strptime(SLCs[1][-19: -11], "%Y%m%d"))

    # ASSUMES THAT GMTSAR INTERFEROGRAM DIRECTORIES HAVE ALREADY BEEN RENAMED TO YYYYMMDD CONVENTION
    elif stage == 'renamed':
        for line in path_list:
            # Get date pair string from path
            date_pairs.append(line[0: 17])
            # Get datetime objects for master and slave dates
            master.append(dt.datetime.strptime(line[0: 8], "%Y%m%d"))
            slave.append(dt.datetime.strptime(line[9: 17], "%Y%m%d"))

    # Compute perpendicular (B_p) and temporal (B_t) baselines for pair
    orbit, dt_dates, jday, blpara, blperp, datelabels = new_baseline_table.readBaselineTable(baseline_table)

    print("Datetime dates:")
    print(dt_dates)
    # Convert date list to strings for searching with master and slave lists
    str_dates = []

    print("String dates: ")
    for date in dt_dates:
        str_dates.append(date.strftime("%Y%m%d"))
        print(str_dates[-1])
    # print(str_dates)

    # Calulate baselines
    temporal_baseline = []
    orbital_baseline = []

    for i in range(len(master)):

        print('Searching for ' + master[i].strftime("%Y%m%d"))

        # Find master index
        Mi = str_dates.index(master[i].strftime("%Y%m%d"))
        # Find slave index
        Si = str_dates.index(slave[i].strftime("%Y%m%d"))
        # Calculate baselines
        B_p = abs(blperp[Si] - blperp[Mi])
        B_t_dt = dt_dates[Si] - dt_dates[Mi]
        B_t = B_t_dt.days

        print("For " + master[i].strftime("%Y%m%d") + "_" + slave[i].strftime("%Y%m%d") + ": B_t = " + str(B_t) + ', ' + "B_p = " + str(B_p))

        temporal_baseline.append(B_t)
        orbital_baseline.append(B_p)

    myData = iTuple(paths=path_list, date_pairs=date_pairs, master=master, slave=slave, temporal_baseline=temporal_baseline, orbital_baseline=orbital_baseline, mean_coherence=means)

    print(myData)
    return myData


def writeData(iTuple, intf_table):
    with open(intf_table, 'w') as newTable:
        print("len(iTuple.paths) = " + str(len(iTuple.paths)))
        for i in range(len(iTuple.paths)):
            print(iTuple.paths[i])
            print(iTuple.date_pairs[i])
            print(iTuple.master[i].strftime("%Y%m%d"))
            print(iTuple.slave[i].strftime("%Y%m%d"))
            print(iTuple.mean_coherence[i])
            newTable.write(iTuple.paths[i] + " " + iTuple.date_pairs[i] + " " + iTuple.master[i].strftime("%Y%m%d") + " " + iTuple.slave[i].strftime("%Y%m%d") + " " + str(iTuple.temporal_baseline[i]) + " " + str(iTuple.orbital_baseline[i]) + " " + str(iTuple.mean_coherence[i]) + '\n')

    print()
    print('File written:')
    print(newTable)

    return newTable


def writeIntfList(intf_list, output_name):
    print()
    print('Writing...')

    with open(output_name, 'w') as newList:
        for intf in intf_list:
            print(intf)
            #newList.write(intf + '\n')

    print()
    print('File written:')
    print(newList)


def plotIntfCoherence(iTuple, baseline_table, filetype, stage):
    # Read in data data

    # Establish figure
    fig, ax = plt.subplots()

    # Get range of baseline lengths
    baseline_range = list(range(0, int(math.floor(max(iTuple.orbital_baseline)))))
    print('Baseline range = 0-' + str(math.floor(max(iTuple.orbital_baseline))) + 'm')
    n = len(baseline_range)  # Number of colors
    viridis = cm.get_cmap('plasma', n)
    print(viridis)

    for i in range(len(iTuple.mean_coherence)):
        # Get color coefficient
        line_color = np.floor(iTuple.orbital_baseline[i]) / n
        print(line_color)

        plt.plot([iTuple.master[i], iTuple.slave[i]], [iTuple.mean_coherence[i], iTuple.mean_coherence[i]], zorder=3, color=viridis(line_color))

    # Figure features
    plt.grid(axis='x', zorder=1)
    plt.xlim(min(iTuple.master, default=dt.datetime.strptime('20141201', "%Y%m%d")) - dt.timedelta(days=5), max(iTuple.slave, default=dt.datetime.strptime('20190801', "%Y%m%d")) + dt.timedelta(days=5))
    plt.ylim(0, 1)
    plt.xlabel('Date')
    plt.ylabel('Mean coherence')

    plt.show()


def plotSceneCoherence(scene_dates, mean_scene_coherence, date_sums):
    # Establish figure
    fig, ax = plt.subplots()

    plt.scatter(scene_dates, mean_scene_coherence, c=date_sums, zorder=3)

    # Plot scene IDs
    n = np.arange(1, len(scene_dates) + 1)
    (n)

    # Figure features
    plt.grid(axis='x', zorder=1)
    plt.grid(axis='y', zorder=1)
    plt.xlim(min(scene_dates, default=dt.datetime.strptime('20141201', "%Y%m%d")) - dt.timedelta(days=30), max(scene_dates, default=dt.datetime.strptime('20190801', "%Y%m%d")) + dt.timedelta(days=30))
    plt.ylim(0, np.ceil(max(mean_scene_coherence) * 10) / 10)
    plt.xlabel('Date')
    plt.ylabel('Mean SAR scene coherence ')

    plt.yticks(np.arange(0, np.ceil(max(mean_scene_coherence) * 10) / 10 + 0.1, 0.1))

    cticks = np.arange(0, max(date_sums) + 1, 4)
    cbar = plt.colorbar(ticks=cticks)
    cbar.set_label("Number of interferograms")

    plt.show()


def plotCorrHist(iTuple):
    # Establish figure
    fig, ax = plt.subplots()

    plt.hist(iTuple.mean_coherence, bins=100)

    # Figure features
    plt.grid(axis='x', zorder=1)
    plt.xlabel('Mean coherence ')
    plt.ylabel('Number of interferograms')

    plt.show()


if __name__ == "__main__":
    driver()
