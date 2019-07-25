#!/usr/bin/python
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
from cycler import cycler
import math
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

"""
Written by Ellis Vavra, July 21, 2019

Want to be able to take a list of all interferogram directories and calculate the (1) mean coherence for each intf and (2) how many pixels are above/below a certain threshold (make a plot?)

INPUT:
- List of interferogram directories
- List of interferograms to skip
- Coherence threshold
- Count Nans

OUTPUT:
- list of mean coheence for each interferogram pair
-
"""

# -------------------------------- TOP LEVEL DRIVER -------------------------------- #

def driver():
    
    homedir = '/Volumes/EV-Drive/Lorax/insar/des/f2/intf_all_*/20*_20*/'
    # homedir = '/Users/ellisvavra/Thesis/insar/des/f2/intf_all/20*_20*/'
    filetype = 'corr.grd'
    skip_list = []
    area = [0,20000,0,6000]
    baseline_table = 'baseline_table.dat'
    stage = 'original'
    corr_table = 'coherence_analysis.dat'

    # Find all .grd files of given type

    print("Searching for: " + homedir + filetype)
    path_list = glob.glob(homedir + filetype)
    print(path_list)

    # Calulate means
    means = getMeans(path_list, area, filetype)

    # Write to tuple
    corr_data = coherenceTuple(path_list, means, baseline_table, filetype, stage)

    # Save tuple to .dat file
    writeData(corr_data)

        
    # Make coherence plot
    plotCoherence(corr_table, baseline_table, filetype, stage)



# -------------------------------- INPUT -------------------------------- #

def readCorrTable(corr_table, baseline_table, filetype, stage):

    paths = []
    date_pairs = []
    master = []
    slave = []
    temporal_baseline = []
    orbital_baseline = []
    mean_coherence = []

    with open(corr_table, "r") as corr_dat:
        for line in corr_dat:
            temparray = line.split()
            paths.append(temparray[0])
            date_pairs.append(temparray[1])
            master.append(dt.datetime.strptime(temparray[2], "%Y%m%d"))
            slave.append(dt.datetime.strptime(temparray[3], "%Y%m%d"))
            temporal_baseline.append(float(temparray[4]))
            orbital_baseline.append(float(temparray[5]))
            mean_coherence.append(float(temparray[6]))

            #paths, date_pairs, master, slave, temporal_baseline, orbital_baseline, mean_coherence = np.loadtxt(corr_table, usecols=(2, 3, 4, 5), unpack=True)

    corrTuple = coherenceTuple(paths, mean_coherence, baseline_table, filetype, stage)

    return corrTuple


    with open(input, 'r') as tempfile:
        for line in tempfile:
            temparray = line.split()
            print(temparray)
            orbit.append(temparray[0])
            jday.append(float(temparray[2]))
            blpara.append(float(temparray[3]))
            blperp.append(float(temparray[4]))


# OLD
# def readCorrTable(datafile, filetype, stage):
#     paths, date_pairs, master, slave, mean_coherence = np.loadtxt(datafile, unpack=True)

#     corrTuple = coherenceTuple(paths, mean_coherence, filetype, stage)

#     return corrTuple


# -------------------------------- ANALYSIS -------------------------------- #

def getMeans(path_list, area, filetype):
    # Use GMT to calculate means of .grd files

    means = []

    for path in path_list:
        # Calulate mean and save to temporary file
        newFilePath = path[0:(len(path) - len(filetype))] + "mean_corr.grd"
        #subprocess.call("gmt grdmath " + path + " MEAN = " + newFilePath, shell=True)

        # Extract mean
        mean_value = float(subprocess.check_output("gmt grdinfo " + newFilePath + " | grep z_min | awk '{print $3}'", shell=True))

        # Add to means list
        means.append(mean_value)

        print(path + ": " + str(mean_value))

    return means



# -------------------------------- OUTPUT -------------------------------- #

def coherenceTuple(path_list, means, baseline_table, filetype, stage):
    corrTuple = collections.namedtuple('coherence_data', ['paths', 'date_pairs', 'master', 'slave', 'temporal_baseline', 'orbital_baseline', 'mean_coherence'])

    date_pairs = []
    master=[]
    slave=[]

    # ASSUMES THAT GMTSAR INTERFEROGRAM DIRECTORIES ARE IN ORIGINAL NAMING CONVENTION
    if stage == 'original':
        for line in path_list:

            print('Searching: */' + line[(-16 - len(filetype)):(-len(filetype) - 1)] + "/*SLC")
            SLCs = glob.glob("*/" + line[(-16 - len(filetype)):(-len(filetype) - 1)] + "/*SLC")
            print(SLCs)

            date_pairs.append(line[(-16 - len(filetype)):(-len(filetype) - 1)])

            # Add their dates to the master and slave image lists
            # S1_20190122_ALL_F2.SLC
            len(line)
            master.append(dt.datetime.strptime(SLCs[0][-19:-11], "%Y%m%d"))
            slave.append(dt.datetime.strptime(SLCs[1][-19:-11], "%Y%m%d"))

    # ASSUMES THAT GMTSAR INTERFEROGRAM DIRECTORIES HAVE ALREADY BEEN RENAMED TO YYYYMMDD CONVENTION
    elif stage == 'renamed':
        for line in path_list:
            # Get date pair string from path 
            date_pairs.append(line[0:17])
            # Get datetime objects for master and slave dates
            master.append(dt.datetime.strptime(line[0:8], "%Y%m%d"))
            slave.append(dt.datetime.strptime(line[9:17], "%Y%m%d"))


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


    myData = corrTuple(paths=path_list, date_pairs=date_pairs, master=master, slave=slave, temporal_baseline=temporal_baseline, orbital_baseline=orbital_baseline, mean_coherence=means)

    print(myData)
    return myData



def writeData(corrTuple):
    with open('coherence_analysis.dat', 'w') as newTable:
        print("len(corrTuple.paths) = " + str(len(corrTuple.paths)))
        for i in range(len(corrTuple.paths)):
            print(corrTuple.paths[i])
            print(corrTuple.date_pairs[i])
            print(corrTuple.master[i].strftime("%Y%m%d"))
            print(corrTuple.slave[i].strftime("%Y%m%d"))
            print(corrTuple.mean_coherence[i])
            newTable.write(corrTuple.paths[i] + " " + corrTuple.date_pairs[i] + " " + corrTuple.master[i].strftime("%Y%m%d") + " " + corrTuple.slave[i].strftime("%Y%m%d") + " " + str(corrTuple.temporal_baseline[i]) + " " + str(corrTuple.orbital_baseline[i]) + " " + str(corrTuple.mean_coherence[i]) + '\n')

    print()
    print('File written:')
    print(newTable)

    return newTable



def plotCoherence(corr_table, baseline_table, filetype, stage):

    # Read in data data
    corrTuple = readCorrTable(corr_table, baseline_table, filetype, stage)

    # Establish figure
    fig, ax = plt.subplots()



    # Make orbital baseline colormap


    # Get range of baseline lengths
 
    baseline_range = list(range(0, int(math.floor(max(corrTuple.orbital_baseline)))))
    print('Baseline range = 0-' + str(math.floor(max(corrTuple.orbital_baseline))) + 'm')


    n = len(baseline_range) # Number of colors

    viridis = cm.get_cmap('plasma', n)
    print(viridis) 


    for i in range(len(corrTuple.mean_coherence)):

        # Get color coefficient
        line_color = np.floor(corrTuple.orbital_baseline[i]) / n
        print(line_color)

        plt.plot([corrTuple.master[i], corrTuple.slave[i]], [corrTuple.mean_coherence[i], corrTuple.mean_coherence[i]], zorder=3, color=viridis(line_color))

    # OLD
    # # Plot interferograms and their coherence
    # for i in range(len(corrTuple.mean_coherence)):
    #     plt.plot([corrTuple.master[i], corrTuple.slave[i]], [corrTuple.mean_coherence[i], corrTuple.mean_coherence[i]], zorder=3, color='C3')

    # Figure features
    plt.grid(axis='x', zorder=1)
    plt.xlim(min(corrTuple.master, default=dt.datetime.strptime('20141201', "%Y%m%d")) - dt.timedelta(days=5), max(corrTuple.slave, default=dt.datetime.strptime('20190801', "%Y%m%d")) + dt.timedelta(days=5))
    plt.ylim(0, 1)
    plt.xlabel('Date')
    plt.ylabel('Mean coherence')


    # cbar = plt.colorbar(fig)
    plt.show()


# OLD
# def plotCoherence(corrTuple, baseline_table):
#     # Establish figure
#     fig, ax = plt.subplots()

#     # Plot interferograms and their coherence
#     for i in range(len(corrTuple.mean_coherence)):
#         plt.plot([corrTuple.master[i], corrTuple.slave[i]], [corrTuple.mean_coherence[i], corrTuple.mean_coherence[i]], zorder=3, color='C3')

#     # Figure features
#    # ax.xaxis.set_minor_locator(AutoMinorLocator())
#     plt.grid(axis='x', zorder=1)
#     plt.xlim(min(corrTuple.master, default=dt.datetime.strptime('20141201', "%Y%m%d")) - dt.timedelta(days=5), max(corrTuple.slave, default=dt.datetime.strptime('20190801', "%Y%m%d")) + dt.timedelta(days=5))
#     plt.ylim(0, 1)
#     plt.xlabel('Date')
#     plt.ylabel('Mean coherence')
#     plt.show()



if __name__ == "__main__":
    driver()







