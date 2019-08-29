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

    homedir = '/Users/ellisvavra/Thesis/insar/des/f2/intf_all/*/'   # Full path to all intf .grd files specified in filetype
    filetype = 'corr.grd'                                           # Target coherence grid name (usually corr.grd)
    skip_list = []
    area = [1000, 13000, 2000, 4500]                                # Region for coherence analysis
    baseline_table = 'baseline_table.dat'                           # GMTSAR baseline info table
    stage = 'GMTSAR'                                                # 'GMTSAR' for GMTSAR formatted directories, 'CANDIS' for CANDIS formatted directories
    level = 1                                                       # 1 for same directory as interferogram directories, 2 for directory above home diretory for intf directories
    calc_means = 'no'                                               # 'yes' if mean interferometric coherence needs to be calculated using GMT; '2' if mean_corr.grd already exists
    calc_std = 'no'
    calc_sigma = 'yes'
    intf_table = 'intf_table.dat'
    corr_min = 0.20                                                 # Min. coherence threshold for intfs going into NSBAS
    corr_max = 1.00                                                 # Max. coherence threshold for intfs going into NSBAS (usually 1.0)
    max_count = 12                                                  # Set maximum number of interferograms to be used in common scene stacking (n most coherent pairs)

    output_list_name = 'new_intfs_08282019'
    filt_intf_table = 'selected_table_new.dat'

    NSBAS_list_GMTSAR = 'intfs_for_CANDIS.GMTSAR'
    NSBAS_list_CANDIS = 'intfs_for_CANDIS.CANDIS'
    NSBAS_table = 'intfs_for_CANDIS.dat'

    # step = 1                                                        # 1. Do original mean calculation, make intf_table.dat, make plots
    step = 2                                                        # 2. Work from intf_table.dat to test threshold cmin

    if step == 1:                                                  
    # 1. Do original mean calculation, make intf_table.dat, make plots

        # Get paths to target .grds
        path_list = getPaths(homedir, filetype)

        # Calulate means
        means = getMeans(path_list, area, filetype, calc_means)

        # Calulate standard deviation
        sigma1, sigma2, sigma3 = getSigma(path_list, area, filetype, calc_sigma)

        # Write data to interferogram tuple
        iTuple = intfTuple(path_list, means, baseline_table, filetype, stage, level)

        # Save tuple to intf_table file for safe keeping
        writeData(iTuple, intf_table)

        # Make coherence plot
        plotIntfCoherence(iTuple, baseline_table, filetype, stage)

        # Count number of times used
        scene_dates, date_sums = count(iTuple)

        # Get mean coherence for each scene
        scene_dates, mean_scene_coherence = sceneCorr(iTuple)

        # Plot mean scene coherence
        plotSceneCoherence(scene_dates, mean_scene_coherence, date_sums)

        # Plot interferogram coherence distribution
        plotCorrHist(iTuple)

        # Plot sigma
        plotSigmaHist(sigma1, 1)
        plotSigmaHist(sigma2, 2)
        plotSigmaHist(sigma3, 3)

        plotIntfCoherenceBounds(iTuple, baseline_table, filetype, stage, sigma_n)


    elif step == 2:
    # 2. Work from intf_table.dat to test threshold cmin

        # Load data from saved intf_table.dat
        iTuple = readIntfTable(intf_table)

        # Filter intferogram list based off of coherence
        intf_list, new_intf_table = sortFromCorr(iTuple, corr_min, corr_max, intf_table, output_list_name, filt_intf_table)
        
        # Write filtered list to file
        writeIntfList(intf_list, output_list_name)

        # RELOAD ITUPLE WITH COHERENCE-FILTERED DATASET
        iTuple = readIntfTable(filt_intf_table)

        # Filter out redundant interferograms
        filterRedundant(filt_intf_table, max_count, NSBAS_list_GMTSAR, NSBAS_list_CANDIS, NSBAS_table, filetype)

        # RELOAD ITUPLE WITH FINAL COHERENCE AND REDUNDANCY FILTERED DATASET
        iTuple = readIntfTable(NSBAS_table)

        # Count number of times used
        scene_dates, date_sums = count(iTuple)

        # Get mean coherence for each scene
        scene_dates, mean_scene_coherence = sceneCorr(iTuple)

        # Write mean scene coherence to a file
        writeSceneCorr(mean_scene_coherence)

        # Make interferogram coherence plot
        plotIntfCoherence(iTuple, baseline_table, filetype, stage)

        # Plot mean scene coherence
        plotSceneCoherence(scene_dates, mean_scene_coherence, date_sums)

        # Plot interferogram coherence histogram
        plotCorrHist(iTuple)



# -------------------------------- CONFIGURE -------------------------------- #

def getPaths(homedir, filetype):
    # Find all .grd files of given type
    print("Searching for: " + homedir + filetype)
    path_list = glob.glob(homedir + filetype)
    print(path_list)

    return path_list


def readIntfTable(intf_table):
    # Read intf_table.dat to intfTuple

    iTuple = collections.namedtuple('intf_data', ['paths', 'date_pairs', 'date1', 'date2', 'temporal_baseline', 'orbital_baseline', 'mean_coherence'])

    paths = []
    date_pairs = []
    date1 = []
    date2 = []
    temporal_baseline = []
    orbital_baseline = []
    mean_coherence = []

    with open(intf_table, "r") as intf_dat:
        for line in intf_dat:
            temparray = line.split()
            paths.append(temparray[0])
            date_pairs.append(temparray[1])
            date1.append(dt.datetime.strptime(temparray[2], "%Y%m%d"))
            date2.append(dt.datetime.strptime(temparray[3], "%Y%m%d"))
            temporal_baseline.append(float(temparray[4]))
            orbital_baseline.append(float(temparray[5]))
            mean_coherence.append(float(temparray[6]))

        myData = iTuple(paths=paths, date_pairs=date_pairs, date1=date1, date2=date2, temporal_baseline=temporal_baseline, orbital_baseline=orbital_baseline, mean_coherence=mean_coherence)

    return myData


def intfTuple(path_list, means, baseline_table, filetype, dir_type, level):
    # Create intfTuple from scratch

    iTuple = collections.namedtuple('intf_data', ['paths', 'date_pairs', 'date1', 'date2', 'temporal_baseline', 'orbital_baseline', 'mean_coherence'])

    date_pairs = []
    date1 = []
    date2 = []

    # ASSUMES THAT GMTSAR INTERFEROGRAM DIRECTORIES ARE IN ORIGINAL NAMING CONVENTION
    if dir_type == 'GMTSAR':
        for line in path_list:

            # Script run in same directory as interferogram directories 
            if level == 1: 
                print('Searching: ' + line[(-16 - len(filetype)): (-len(filetype) - 1)] + "/*SLC")
                SLCs = glob.glob(line[(-16 - len(filetype)): (-len(filetype) - 1)] + "/*SLC")     

            # Or if script is run in directory above interferogram directories 
            elif level == 2:
                print('Searching: */' + line[(-16 - len(filetype)): (-len(filetype) - 1)] + "/*SLC")
                SLCs = glob.glob('*/' + line[(-16 - len(filetype)): (-len(filetype) - 1)] + "/*SLC")

            # SLCs = glob.glob("*/" + line[(-16 - len(filetype)): (-len(filetype) - 1)] + "/*SLC")
            print(SLCs)

            date_pairs.append(line[(-16 - len(filetype)): (-len(filetype) - 1)])

            # Add their dates to the date1 and date2 image lists
            # S1_20190122_ALL_F2.SLC
            len(line)
            date1.append(dt.datetime.strptime(SLCs[0][-19: -11], "%Y%m%d"))
            date2.append(dt.datetime.strptime(SLCs[1][-19: -11], "%Y%m%d"))


    # ASSUMES THAT GMTSAR INTERFEROGRAM DIRECTORIES HAVE ALREADY BEEN RENAMED TO YYYYMMDD CONVENTION
    elif dir_type == 'CANDIS':
        for line in path_list:
            # Get date pair string from path
            date_pairs.append(line[0: 17])
            # Get datetime objects for date1 and date2 dates
            date1.append(dt.datetime.strptime(line[0: 8], "%Y%m%d"))
            date2.append(dt.datetime.strptime(line[9: 17], "%Y%m%d"))

    # Compute perpendicular (B_p) and temporal (B_t) baselines for pair
    orbit, dt_dates, jday, blpara, blperp, datelabels = new_baseline_table.readBaselineTable(baseline_table)

    print("Datetime dates:")
    print(dt_dates)
    # Convert date list to strings for searching with date1 and date2 lists
    str_dates = []

    print("String dates: ")
    for date in dt_dates:
        str_dates.append(date.strftime("%Y%m%d"))
        print(str_dates[-1])
    # print(str_dates)

    # Calulate baselines
    temporal_baseline = []
    orbital_baseline = []

    for i in range(len(date1)):

        print('Searching for ' + date1[i].strftime("%Y%m%d"))

        # Find date1 index
        Mi = str_dates.index(date1[i].strftime("%Y%m%d"))
        # Find date2 index
        Si = str_dates.index(date2[i].strftime("%Y%m%d"))
        # Calculate baselines
        B_p = abs(blperp[Si] - blperp[Mi])
        B_t_dt = dt_dates[Si] - dt_dates[Mi]
        B_t = B_t_dt.days

        print("For " + date1[i].strftime("%Y%m%d") + "_" + date2[i].strftime("%Y%m%d") + ": B_t = " + str(B_t) + ', ' + "B_p = " + str(B_p))

        temporal_baseline.append(B_t)
        orbital_baseline.append(B_p)

    myData = iTuple(paths=path_list, date_pairs=date_pairs, date1=date1, date2=date2, temporal_baseline=temporal_baseline, orbital_baseline=orbital_baseline, mean_coherence=means)

    print(myData)
    return myData


# -------------------------------- ANALYSIS -------------------------------- #

def getMeans(path_list, area, filetype, calc_means):
    # Use GMT to calculate means of .grd files
    means = []

    print()
    print()
    print()
    print('Calculating means...')

    for path in path_list:
        # Calulate mean and save to temporary file
        newFilePath = path[0:(len(path) - len(filetype))] + "mean_corr.grd"

        if calc_means == 'yes':
            subprocess.call("gmt grdmath " + path + " MEAN = " + newFilePath, shell=True)
             # Extract mean
            mean_value = float(subprocess.check_output("gmt grdinfo " + newFilePath + " | grep z_min | awk '{print $3}'", shell=True))

            # Add to means list
            means.append(mean_value)
            print(path + ": " + str(mean_value))


        elif calc_means == 'no':
            # print('Reading in mean coherence values... ')
            # Extract mean
            mean_value = float(subprocess.check_output("gmt grdinfo " + newFilePath + " | grep z_min | awk '{print $3}'", shell=True))

            # Add to means list
            means.append(mean_value)
            print(path + ": " + str(mean_value))

    return means


def getSigma(path_list, area, filetype, calc_sigma):
    # Use GMT to calculate n-sigma of .grd files
    sigma1 = []
    sigma2 = []
    sigma3 = []

    print()
    print()
    print()
    print('Calculating sigmas...')

    
    for path in path_list:
        # Calulate standard deviation and save to temporary file
        newFilePath = path[0:(len(path) - len(filetype))] + "std_corr.grd"

        if calc_sigma == 'yes':
            subprocess.call("gmt grdmath " + path + " STD = " + newFilePath, shell=True)
             # Extract standard deviation
            sigma_value = float(subprocess.check_output("gmt grdinfo " + newFilePath + " | grep z_min | awk '{print $3}'", shell=True))

            # Add to std list
            sigma1.append(sigma_value)
            sigma2.append(sigma_value * 2)
            sigma3.append(sigma_value * 3)
            print(path + ": " + str(sigma_value))


        elif calc_sigma == 'no':
            # print('Reading in std coherence values... ')
            # Extract std
            sigma_value = float(subprocess.check_output("gmt grdinfo " + newFilePath + " | grep z_min | awk '{print $3}'", shell=True))

            # Add to std list
            sigma1.append(sigma_value)
            sigma2.append(sigma_value * 2)
            sigma3.append(sigma_value * 3)
            print(path + ": " + str(sigma_value))

    return sigma1, sigma2, sigma3





def count(iTuple):
    # Count how many time each scene is used in interferograms
    print('date1s: ')
    for date in iTuple.date1:
        print(date.strftime("%Y-%m-%d"))
    print()
    print('date2s: ')
    for date in iTuple.date2:
        print(date.strftime("%Y-%m-%d"))
    print()

    # Initiate
    all_dates = iTuple.date1 + iTuple.date2
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
        print(date.strftime("%Y%m%d") + " " + str(date_sums[-1]))
    print()

    return scene_dates, date_sums


def sceneCorr(iTuple):
    # Determine mean coherence of all interferograms each scene is used in

    # Combine date lists
    all_dates = iTuple.date1 + iTuple.date2
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
    corrTuple = collections.namedtuple('coherence_data', ['dates', 'date1_indicies', 'date2_indicies', 'mean_scene_coherence'])

    date1_indicies = []
    date2_indicies = []

    print('Number of interferograms: ' + str(len(iTuple.date1)))
    print('Number of scenes: ' + str(len(scene_dates)))
    print()

    # Find indicies of all usages of each SAR scene by querying the date1 and date2 lists
    for date in scene_dates:
        print('Finding occurences of ' + date.strftime("%Y-%m-%d"))
        if date in iTuple.date1:
            temp_date1_index = [i for i, x in enumerate(iTuple.date1) if x == date]
            date1_indicies.append(temp_date1_index)
            print(temp_date1_index)
        else:
            date1_indicies.append([])
            print([])

        if date in iTuple.date2:
            temp_date2_index = [i for i, x in enumerate(iTuple.date2) if x == date]
            date2_indicies.append(temp_date2_index)
            print(temp_date2_index)
        else:
            date2_indicies.append([])
            print([])
    print()

    # Calulate mean coherence for each scene
    mean_scene_coherence = []

    print('Mean scene coherence: ')
    for i in range(len(scene_dates)):
        date1_coherences = [iTuple.mean_coherence[j] for j in date1_indicies[i]]
        date2_coherences = [iTuple.mean_coherence[j] for j in date2_indicies[i]]
        mean_coherence = sum(date1_coherences + date2_coherences) / len(date1_coherences + date2_coherences)
        mean_scene_coherence.append(mean_coherence)
        print(mean_coherence)

    print()
    return scene_dates, mean_scene_coherence


def sortFromCorr(iTuple, corr_min, corr_max, intf_table, output_list_name, output_table_name):
    # Find indicies of all interferograms who meet the minimum coherence threshold
    intf_list = []

    print()
    print('Interferograms with coherence between ' + str(corr_min) + ' and ' + str(corr_max))

    for i in range(len(iTuple.paths)):
        if iTuple.mean_coherence[i] >= corr_min and iTuple.mean_coherence[i] < corr_max:
            intf_list.append(iTuple.date_pairs[i])
            print("[" + str(i) + "] " + iTuple.date_pairs[i] + ': ' + str(iTuple.mean_coherence[i]))


    # indicies = []

    # for i in range(len(intf_list)):
    #     if intf_list[i] in iTuple.date_pairs:
    #         indicies.append(i)

    # print(indicies)
    
    
    with open(output_table_name, 'w') as intf_table:
        for i in range(len(iTuple.paths)):
            if iTuple.mean_coherence[i] >= corr_min and iTuple.mean_coherence[i] < corr_max:
                intf_list.append(iTuple.date_pairs[i])
                print("[" + str(i) + "] " + iTuple.date_pairs[i] + ': ' + str(iTuple.mean_coherence[i]))
                intf_table.write(iTuple.paths[i] + " " + iTuple.date_pairs[i] + " " + iTuple.date1[i].strftime("%Y%m%d") + " " + iTuple.date2[i].strftime("%Y%m%d") + " " + str(iTuple.temporal_baseline[i]) + " " + str(iTuple.orbital_baseline[i]) + " " + str(iTuple.mean_coherence[i]) + '\n')

    print()
    print('File written:')
    print(intf_table)

    # print(intf_list)
    return intf_list, intf_table


def filterRedundant(filt_intf_table, max_count, NSBAS_list_GMTSAR, NSBAS_list_CANDIS, NSBAS_table, filetype):

    def take2(elem):
        # Helper function for sorting based off of date1
        return elem[2]

    def take5(elem):
        # Helper function for sorting based off of mean_coherence
        return elem[5]

    # Intialize master interferogram list
    intf_master_list = []

    # Read in selected_intf_table.dat
    iTuple = readIntfTable(filt_intf_table)

    # Make full list of scene usages
    all_dates = iTuple.date1 + iTuple.date2 
    print('Initial number of intergerograms: ' + str(len(all_dates)/2))
    dates_tested = []

    # Now, here we are going to utilze row vector lists (scenes) instead of column vector lists (values) in order to be able to sort by mean coherence

    # Begin looping through all scene usages
    for date in all_dates:

        # Determine if the scene has already been analyzed or not. If not, continue.
        if date not in dates_tested:

            # Flag to prevent repetition/readding intfs to final list
            dates_tested.append(date)

            # Get list of all interferograms using a given date
            intfs = []

            for i in range(len(iTuple.date_pairs)):
                intf = []

                # Add intf if scene is date1
                if date == iTuple.date1[i]:
                    intf.append(iTuple.paths[i])               # [0]
                    intf.append(iTuple.date_pairs[i])          # [1]
                    intf.append(iTuple.date1[i])               # [2]
                    intf.append(iTuple.date2[i])               # [3]
                    intf.append(iTuple.temporal_baseline[i])   # [4]
                    intf.append(iTuple.orbital_baseline[i])    # [5]
                    intf.append(iTuple.mean_coherence[i])      # [6]
                    intfs.append(intf)

                # Add intf if scene is date2
                elif date == iTuple.date2[i]:
                    intf.append(iTuple.paths[i])               # [0]
                    intf.append(iTuple.date_pairs[i])          # [1]
                    intf.append(iTuple.date1[i])               # [2]
                    intf.append(iTuple.date2[i])               # [3]
                    intf.append(iTuple.temporal_baseline[i])   # [4]
                    intf.append(iTuple.orbital_baseline[i])    # [5]
                    intf.append(iTuple.mean_coherence[i])      # [6]
                    intfs.append(intf)        

            # Determine number of scene usages
            n = len(intfs)

            print()
            print(date.strftime("%Y%m%d") + ' used ' + str(n) + ' times')

            # Sort intf vectors by mean coherence
            intfs.sort(reverse=True, key=take5)

            for intf in intfs:
                print(intf[1] + ': ' + str(intf[5]))

            # Now, add most coherent interferograms to output list based off of max_count
            final_intfs = []

            while len(final_intfs) < max_count:
                for intf in intfs:
                    final_intfs.append(intf)

            # Rearrange final intf list by date1
            final_intfs.sort(key=take2)
            # print('Interferograms to use for ' + date.strftime('%Y%m%d'))
            # print(final_intfs[:n])

            # Add to master list (GMTSAR formatted date-pairs)

            for intf in final_intfs:
                if intf not in intf_master_list: # Prevent duplicates
                    intf_master_list.append(intf)
                    print()
                    print(intf[1] + ' added to master list')




    with open(NSBAS_list_GMTSAR, 'w') as final_gmtsar_list:
        for intf in intf_master_list:
            final_gmtsar_list.write(intf[1] + '\n')

        print()
        print('File written:')
        print(final_gmtsar_list)
        print()

    with open(NSBAS_list_CANDIS, 'w') as final_candis_list:
        for intf in intf_master_list:
            final_candis_list.write(intf[2].strftime("%Y%m%d") + '_' +  intf[3].strftime("%Y%m%d") + '\n')

        print()
        print('File written:')
        print(final_candis_list)
        print()
        print()


    with open(NSBAS_table, 'w') as final_intf_table:
        for intf in intf_master_list:
            print(intf)
            final_intf_table.write(intf[0] + ' ' + intf[1] + ' ' +  intf[2].strftime("%Y%m%d") + ' ' +  intf[3].strftime("%Y%m%d") + ' ' +  str(intf[4]) + ' ' +  str(intf[5]) + ' ' + str(intf[6]) + '\n')
                

        print()
        print('File written:')
        print(final_intf_table)
        print()
        print('Number of intergerograms for CANDIS: ' + str(len(intf_master_list)))
        print()


    return intf_master_list


# -------------------------------- OUTPUT -------------------------------- #

def writeData(iTuple, intf_table):
    with open(intf_table, 'w') as newTable:
        print("len(iTuple.paths) = " + str(len(iTuple.paths)))
        print("len(iTuple.date_pairs) = " + str(len(iTuple.date_pairs)))

        for i in range(len(iTuple.paths)):
            print(iTuple.paths[i])
            print(iTuple.date_pairs[i])
            print(iTuple.date1[i].strftime("%Y%m%d"))
            print(iTuple.date2[i].strftime("%Y%m%d"))
            print(iTuple.mean_coherence[i])
            newTable.write(iTuple.paths[i] + " " + iTuple.date_pairs[i] + " " + iTuple.date1[i].strftime("%Y%m%d") + " " + iTuple.date2[i].strftime("%Y%m%d") + " " + str(iTuple.temporal_baseline[i]) + " " + str(iTuple.orbital_baseline[i]) + " " + str(iTuple.mean_coherence[i]) + '\n')

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
            newList.write(intf + '\n')

    print()
    print('File written:')
    print(newList)


def writeSceneCorr(mean_scene_coherence):
    with open('scene.coherence', 'w') as newList:
        for corr_value in mean_scene_coherence:
            newList.write(str(corr_value) + '\n')

    print('Scene coherences saved to scene.coherence')
    print()


# -------------------------------- PLOTTING -------------------------------- #

def plotIntfCoherence(iTuple, baseline_table, filetype, stage):
    # Read in data data

    # Establish figure
    fig, ax = plt.subplots()

    # Get range of baseline lengths
    baseline_range = list(range(0, int(math.floor(max(iTuple.orbital_baseline)))))
    print('Baseline range = 0-' + str(math.floor(max(iTuple.orbital_baseline))) + 'm')
    n = len(baseline_range)  # Number of colors
    viridis = cm.get_cmap('viridis', n)
    print(viridis)

    for i in range(len(iTuple.mean_coherence)):
        # Get color coefficient
        line_color = np.floor(iTuple.orbital_baseline[i]) / n
        print(line_color)

        plt.plot([iTuple.date1[i], iTuple.date2[i]], [iTuple.mean_coherence[i], iTuple.mean_coherence[i]], zorder=3, color=viridis(line_color))

    # Figure features
    plt.grid(axis='x', zorder=1)
    plt.xlim(min(iTuple.date1, default=dt.datetime.strptime('20141201', "%Y%m%d")) - dt.timedelta(days=5), max(iTuple.date2, default=dt.datetime.strptime('20190801', "%Y%m%d")) + dt.timedelta(days=5))
    plt.ylim(0, np.ceil(max(iTuple.mean_coherence)*10)/10)
    plt.xlabel('Date')
    plt.ylabel('Mean coherence')

    plt.show()


def plotSceneCoherence(scene_dates, mean_scene_coherence, date_sums):
    # Establish figure
    fig, ax = plt.subplots()

    plt.scatter(scene_dates, mean_scene_coherence, c=date_sums, zorder=3)

    # Plot scene IDs
    n = np.arange(1, len(scene_dates) + 1)

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


def plotSigmaHist(sigma_n, n):

    # Establish figure
    fig, ax = plt.subplots()

    plt.hist(sigma_n, bins=100)

    # Figure features
    plt.grid(axis='x', zorder=1)
    plt.xlabel(str(n) + '-sigma of coherence')
    plt.ylabel('Number of interferograms')

    plt.show()


def plotIntfCoherenceBounds(iTuple, baseline_table, filetype, stage, sigma_n):
    # Read in data data

    # Establish figure
    fig, ax = plt.subplots()

    # Get range of baseline lengths
    baseline_range = list(range(0, int(math.floor(max(iTuple.orbital_baseline)))))
    print('Baseline range = 0-' + str(math.floor(max(iTuple.orbital_baseline))) + 'm')
    n = len(baseline_range)  # Number of colors
    viridis = cm.get_cmap('viridis', n)
    print(viridis)

    for i in range(len(iTuple.mean_coherence)):
        # Get color coefficient
        line_color = np.floor(iTuple.orbital_baseline[i]) / n
        print(line_color)

        plt.plot([iTuple.date1[i], iTuple.date2[i]], [iTuple.mean_coherence[i] - sigma_n[i], iTuple.mean_coherence[i] - sigma_n[i]], zorder=3, color=viridis(line_color))

    # Figure features
    plt.grid(axis='x', zorder=1)
    plt.xlim(min(iTuple.date1, default=dt.datetime.strptime('20141201', "%Y%m%d")) - dt.timedelta(days=5), max(iTuple.date2, default=dt.datetime.strptime('20190901', "%Y%m%d")) + dt.timedelta(days=5))
    plt.ylim(0, np.ceil(max(iTuple.mean_coherence)*10)/10)
    plt.xlabel('Date')
    plt.ylabel('Mean coherence')

    plt.show()

if __name__ == "__main__":
    driver()
