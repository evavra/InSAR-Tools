import numpy as np
import collections
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import datetime as dt
import math
import glob as glob


# -------------------------------- DRIVER -------------------------------- #

def driver():
    baseline_table = 'baseline_table.dat'
    scene_corr = 'scene_coherence.dat'
    tmin = 0
    tmax = 395
    bmax = 50
    cmin = 0.15
    stack_min = 4
    # datemin = '20170707'
    # datemax = '20171104'
    satellite = 1
    swath = 2
    new_list_all = ''
    new_list_new = 'intfs_to_append'
    old_list = 'attempted_intfs_08232019'
    good_intfs_counts = 'good_intfs.count'              # List of scenes to be used in time-series (including new additions) and the number of interferograms each scene is used in 


    
    # Read in scene data from baseline_table.dat and scene_coherence.dat
    orbit, dates, jday, blpara, blperp, datelabels = readBaselineTable(baseline_table)

    
    # with open(scene_corr, 'r') as file:
    #     corr_list = []
    #     for line in file:
    #         corr_list.append(float(line[:-1]))
    #         print(corr_list[-1])


    # # Make pairs, save to output_filename1
    # intf_list, updated_intf_in_dates, updated_intf_in_SLCs = makePairs(dates, blperp, corr_list, tmin, tmax, bmax, cmin, satellite, swath, new_list_all)
    
    # Read in old intf_in
    getListfromDir('/Users/ellisvavra/Thesis/insar/des/f2/intf_all/20*_20*', 1, 2, 'intf.in.pre07302019')
    old_intf_in = readIntfList(old_list, 'SLCs')

    # # Cross-reference new intf_in with 
    # new_intf_in = crossRefList(updated_intf_in_SLCs, old_intf_in, new_list_new)
    
    # # noisy_intfs = readIntfList("noisy_intfs_all.txt", 'date_pairs')

    # print("Number of interferograms: " + str(len(intf_list)))
    # print("Number of new interferograms: " + str(len(new_intf_in)))
    # # print("Number of noisy interferograms: " + str(len(noisy_intfs)))

    # plotBaselineTable(dates, blperp, intf_list, [])
    # # plotIntfDist(dates, c, [])
    
    scenes, count = readCounts(good_intfs_counts)
    intf_list, intf_in_dates, intf_in_SLCs = appendPairs(dates, blperp, tmin, tmax, bmax, stack_min, scenes, count, satellite, swath, 'test.append_ts')

    # Cross-reference new intf_in with 
    new_intf_in = crossRefList(intf_in_SLCs, old_intf_in, new_list_new)

# -------------------------------- CONFIGURE -------------------------------- #

def readIntfList(filename, list_type):
    print()
    print("Opening " + filename)

    with open(filename, "r") as file:
        str_list = file.readlines()

    output_list = []

    if list_type == 'date_pairs':
        for line in str_list:
            output_list.append([dt.datetime.strptime(line[0:8], "%Y%m%d"), dt.datetime.strptime(line[9:17], "%Y%m%d")])
            print(line)

        # print(output_list)

    elif list_type == 'SLCs':
        output_list = []

        for intf in str_list:
            output_list.append(intf[:-1])
            print(intf[:-1])

        # print(output_list)

    return output_list


def readBaselineTable(baseline_table):
    orbit = []
    dates = []
    jday = []
    blpara = []
    blperp = []
    temparray = []
    datelabels = []

    with open(baseline_table, 'r') as tempfile:
        for line in tempfile:
            temparray = line.split()
            print(temparray)
            orbit.append(temparray[0])
            jday.append(float(temparray[2]))
            blpara.append(float(temparray[3]))
            blperp.append(float(temparray[4]))

            # Get string dates from orbit header, i.e. s1a-iw2-slc-vv-20180503t135933-20180503t135952-021741-02582f-005
            datelabels.append(temparray[0][15:23])
            # Get datetime object from orbit header
            dates.append(dt.datetime.strptime(temparray[0][15:23] + temparray[0][24:30], "%Y%m%d%H%M%S"))


    #print(orbit, numdate, jday, blpara, blperp, date)
    return orbit, dates, jday, blpara, blperp, datelabels


def readCounts(good_intfs_list):
    scenes = []
    counts = []
    with open(good_intfs_list, 'r') as good_intfs:
        for line in good_intfs:
            scenes.append(line.split()[0])
            counts.append(int(line.split()[1]))
            print(line[:-1])

    return scenes, counts

# -------------------------------- INPUTS -------------------------------- #

def makePairs(dates, blperp, scene_corr, tmin, tmax, bmax, cmin, satellite, swath, output_filename):

    intf_list = []
    intf_in_dates = []
    intf_in_SLCs = []

    # Allow each scene to be the date1, test pairings with all viable date2 scenes
    for i in range(len(dates)):
         # Set/reset date2 index to be one greater than date1 index before for-loop iteration
        j = i + 1

        while j < len(dates):
            # Compute perpendicular (B_p) and temporal (B_t) baselines for pair
            B_p = abs(blperp[i] - blperp[j])
            B_t = dates[j] - dates[i]

            # Check if pair meets baseline criteria
            # Add pair to intf_list if it meets the input B_p and B_t thresholds
            if B_p <= bmax and B_t.days >= tmin and B_t.days <= tmax and scene_corr[i] >= cmin:
                print(dates[i].strftime("%Y%m%d")  + "_" + dates[j].strftime("%Y%m%d")  + ": " + str(round(B_p, 3)) + "m (" + str(B_t.days) + " days)")
                intf_list.append([i, j])
                intf_in_dates.append(dates[i].strftime("%Y%m%d")  + "_" + dates[j].strftime("%Y%m%d"))
                j+=1
            else:
                j+=1


    # Write new intf.in.NEW in following format:
    #   S1_20180508_ALL_F1:S1_20180514_ALL_F1

    print()
    print('Writing new intf.in...')
    print()
    print('Pairs:')

    with open(output_filename, 'w') as newList:
        for pair in intf_in_dates:
            # print(pair)
            print("S" + str(satellite) + '_' + pair[0:8] + '_ALL_F' + str(swath) + ':' + "S" + str(satellite) + '_' + pair[9:17] + '_ALL_F' + str(swath) + '\n')
            newList.write("S" + str(satellite) + '_' + pair[0:8] + '_ALL_F' + str(swath) + ':' + "S" + str(satellite) + '_' + pair[9:17] + '_ALL_F' + str(swath) + '\n')
            intf_in_SLCs.append("S" + str(satellite) + '_' + pair[0:8] + '_ALL_F' + str(swath) + ':' + "S" + str(satellite) + '_' + pair[9:17] + '_ALL_F' + str(swath))

        print()
        print('New file:')
        print(newList)

    print()
    print("Number of interferograms: " + str(len(intf_list)))
    print()
    
    
    return intf_list, intf_in_dates, intf_in_SLCs


def crossRefList(new_list, old_list, output_filename):
    with open(output_filename, "w") as newFile:

        listOut = []

        for item in new_list:
            if item not in old_list:
                listOut.append(item)

        for line in listOut:
            newFile.write("%s\n" % line)
            print(line)

    print()
    print("Number of new interferograms: " + str(len(listOut)))
    
    return listOut


def getListfromDir(search_str, swath, satellite, output_filename):
    print("Searching " + search_str + ' ...')
    dir_list = glob.glob(search_str)
    print(dir_list)

    # Write new intf.in.NEW in following format:
    #   S1_20180508_ALL_F1:S1_20180514_ALL_F1

    scene1=[]
    scene2=[]
    print()
    print('Pairs:')
    for line in dir_list:
        # DONT use datetime because GMTSAR doesnt use real julian day convention
        # Find SLCs in each directory - they have the full YYYYMMDD dates that we want.
        # print('Searching: ' + line + "/*SLC")
        SLCs = glob.glob(line + "/*SLC")
        # print(SLCs)
        # Add their dates to the date1 and date2 image lists
        scene1.append(SLCs[0][-22:-4])
        scene2.append(SLCs[1][-22:-4] )
        print(scene1[-1] + ':' + scene2[-1])


    with open(output_filename, 'w') as newList:
        for i in range(len(scene1)):
            newList.write(scene1[i] + ':' + scene2[i] + '\n')
        
        # print('Pairs:')
        print(newList)


def appendPairs(dates, blperp, tmin, tmax, bmax, stack_min, scenes, count, satellite, swath, output_filename):
    
    # First we need to get selected baseline info for the dates to be used in the timeseries
    new_blperp = []
    for i in range(len(dates)):
        if dates[i].strftime('%Y%m%d') in scenes:
            new_blperp.append(blperp[i])

    # For every scene in time series
    intf_list = []
    intf_in_dates = []
    intf_in_SLCs = []

    for i in range(len(scenes)):
        # If count is less than minimum stacking threshold, make more intfs
        if count[i] < stack_min:
            # For each scene under threshold, test all possible pairs under given thresholds
            for j in range(len(scenes)):
                # But don't let it try to pair with itself
                if i != j:
                    # Compute perpendicular (B_p) and temporal (B_t) baselines for pair
                    B_p = abs(new_blperp[i] - new_blperp[j])
                    B_t = abs(dt.datetime.strptime(scenes[i], '%Y%m%d') - dt.datetime.strptime(scenes[j], '%Y%m%d'))

                    # Check if pair meets baseline criteria
                    # Add pair to intf_list if it meets the input B_p and B_t thresholds
                    if B_p <= bmax and B_t.days >= tmin and B_t.days <= tmax and scenes[i] + "_" + scenes[j] not in intf_in_dates and scenes[j] + "_" + scenes[i] not in intf_in_dates:

                        if i < j:
                            print(scenes[i]  + "_" + scenes[j]  + ": " + str(round(B_p, 3)) + "m (" + str(B_t.days) + " days)")
                            intf_list.append([i, j])
                            intf_in_dates.append(scenes[i] + "_" + scenes[j])

                        else:
                            print(scenes[j]  + "_" + scenes[i]  + ": " + str(round(B_p, 3)) + "m (" + str(B_t.days) + " days)")
                            intf_list.append([i, j])
                            intf_in_dates.append(scenes[j] + "_" + scenes[i])


    # Write new append.intf.in in following format:
    #   S1_20180508_ALL_F1:S1_20180514_ALL_F1
    print()
    print('Writing new append.intf.in...')
    print()
    print('Pairs:')

    with open(output_filename, 'w') as newList:
        for pair in intf_in_dates:
            print("S" + str(satellite) + '_' + pair[0:8] + '_ALL_F' + str(swath) + ':' + "S" + str(satellite) + '_' + pair[9:17] + '_ALL_F' + str(swath))
            newList.write("S" + str(satellite) + '_' + pair[0:8] + '_ALL_F' + str(swath) + ':' + "S" + str(satellite) + '_' + pair[9:17] + '_ALL_F' + str(swath) + '\n')
            intf_in_SLCs.append("S" + str(satellite) + '_' + pair[0:8] + '_ALL_F' + str(swath) + ':' + "S" + str(satellite) + '_' + pair[9:17] + '_ALL_F' + str(swath))

        print()
        print('New file:')
        print(newList)

    print()
    print("Number of new interferograms: " + str(len(intf_list)))
    print()

    return intf_list, intf_in_dates, intf_in_SLCs


# -------------------------------- OUTPUT -------------------------------- #

def plotBaselineTable(dates, blperp, intf_list, noisy_intfs):
    # Establish figure
    fig, ax = plt.subplots()

    for bad_date in noisy_intfs:
        plt.plot([bad_date[0],bad_date[0]], [-1000,1000], 'r', alpha=0.05,zorder=3)
        plt.plot([bad_date[1],bad_date[1]], [-1000,1000], 'r', alpha=0.05, zorder=3)

    # Plot interferogram pairs as lines
    for i in intf_list:
        plt.plot([dates[i[0]], dates[i[1]]], [blperp[i[0]], blperp[i[1]]], 'k', zorder=2, lw=1)

    # Plot scenes over pair lines
    plt.scatter(dates, blperp, s=15, zorder=3)

    # Figure features
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    plt.grid(axis='x', zorder=1)
    plt.xlim(min(dates) - dt.timedelta(days=100), max(dates) + dt.timedelta(days=100))
    plt.ylim(int(math.ceil((min(blperp) - 50) / 50.0) ) * 50, int(math.floor((max(blperp) + 50) / 50.0)) * 50)
    plt.xlabel('Year')
    plt.ylabel('Baseline (m)')
    plt.show()


def plotIntfDist(dates, intf_list, noisy_intfs):

    # Create chronological IDs (from first scene date) for interferograms
    index = list(range(len(intf_list)))
    ID = [i + 1 for i in index]
    print(len(ID))
    print(len(intf_list))

    # Establish figure
    plt.figure(figsize=(8,8))

    # Plot temporal baseline for each interferogram
    for i in iter(index):
        plt.plot([dates[intf_list[i][0]], dates[intf_list[i][1]]], [i + 1, i + 1], 'k', zorder=2, lw=0.5)

        print("Plotting " + dates[intf_list[i][0]].strftime("%Y%m%d")  + "_" + dates[intf_list[i][1]].strftime("%Y%m%d") + "...")

    # Plot NOISY INTFS
    for bad_date in noisy_intfs:
        plt.plot([bad_date[0],bad_date[0]], [-1000,1000], 'r',alpha=0.05)
        plt.plot([bad_date[1],bad_date[1]], [-1000,1000], 'r',alpha=0.05)

    # Figure settings
    plt.grid(axis='x', zorder=1, linestyle=':')
    plt.xlim(min(dates) - dt.timedelta(days=100), max(dates) + dt.timedelta(days=100))
    plt.ylim(min(index), max(index))
    plt.xlabel('Year')
    plt.ylabel('Interferogram ID number')

    plt.show()


if __name__ == "__main__":
    driver()
