import numpy as np
import collections
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import datetime as dt
import math


# -------------------------------- CONFIGURE -------------------------------- #
def readIntfList(filename):
    print("Opening " + filename)

    with open(filename, "r") as file:
        str_list = file.readlines()

    intf_dates = []

    for line in str_list:
        intf_dates.append([dt.datetime.strptime(line[0:7], "%Y%j"), dt.datetime.strptime(line[8:15], "%Y%j")])

    print(intf_dates)

    return intf_dates


def readBaselineTable(input):
    orbit = []
    dates = []
    jday = []
    blpara = []
    blperp = []
    temparray = []
    datelabels = []

    with open(input, 'r') as tempfile:
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

# # -------------------------------- INPUTS -------------------------------- #

def makePairs(dates, blperp, tmax, bmax):

    intf_list = []
    intf_in = []

    # Allow each scene to be the master, test pairings with all viable slave scenes
    for i in range(len(dates)):
         # Set/reset slave index to be one greater than master index before for-loop iteration
        j = i + 1

        while j < len(dates):
            # Compute perpendicular (B_p) and temporal (B_t) baselines for pair
            B_p = abs(blperp[i] - blperp[j])
            B_t = dates[j] - dates[i]

            # Check if pair meets baseline criteria
            # Add pair to intf_list if it meets the input B_p and B_t thresholds
            if B_p <= bmax and B_t.days <= tmax:
                print(dates[i].strftime("%Y%m%d")  + "_" + dates[j].strftime("%Y%m%d")  + ": " + str(round(B_p, 3)) + "m (" + str(B_t.days) + " days)")
                intf_list.append([i, j])
                intf_in.append(dates[i].strftime("%Y%m%d")  + "_" + dates[j].strftime("%Y%m%d"))
                j+=1
            else:
                j+=1

    print("Number of interferograms: " + str(len(intf_list)))

    return intf_list, intf_in

# -------------------------------- PLOTTING -------------------------------- #
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
    orbit, dates, jday, blpara, blperp, datelabels = readBaselineTable('baseline_table.dat')
    intf_list, intf_in = makePairs(dates, blperp, 120, 300)

    noisy_intfs = readIntfList("noisy_intfs_1&2")

    print("Number of interferograms: " + str(len(intf_list)))
    print("Number of noisy interferograms: " + str(len(noisy_intfs)))

    plotBaselineTable(dates, blperp, intf_list, noisy_intfs)
    #plotIntfDist(dates, intf_list, noisy_intfs)
