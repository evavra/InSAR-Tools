import numpy as np
import collections
import datetime as dt
import matplotlib.pyplot as plt
from getUNR import readStationList
import math
import pandas as pd

"""
Basic GPS processing scripts
"""


def driver():
    station()


# ------------------------- CONFIGURE -------------------------

def station():
    filename = '/Users/ellisvavra/GPS-Tools/GPS_data_20200529/RDOM.tenv3'
    data_format = 'env'
    component = 'V'
    start = '20141101'
    end = '20200401'
    theta = 31

    ax = plt.subplot(111)
    start_end = [dt.datetime.strptime(start, '%Y%m%d'), dt.datetime.strptime(end, '%Y%m%d')]
    gps_data = readUNR(filename, data_format)
    stationTimeSeries(gps_data.dates, gps_data.up, component, start_end, ax)

    plt.xlabel('Date')
    plt.ylabel('Vertical displacement (m)')

    plt.show()


def baseline():
    # CALCULATE ONE BASELINE
    # Load data
    station1 = readUNR('/Users/ellisvavra/Thesis/gps/GPS_data_20190904/RDOM.NA12.tenv3', 'env')
    # station2 = readUNR('GPS_data_20190904/P636.NA12.tenv3', 'env')
    station2 = readUNR('/Users/ellisvavra/Thesis/gps/GPS_data_20190904/CA99.NA12.tenv3', 'env')

    start = '20141108'
    end = '20190801'

    component = 'up'
    outDir = 'BaselinePlots'

    # Compute baseline time series
    baselineDates, baselineChange = calcBaseline(station1, station2, start, end, component)
    fileName = outDir + '/' + station1.station_name[0] + '-' + station2.station_name[0]
    plotBaseline(baselineDates, baselineChange, fileName, component)


def baselineMean():
    # SMOOTH BASELINE WITH MOVING WINDOW MEAN
    # Load data
    station1 = readUNR('/Users/ellisvavra/Thesis/gps/GPS_data_20190904/RDOM.NA12.tenv3', 'env')
    # station2 = readUNR('GPS_data_20190904/P636.NA12.tenv3', 'env')
    station2 = readUNR('/Users/ellisvavra/Thesis/gps/GPS_data_20190904/CA99.NA12.tenv3', 'env')

    start = '20141108'
    end = '20190801'

    component = 'up'
    outDir = 'BaselinePlots'

    # Compute baseline time series
    baselineDates, baselineChange = calcBaseline(station1, station2, start, end, component)
    fileName = outDir + '/' + station1.station_name[0] + '-' + station2.station_name[0]


# ------------------------- READ -------------------------

def readASCII(fileName, format):

    if format == 'UNR':
        # Read in UNR ENV-formatted cGPS daily positions
        # [0] station_name     # [10] north(m)
        # [1] YYMMMDD          # [11] u0(m)
        # [2] yyyy_yyyy        # [12] up(m)
        # [3] MJD              # [13] ant(m)
        # [4] week             # [14] sig_e(m)
        # [5] day              # [15] sig_n(m)
        # [6] reflon           # [16] sig_u(m)
        # [7] e0(m)            # [17] corr_en
        # [8] east(m)          # [18] corr_eu
        # [9] n0(m)            # [19] corr_nu

        data = pd.read_csv(fileName, engine='python', sep='\s+', header=0)
        dates = pd.to_datetime(data['YYMMMDD'], format='%y%b%d')
        pos = pd.DataFrame()
        data['Date'] = dates

    elif format == 'stationInfo':
        # Read in master list of GPS station names and locations
        data = pd.read_csv(fileName, engine='python', sep='\s+', header=None)
        data.columns = ['Site', 'Lat', 'Lon', 'Elev']

    elif format == 'lookTable':
        # Read in GMTSAR lookup table to get LOS vector components
        data = pd.read_csv(fileName, sep=' ', header=None)
        data.columns = ['lon', 'lat', 'elev', 'Ue', 'Un', 'Uz']

    return data


def readUNR(filename, data_format):  # DEPRECIATED - use readASCII instead

    if data_format == 'xyz':

        xyz = collections.namedtuple('TimeS', ['name', 'coords', 'dt', 'dN', 'dE', 'dU'])

    elif data_format == 'env':
        # [0] station_name     # [10] north(m)
        # [1] YYMMMDD          # [11] u0(m)
        # [2] yyyy_yyyy        # [12] up(m)
        # [3] MJD              # [13] ant(m)
        # [4] week             # [14] sig_e(m)
        # [5] day              # [15] sig_n(m)
        # [6] reflon           # [16] sig_u(m)
        # [7] e0(m)            # [17] corr_en
        # [8] east(m)          # [18] corr_eu
        # [9] n0(m)            # [19] corr_nu

        enz = collections.namedtuple('dataGPS', ['station_name', 'dates', 'yyyy_yyyy', 'MJD', 'week', 'day', 'reflon', 'e0', 'east', 'n0', 'north', 'u0', 'up', 'ant', 'sig_e', 'sig_n', 'sig_u', 'corr_en', 'corr_eu', 'corr_nu'])

        print('Opening ' + str(filename))

        with open(filename) as f:
            lines = f.readlines()[1:]

        station_name = []
        dates = []
        yyyy_yyyy = []
        MJD = []
        week = []
        day = []
        reflon = []
        e0 = []
        east = []
        n0 = []
        north = []
        u0 = []
        up = []
        ant = []
        sig_e = []
        sig_n = []
        sig_u = []
        corr_en = []
        corr_eu = []
        corr_nu = []

        for line in lines:
            temp_line = line.split()

            station_name.append(temp_line[0])
            dates.append(dt.datetime.strptime(temp_line[1], '%y%b%d'))
            yyyy_yyyy.append(float(temp_line[2]))
            MJD.append(int(temp_line[3]))
            week.append(int(temp_line[4]))
            day.append(int(temp_line[5]))
            reflon.append(float(temp_line[6]))
            e0.append(int(temp_line[7]))
            east.append(float(temp_line[8]))
            n0.append(int(temp_line[9]))
            north.append(float(temp_line[10]))
            u0.append(float(temp_line[11]))
            up.append(float(temp_line[12]))
            ant.append(float(temp_line[13]))
            sig_e.append(float(temp_line[14]))
            sig_n.append(float(temp_line[15]))
            sig_u.append(float(temp_line[16]))
            corr_en.append(float(temp_line[17]))
            corr_eu.append(float(temp_line[18]))
            corr_nu.append(float(temp_line[19]))

        data = enz(station_name=station_name, dates=dates, yyyy_yyyy=yyyy_yyyy, MJD=MJD, week=week, day=day, reflon=reflon, e0=e0, east=east, n0=n0, north=north, u0=u0, up=up, ant=ant, sig_e=sig_e, sig_n=sig_n, sig_u=sig_u, corr_en=corr_en, corr_eu=corr_eu, corr_nu=corr_nu)

    return data


# ------------------------- ANALYSIS -------------------------

def filtStations(stationInfo, minLon, maxLon, minLat, maxLat, outName):
    # Filter master list of GPS stations by geographic coordinates
    print()
    print('Search bounds: ' + str(minLon) + ', ' + str(maxLon) + ', ' + str(minLat) + ', ' + str(maxLat))
    newStations = stationInfo[(stationInfo['Lon'] >= minLon) &
                              (stationInfo['Lon'] <= maxLon) &
                              (stationInfo['Lat'] >= minLat) &
                              (stationInfo['Lat'] <= maxLat)]
    # print(newStations[0])

    with open(outName, 'w') as newList:
        for i in range(len(newStations)):
            newList.write(str(newStations.iloc[i, 0]) + ' ' + str(newStations.iloc[i, 1]) + ' ' + str(newStations.iloc[i, 2]) + ' ' + str(newStations.iloc[i, 3]) + '\n')

        print('Filted list of GPS stations saved to ' + outName)
        print()
    return newStations


def proj2LOS(gpsData, stationList, lookTable):
    """
    Project GPS data into InSAR LOS direction

    INPUT:
    gpsData - Path to GPS data file. Currently only supports UNR format.
    stationList - Table containing station names and locations. Currently only supports UNR format.
    lookTable - look-up table for LOS unit vector at all InSAR pixels.

    OUTPUT:
    stationData - Pandas DataFrame with LOS projection appended to regular GPS data
    """

    # Load GPS data
    stationData = readASCII(gpsData, 'UNR')

    # Load station info
    stationInfo = readASCII(stationList, 'stationInfo')

    # Load LOS vector data
    lookData = readASCII(lookTable, 'lookTable')

    # Convert useful columns to arrays
    east = np.array(stationData['__east(m)'])
    north = np.array(stationData['_north(m)'])
    up = np.array(stationData['____up(m)'])

    # First, get coordinates of GPS station
    stationLon = stationInfo[stationInfo['Site'] == stationData['site'].iloc[0]]['Lon'].iloc[0]
    stationLat = stationInfo[stationInfo['Site'] == stationData['site'].iloc[0]]['Lat'].iloc[0]

    print(stationData['site'].iloc[0] + ' coordinates:')
    print(stationLon)
    print(stationLat)

    # Find InSAR pixel closest to selected GPS station
    dLon = lookData['lon'] - stationLon
    dLat = lookData['lat'] - stationLat
    gpsPixel = lookData[(abs(dLon) == min(abs(dLon))) & (abs(dLat) == min(abs(dLat)))]

    print()
    print('Pixel info for ' + stationData['site'].iloc[0] + ':')
    print(gpsPixel)

    Ue = np.array(gpsPixel['Ue'])
    Un = np.array(gpsPixel['Un'])
    Uz = np.array(gpsPixel['Uz'])

    # Now, we want to find the component of displacement D along LOS unit vector U.
    D = np.column_stack((east, north, up))
    U = np.column_stack((Ue, Un, Uz))

    # print(D.shape, U.transpose().shape)
    projLOS = np.dot(D, U.transpose())
    # projLOS = np.dot(D, U.transpose()) / (Ue**2 + Un**2 + Uz**2)**0.25

    # Add new column to stationData which corresponds to LOS-projected data
    stationData['LOS'] = pd.DataFrame(projLOS)

    return stationData


def difference(ts1, ts2, data1, data2):
    """
    Compute difference between two time series.
    Data must be DataFrames containing a datetime 'Date' column.

    INPUT:
    ts1 - first time series
    ts2 - second time series
    date_col - name of date column (must be common between both DataFrames!)
    data_col - name of data column (must be common between both DataFrames!)

    OUTPUT:
    difference  - DataFrame containing Date and Difference columns
    """
    # Merge time series on common dates
    difference = pd.merge(ts1, ts2, how='inner', on=['Date'])
    difference['Difference'] = [difference[data1][i] - difference[data2][i] for i in range(len(difference))]

    return difference


def diffGPS(stationData1, stationData2, dateCol, dataCol):
    # DEPRECIATED - USE DIFFERENCE INTEAD
    # Calculate a differential GPS time series for two GPS stations
    # --------------------------------------------------------------
    # INPUT:
    #   stationData1 - standard GPS dataframe for first station
    #   stationData2 - standard GPS dataframe for second station
    #   dateCol - field in dataframe containing datetime objects
    #   dataCol - relevant data column to difference
    # --------------------------------------------------------------
    # OUTPUT:
    #   dateList - list of all dates within combined station data
    #              coverage period
    #   difference - differenced time series. Dates where data from
    #                both stations is not available are set to Nan.
    # --------------------------------------------------------------

    # Convert date columns to lists
    dates1 = list(stationData1[dateCol])
    dates2 = list(stationData2[dateCol])

    # Convert LOS columns to arrays
    data1 = list(stationData1[dataCol])
    data2 = list(stationData2[dataCol])

    # Initiate new date array with EACH DATE between first and last dates in dataset
    start = min([dates1[0], dates2[0]])
    end = max(dates1[-1], dates2[-1])
    tempDays = end - start
    days = tempDays.days

    dateList = np.array([start + np.timedelta64(x, 'D') for x in range(days + 1)])
    difference = []

    # Do differencing
    for date in dateList:
        # Find dates where both datasets have a data point
        if date in dates1 and date in dates2:
            # Get respective indicies and use them to calulate the difference in station position at date
            i1 = dates1.index(date)
            i2 = dates2.index(date)
            difference.append(data1[i1] - data2[i2])
        else:
            difference.append(np.nan)

    return dateList, difference


def calcDisp(stationData, start, end, window, plot):
    # stationData - dataframe with GPS station time series data
    # start   - start date for displacement calculation
    # end     - end date for displacement calculation
    # window  - number of days within start/end date to include in averaging calulation

    print('Calulating displacement for ' + stationData['site'].iloc[0])
    # Create temporary dataframe which contains data withing the date range of (start - window) to (end + window)
    tempData = stationData[(stationData['dates'] >= start - dt.timedelta(days=window)) &
                           (stationData['dates'] <= end + dt.timedelta(days=window))]

    # Window method
    method = 'mean'

    if method == 'mean':
        # Calculate displacement using specified window size
        # EAST
        startPtE = tempData['__east(m)'].iloc[0:(2 * window + 1)]  # Initial displacement measurements to average over
        startErrE = tempData['sig_e(m)'].iloc[0:(2 * window + 1)]  # Initial displacement measurement uncertainties
        startMeanE = np.mean(startPtE)                            # Mean of initial displacements
        startSumSigE = (sum(startErrE**2))**0.5                   # Uncertainty in sum of initial points
        startFracSigE = ((startSumSigE / sum(startPtE))**2)**0.5    # Fractional uncertainty in mean of initial points
        startSigE = abs(startMeanE) * startFracSigE                 # True uncertainty in mean of intitial points

        endPtE = tempData['__east(m)'].iloc[(-2 * window - 1):]  # Final displacement measurements to average over
        endErrE = tempData['sig_e(m)'].iloc[(-2 * window - 1):]  # Final displacement measurement uncertainties
        endMeanE = np.mean(endPtE)                              # Mean of final displacements
        endSumSigE = (sum(endErrE**2))**0.5                     # Uncertainty in sum of final points
        endFracSigE = ((endSumSigE / sum(endPtE))**2)**0.5        # Fractional uncertainty in mean of final points
        endSigE = abs(endMeanE) * endFracSigE                     # True uncertainty in mean of final points

        dispE = endMeanE - startMeanE
        sigE = (endSigE**2 + startSigE**2)**0.5

        # NORTH
        startPtN = tempData['_north(m)'].iloc[0:(2 * window + 1)]  # Initial displacement measurements to average over
        startErrN = tempData['sig_n(m)'].iloc[0:(2 * window + 1)]  # Initial displacement measurement uncertainties
        startMeanN = np.mean(startPtN)                            # Mean of initial displacements
        startSumSigN = (sum(startErrN**2))**0.5                   # Uncertainty in sum of initial points
        startFracSigN = ((startSumSigN / sum(startPtN))**2)**0.5    # Fractional uncertainty in mean of initial points
        startSigN = abs(startMeanN) * startFracSigN                 # True uncertainty in mean of intitial points

        endPtN = tempData['_north(m)'].iloc[(-2 * window - 1):]  # Final displacement measurements to average over
        endErrN = tempData['sig_n(m)'].iloc[(-2 * window - 1):]  # Final displacement measurement uncertainties
        endMeanN = np.mean(endPtN)                              # Mean of final displacements
        endSumSigN = (sum(endErrN**2))**0.5                     # Uncertainty in sum of final points
        endFracSigN = ((endSumSigN / sum(endPtN))**2)**0.5        # Fractional uncertainty in mean of final points
        endSigN = abs(endMeanN) * endFracSigN                     # True uncertainty in mean of final points

        dispN = endMeanN - startMeanN
        sigN = (endSigN**2 + startSigN**2)**0.5

        # UP
        startPtU = tempData['____up(m)'].iloc[0:(2 * window + 1)]  # Initial displacement measurements to average over
        startErrU = tempData['sig_u(m)'].iloc[0:(2 * window + 1)]  # Initial displacement measurement uncertainties
        startMeanU = np.mean(startPtU)                            # Mean of initial displacements
        startSumSigU = (sum(startErrU**2))**0.5                   # Uncertainty in sum of initial points
        startFracSigU = ((startSumSigU / sum(startPtU))**2)**0.5    # Fractional uncertainty in mean of initial points
        startSigU = abs(startMeanU) * startFracSigU                 # True uncertainty in mean of intitial points

        endPtU = tempData['____up(m)'].iloc[(-2 * window - 1):]  # Final displacement measurements to average over
        endErrU = tempData['sig_u(m)'].iloc[(-2 * window - 1):]  # Final displacement measurement uncertainties
        endMeanU = np.mean(endPtU)                              # Mean of final displacements
        endSumSigU = (sum(endErrU**2))**0.5                     # Uncertainty in sum of final points
        endFracSigU = ((endSumSigU / sum(endPtU))**2)**0.5        # Fractional uncertainty in mean of final points
        endSigU = abs(endMeanU) * endFracSigU                     # True uncertainty in mean of final points

        dispU = endMeanU - startMeanU
        sigU = (endSigU**2 + startSigU**2)**0.5

        if plot == 'yes':
            fig = plt.figure(figsize=(8, 8.25))

            ax1 = plt.subplot(311)
            ax1.set_title(stationData['site'].iloc[0])
            ax1.set_xticks(range(0, 11, 2))

            ax2 = plt.subplot(312)
            ax2.set_ylabel('Calculated displacement (m)')
            ax2.set_xticks(range(0, 11, 2))

            ax3 = plt.subplot(313)
            ax3.set_xlabel('Window size (days)')
            ax3.set_xticks(range(0, 11, 2))

            for tempWindow in range(0, 11):
                tempDispE = np.mean(tempData['__east(m)'].iloc[(-2 * tempWindow - 1):]) - np.mean(tempData['__east(m)'].iloc[0:(2 * tempWindow + 1)])
                tempDispN = np.mean(tempData['_north(m)'].iloc[(-2 * tempWindow - 1):]) - np.mean(tempData['_north(m)'].iloc[0:(2 * tempWindow + 1)])
                tempDispU = np.mean(tempData['____up(m)'].iloc[(-2 * tempWindow - 1):]) - np.mean(tempData['____up(m)'].iloc[0:(2 * tempWindow + 1)])

                ax1.scatter(tempWindow, tempDispE, s=10, c='black')
                ax2.scatter(tempWindow, tempDispN, s=10, c='black')
                ax3.scatter(tempWindow, tempDispU, s=10, c='black')

            ax1.scatter(window, dispE, s=20, c='C0')
            ax2.scatter(window, dispN, s=20, c='C0')
            ax3.scatter(window, dispU, s=20, c='C0')

            plt.show()

    elif method == 'line':
        # Fit linear trend to each component time series
        fitE = np.polyfit(tempData['yyyy.yyyy'], tempData['__east(m)'], 1)
        fitN = np.polyfit(tempData['yyyy.yyyy'], tempData['_north(m)'], 1)
        fitU = np.polyfit(tempData['yyyy.yyyy'], tempData['____up(m)'], 1)

        # Calculate net displacement from
        dispE = fitE[0] + fitE[1]

        # Show linear fits
        fig = plt.figure(figsize=(8, 8.25))

        ax1 = plt.subplot(311)
        ax1.scatter(tempData['dates'], tempData['__east(m)'], s=1)
        ax1.plot(tempData['dates'], tempData['yyyy.yyyy'] * fitE[0] + fitE[1], 'r')
        ax1.set_ylabel('East displacement (m)')

        ax2 = plt.subplot(312)
        ax2.scatter(tempData['dates'], tempData['_north(m)'], s=1)
        ax2.plot(tempData['dates'], tempData['yyyy.yyyy'] * fitN[0] + fitN[1], 'r')
        ax2.set_ylabel('North displacement (m)')

        ax3 = plt.subplot(313)
        ax3.scatter(tempData['dates'], tempData['____up(m)'], s=1)
        ax3.plot(tempData['dates'], tempData['yyyy.yyyy'] * fitU[0] + fitU[1], 'r')
        ax3.set_ylabel('Vertical displacement (m)')
        ax3.set_xlabel('Date')
        plt.show()

    return dispE, dispN, dispU, sigE, sigN, sigU


def clipTimeSeries(dates, data, start, end):
    # Clip time series based on start/end dates and set initial value to zero.
    # INPUT:
    #     dates - dates in time series
    #     data - time series data points
    #     start - start date (datetime object)
    #     end - end date (datetime object)
    # OUTPUT:
    #     clipDates - time series dates between start and end (inclusive).
    #                 If end date is not in time series, clippedDates will
    #                 end with the lastavailable date.
    #     clipData - data points corresponding to dates in clipDates. All
    #                points are referenced to the first point, which is
    #                set to zero.

    if start in dates:

        print('Initial position: ' + str(data[list(dates).index(start)]))
        clipData = np.array(data) - data[list(dates).index(start)]

        try:
            clipDates = dates[list(dates).index(start):list(dates).index(end)]
            clipData = clipData[list(dates).index(start):list(dates).index(end)]
        except ValueError:
            clipDates = dates[list(dates).index(start):]
            clipData = clipData[list(dates).index(start):]
            print('End date ' + end.strftime('%Y-%m-%d') + ' not in dataset')

    else:
        clipDates = []
        clipData = []
        print('Start date ' + start.strftime('%Y-%m-%d') + 'not in dataset')

    return clipDates, clipData


# ------------------------- OUTPUT -------------------------

def stationTimeSeries(dates, gps_data, color, start_end, ax):

    # Get displacements
    plot_displacements = []

    # First find start date
    z_init = 999
    search_date = start_end[0]

    while z_init == 999:
        print('Looking for ' + search_date.strftime('%Y-%m-%d'))
        for i in range(len(dates)):
            if search_date.strftime('%Y%m%d') == dates[i].strftime('%Y%m%d'):
                plot_dates = dates[i:]
                print('GPS time series start: ' + str(plot_dates.iloc[0]))
                plot_data = gps_data[i:]
                z_init = gps_data[i]
                break

        search_date = search_date + dt.timedelta(days=1)

    search_date = search_date - dt.timedelta(days=1)
    print("Initial value: " + str(z_init))

    for value in plot_data:
        plot_displacements.append(value - z_init)

    # fig=plt.figure(figsize=(14, 8))

    ax.grid()
    ax.scatter(plot_dates, plot_displacements, c=color, marker='.', zorder=99)

    # ax1.set_aspect(30)
    ax.set_xlim(start_end[0], start_end[1])
    # ax1.set_ylim(min(plot_displacements) - 0.005, max(plot_displacements) + 0.005)

    # plt.xlabel('Date')
    # plt.ylabel('Vertical displacement (m)')

    # plt.show()


def plotBaseline(baselineDates, baselineChange, fileName, component):

    fig = plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(111)
    plt.grid()
    ax1.scatter(baselineDates, baselineChange, marker='.', zorder=99)
    # ax1.set_aspect(30)
    ax1.set_xlim(baselineDates[0], baselineDates[-1])
    # ax1.set_ylim(min(baselineChange) - 0.0005, max(baselineChange) + 0.0005)
    plt.title(component)
    plt.xlabel('Date')
    plt.ylabel('Length change (m)')

    plt.show()
    # plt.savefig(fileName + '.eps')
    # plt.close()


def exportASCII(stationData, outName):
    with open(outName, 'w') as newList:
        for i in range(len(stationData)):

            newList.write(str(stationData.iloc[i, 0]) + ' ' +
                          str(stationData.iloc[i, 1]) + ' ' +
                          str(stationData.iloc[i, 2]) + ' ' +
                          str(stationData.iloc[i, 3]) + ' ' +
                          str(stationData.iloc[i, 4]) + ' ' +
                          str(stationData.iloc[i, 5]) + ' ' +
                          str(stationData.iloc[i, 6]) + ' ' +
                          str(stationData.iloc[i, 7]) + ' ' +
                          str(stationData.iloc[i, 8]) + ' ' +
                          str(stationData.iloc[i, 9]) + '\n')

        print("Station displacements saved to ASCII file " + outName)
        print()


if __name__ == "__main__":
    driver()

    # DEPRECIATED - use diffGPS instead
    """
    def calcBaseline(station1, station2, start, end, component):
        # ------------------------- DESCRIPTION: -------------------------
        # Calculates daily baseline lenghts and length changes between two input
        # GPS stations over given epoch using x, y, and z components. This function
        # assumes input data is formatted as described below (Nevada Geodetic
        # Laboratory convention), with the addition of a serial data number column.

        # INPUT:
        #   station1  - Time series data for first GPS station (dataGPS tuple)
        #   station2  - Time series data for second GPS station (dataGPS tuple)
        #   start - Start date of observation period (YYYYMMDD string)
        #   end   - End date of observation period (YYYYMMDD string)

        # OUTPUT:
        #   baselineData - three-column cell array containing dates where both
        #   stations have data, baseline lengths, and

        # Make namedTuple for GPS data
        enz = collections.namedtuple('dataGPS', ['station_name', 'dates', 'yyyy_yyyy', 'MJD', 'week', 'day', 'reflon', 'e0', 'east', 'n0', 'north', 'u0', 'up', 'ant', 'sig_e', 'sig_n', 'sig_u', 'corr_en', 'corr_eu', 'corr_nu'])

        # Find number of days in date range
        start_dt = dt.datetime.strptime(start, '%Y%m%d')
        end_dt = dt.datetime.strptime(end, '%Y%m%d')
        numDays = (end_dt - start_dt).days + 1

        print()
        print('Total of ' + str(numDays) + ' days between ' + start_dt.strftime('%Y%m%d') + ' and ' + end_dt.strftime('%Y%m%d'))

        # Create list of each day in range
        days_in_ts = [start_dt + dt.timedelta(days=x) for x in range(numDays)]
        print()
        print('Making list of dates between ' + days_in_ts[0].strftime('%Y%m%d') + ' and ' + days_in_ts[-1].strftime('%Y%m%d'))

        # -------------------------- STEP 1 --------------------------
        # Find data points in selected date range for each station
        # First intiate empty lists to go into GPS tuple
        station_name = []; dates = []; yyyy_yyyy = []; MJD = []; week = []; day = []; reflon = []; e0 = []; east = []; n0 = []; north = []; u0 = []; up = []; ant = []; sig_e = []; sig_n = []; sig_u = []; corr_en = []; corr_eu = []; corr_nu = []

        print()
        print(station1.station_name[0] + ' has ' + str(len(station1.station_name)) + ' data points between ' + min(station1.dates).strftime('%Y%m%d') + ' and ' + max(station1.dates).strftime('%Y%m%d'))

        # Add data points if they fall into specified date range
        for i in range(len(station1.dates)):
            date_str = station1.dates[i].strftime('%Y%m%d')
            if date_str >= start and date_str <= end:
                station_name.append(station1.station_name[i])
                dates.append(station1.dates[i])
                yyyy_yyyy.append(station1.yyyy_yyyy[i])
                MJD.append(station1.MJD[i])
                week.append(station1.week[i])
                day.append(station1.day[i])
                reflon.append(station1.reflon[i])
                e0.append(station1.e0[i])
                east.append(station1.east[i])
                n0.append(station1.n0[i])
                north.append(station1.north[i])
                u0.append(station1.u0[i])
                up.append(station1.up[i])
                ant.append(station1.ant[i])
                sig_e.append(station1.sig_e[i])
                sig_n.append(station1.sig_n[i])
                sig_u.append(station1.sig_u[i])
                corr_en.append(station1.corr_en[i])
                corr_eu.append(station1.corr_eu[i])
                corr_nu.append(station1.corr_nu[i])

        # Create new data tuple for selected date range
        clipped1 = enz(station_name=station_name, dates=dates, yyyy_yyyy=yyyy_yyyy, MJD=MJD, week=week, day=day, reflon=reflon, e0=e0, east=east, n0=n0, north=north, u0=u0, up=up, ant=ant, sig_e=sig_e, sig_n=sig_n, sig_u=sig_u, corr_en=corr_en, corr_eu=corr_eu, corr_nu=corr_nu)

        print(clipped1.station_name[0] + ' has ' + str(len(clipped1.station_name)) + ' data points between ' + clipped1.dates[0].strftime('%Y%m%d') + ' and ' + clipped1.dates[-1].strftime('%Y%m%d'))

        print()

        for i in range(len(clipped1.dates)):
            print(station_name[i] + ' ' + dates[i].strftime('%Y%m%d')
                 + ' ' + str(yyyy_yyyy[i])
                 + ' ' + str(MJD[i])
                 + ' ' + str(week[i])
                 + ' ' + str(day[i])
                 + ' ' + str(reflon[i])
                 + ' ' + str(e0[i])
                 + ' ' + str(east[i])
                 + ' ' + str(n0[i])
                 + ' ' + str(north[i])
                 + ' ' + str(u0[i])
                 + ' ' + str(up[i])
                 + ' ' + str(ant[i])
                 + ' ' + str(sig_e[i])
                 + ' ' + str(sig_n[i])
                 + ' ' + str(sig_u[i])
                 + ' ' + str(corr_en[i])
                 + ' ' + str(corr_eu[i])
                 + ' ' + str(corr_nu[i]))

        # Reset temporary lists for second station
        station_name = []; dates = []; yyyy_yyyy = []; MJD = []; week = []; day = []; reflon = []; e0 = []; east = []; n0 = []; north = []; u0 = []; up = []; ant = []; sig_e = []; sig_n = []; sig_u = []; corr_en = []; corr_eu = []; corr_nu = []

        print()
        print(station2.station_name[0] + ' has ' + str(len(station2.station_name)) + ' data points between ' + min(station2.dates).strftime('%Y%m%d') + ' and ' + max(station2.dates).strftime('%Y%m%d'))

        # Add data points if they fall into specified date range
        for i in range(len(station2.dates)):
            date_str = station2.dates[i].strftime('%Y%m%d')
            if date_str >= start and date_str <= end:
                station_name.append(station2.station_name[i])
                dates.append(station2.dates[i])
                yyyy_yyyy.append(station2.yyyy_yyyy[i])
                MJD.append(station2.MJD[i])
                week.append(station2.week[i])
                day.append(station2.day[i])
                reflon.append(station2.reflon[i])
                e0.append(station2.e0[i])
                east.append(station2.east[i])
                n0.append(station2.n0[i])
                north.append(station2.north[i])
                u0.append(station2.u0[i])
                up.append(station2.up[i])
                ant.append(station2.ant[i])
                sig_e.append(station2.sig_e[i])
                sig_n.append(station2.sig_n[i])
                sig_u.append(station2.sig_u[i])
                corr_en.append(station2.corr_en[i])
                corr_eu.append(station2.corr_eu[i])
                corr_nu.append(station2.corr_nu[i])

        # Create new data tuple for selected date range
        clipped2 = enz(station_name=station_name, dates=dates, yyyy_yyyy=yyyy_yyyy, MJD=MJD, week=week, day=day, reflon=reflon, e0=e0, east=east, n0=n0, north=north, u0=u0, up=up, ant=ant, sig_e=sig_e, sig_n=sig_n, sig_u=sig_u, corr_en=corr_en, corr_eu=corr_eu, corr_nu=corr_nu)

        print(clipped2.station_name[0] + ' has ' + str(len(clipped2.station_name)) + ' data points between ' + clipped2.dates[0].strftime('%Y%m%d') + ' and ' + clipped2.dates[-1].strftime('%Y%m%d'))

        for i in range(len(clipped2.dates)):
            print(station_name[i] + ' ' + dates[i].strftime('%Y%m%d')
                 + ' ' + str(yyyy_yyyy[i])
                 + ' ' + str(MJD[i])
                 + ' ' + str(week[i])
                 + ' ' + str(day[i])
                 + ' ' + str(reflon[i])
                 + ' ' + str(e0[i])
                 + ' ' + str(east[i])
                 + ' ' + str(n0[i])
                 + ' ' + str(north[i])
                 + ' ' + str(u0[i])
                 + ' ' + str(up[i])
                 + ' ' + str(ant[i])
                 + ' ' + str(sig_e[i])
                 + ' ' + str(sig_n[i])
                 + ' ' + str(sig_u[i])
                 + ' ' + str(corr_en[i])
                 + ' ' + str(corr_eu[i])
                 + ' ' + str(corr_nu[i]))

        # -------------------------- STEP 2 --------------------------
        # Align both station datasets adding empty spacer lists for dates with no data

        # Set loop-independent index for GPS data arrays.
        j1 = 0
        j2 = 0
        save_j1 = 0
        save_j2 = 0

        # Initialize empty arrays for aligned data
        station_name = []; dates = []; yyyy_yyyy = []; MJD = []; week = []; day = []; reflon = []; e0 = []; east = []; n0 = []; north = []; u0 = []; up = []; ant = []; sig_e = []; sig_n = []; sig_u = []; corr_en = []; corr_eu = []; corr_nu = []

        while len(corr_nu) < numDays:
            station_name.append([])
            dates.append([])
            yyyy_yyyy.append([])
            MJD.append([])
            week.append([])
            day.append([])
            reflon.append([])
            e0.append([])
            east.append([])
            n0.append([])
            north.append([])
            u0.append([])
            up.append([])
            ant.append([])
            sig_e.append([])
            sig_n.append([])
            sig_u.append([])
            corr_en.append([])
            corr_eu.append([])
            corr_nu.append([])

        print()
        print('Confirming length of empty lists is equal to number of days in date range: ' + str(len(reflon)) + ' = ' + str(numDays))

        # Align GPS data arrays to have indexing consistent with the selected date range. Days with no GPS data are left empty.
        print()
        print('Number of data points from ' + clipped1.station_name[0] + ': ' + str(len(clipped1.yyyy_yyyy)))

        for i in range(len(days_in_ts)):
            station_name[i] = clipped1.station_name[0]

            while j1 < len(clipped1.dates):
                if days_in_ts[i].strftime('%Y%m%d') == clipped1.dates[j1].strftime('%Y%m%d'):
                    # print(days_in_ts[i].strftime('%Y%m%d') + ' == ' + clipped1.dates[j1].strftime('%Y%m%d'))

                    # Add all the data to the new tuple
                    # print('Adding data for ' + clipped1.dates[j1].strftime('%Y%m%d') + ' to [' + str(i) + '] ')
                    dates[i] = clipped1.dates[j1]
                    yyyy_yyyy[i] = clipped1.yyyy_yyyy[j1]
                    MJD[i] = clipped1.MJD[j1]
                    week[i] = clipped1.week[j1]
                    day[i] = clipped1.day[j1]
                    reflon[i] = clipped1.reflon[j1]
                    e0[i] = clipped1.e0[j1]
                    east[i] = clipped1.east[j1]
                    n0[i] = clipped1.n0[j1]
                    north[i] = clipped1.north[j1]
                    u0[i] = clipped1.u0[j1]
                    up[i] = clipped1.up[j1]
                    ant[i] = clipped1.ant[j1]
                    sig_e[i] = (clipped1.sig_e[j1])
                    sig_n[i] = clipped1.sig_n[j1]
                    sig_u[i] = clipped1.sig_u[j1]
                    corr_en[i] = clipped1.corr_en[j1]
                    corr_eu[i] = clipped1.corr_eu[j1]
                    corr_nu[i] = clipped1.corr_nu[j1]

                    # Save index of most recently mathched data point
                    save_j1 = j1

                    # Move to next data point
                    j1 += 1

                    # print(up[:i])

                    break

                elif days_in_ts[i].strftime('%Y%m%d') != clipped1.dates[j1].strftime('%Y%m%d'):
                    # print('Checking next data point')
                    j1 += 1

            # if up[i] == []:
                # print(station_name[i] + ' has no data on ' + days_in_ts[i].strftime('%Y%m%d'))

            if j1 == len(clipped1.dates):
                    dates[i] = np.nan
                    yyyy_yyyy[i] = np.nan
                    MJD[i] = np.nan
                    week[i] = np.nan
                    day[i] = np.nan
                    reflon[i] = np.nan
                    e0[i] = np.nan
                    east[i] = np.nan
                    n0[i] = np.nan
                    north[i] = np.nan
                    u0[i] = np.nan
                    up[i] = np.nan
                    ant[i] = np.nan
                    sig_e[i] = np.nan
                    sig_n[i] = np.nan
                    sig_u[i] = np.nan
                    corr_en[i] = np.nan
                    corr_eu[i] = np.nan
                    corr_nu[i] = np.nan

            # Start search for next date after last matched date
            j1 = save_j1

        aligned1 = enz(station_name=station_name, dates=dates, yyyy_yyyy=yyyy_yyyy, MJD=MJD, week=week, day=day, reflon=reflon, e0=e0, east=east, n0=n0, north=north, u0=u0, up=up, ant=ant, sig_e=sig_e, sig_n=sig_n, sig_u=sig_u, corr_en=corr_en, corr_eu=corr_eu, corr_nu=corr_nu)

        # ----------------- Repeat for second station -----------------
        # Initialize empty arrays for aligned data
        station_name = []; dates = []; yyyy_yyyy = []; MJD = []; week = []; day = []; reflon = []; e0 = []; east = []; n0 = []; north = []; u0 = []; up = []; ant = []; sig_e = []; sig_n = []; sig_u = []; corr_en = []; corr_eu = []; corr_nu = []

        while len(corr_nu) < numDays:
            station_name.append(([]))
            dates.append([])
            yyyy_yyyy.append([])
            MJD.append([])
            week.append([])
            day.append([])
            reflon.append([])
            e0.append([])
            east.append([])
            n0.append([])
            north.append([])
            u0.append([])
            up.append([])
            ant.append([])
            sig_e.append([])
            sig_n.append([])
            sig_u.append([])
            corr_en.append([])
            corr_eu.append([])
            corr_nu.append([])

        print()
        print('Confirming length of empty lists is equal to number of days in date range: ' + str(len(reflon)) + ' = ' + str(numDays))
        print()
        print('Number of data points from ' + clipped2.station_name[0] + ': ' + str(len(clipped2.yyyy_yyyy)))

        for i in range(len(days_in_ts)):
            station_name[i] = clipped2.station_name[0]

            while j2 < len(clipped2.dates):
                if days_in_ts[i].strftime('%Y%m%d') == clipped2.dates[j2].strftime('%Y%m%d'):
                    # print(days_in_ts[i].strftime('%Y%m%d') + ' == ' + clipped2.dates[j2].strftime('%Y%m%d'))

                    # Add all the data to the new tuple
                    # print('Adding data for ' + clipped2.dates[j2].strftime('%Y%m%d') + ' to [' + str(i) + '] ')

                    dates[i] = clipped2.dates[j2]
                    yyyy_yyyy[i] = clipped2.yyyy_yyyy[j2]
                    MJD[i] = clipped2.MJD[j2]
                    week[i] = clipped2.week[j2]
                    day[i] = clipped2.day[j2]
                    reflon[i] = clipped2.reflon[j2]
                    e0[i] = clipped2.e0[j2]
                    east[i] = clipped2.east[j2]
                    n0[i] = clipped2.n0[j2]
                    north[i] = clipped2.north[j2]
                    u0[i] = clipped2.u0[j2]
                    up[i] = clipped2.up[j2]
                    ant[i] = clipped2.ant[j2]
                    sig_e[i] = (clipped2.sig_e[j2])
                    sig_n[i] = clipped2.sig_n[j2]
                    sig_u[i] = clipped2.sig_u[j2]
                    corr_en[i] = clipped2.corr_en[j2]
                    corr_eu[i] = clipped2.corr_eu[j2]
                    corr_nu[i] = clipped2.corr_nu[j2]

                    # Save index of most recently mathched data point
                    save_j2 = j2

                    # Move to next data point
                    j2 += 1

                    break

                elif days_in_ts[i].strftime('%Y%m%d') != clipped2.dates[j2].strftime('%Y%m%d'):
                    j2 += 1


            if j2 == len(clipped2.dates):
                    dates[i] = np.nan
                    yyyy_yyyy[i] = np.nan
                    MJD[i] = np.nan
                    week[i] = np.nan
                    day[i] = np.nan
                    reflon[i] = np.nan
                    e0[i] = np.nan
                    east[i] = np.nan
                    n0[i] = np.nan
                    north[i] = np.nan
                    u0[i] = np.nan
                    up[i] = np.nan
                    ant[i] = np.nan
                    sig_e[i] = np.nan
                    sig_n[i] = np.nan
                    sig_u[i] = np.nan
                    corr_en[i] = np.nan
                    corr_eu[i] = np.nan
                    corr_nu[i] = np.nan

            # Start search for next date after last matched date
            j2 = save_j2

        aligned2 = enz(station_name=station_name, dates=dates, yyyy_yyyy=yyyy_yyyy, MJD=MJD, week=week, day=day, reflon=reflon, e0=e0, east=east, n0=n0, north=north, u0=u0, up=up, ant=ant, sig_e=sig_e, sig_n=sig_n, sig_u=sig_u, corr_en=corr_en, corr_eu=corr_eu, corr_nu=corr_nu)


        # Calculate the change in GPS station displacement during the observation period. First, find the first data point
        initialDispE1 = np.nan;
        initialDispE2 = np.nan;
        initialDispN1 = np.nan;
        initialDispN2 = np.nan;
        initialDispV1 = np.nan;
        initialDispV2 = np.nan;
        i = 0;

        while math.isnan(initialDispE1):
            # Start from first day in time series. If tested time series date has data, set this to be the reference (position = 0).
            if math.isnan(aligned1.east[i]) == False:
                initialDispE1 = aligned1.east[i]
                initialDispN1 = aligned1.north[i]
                initialDispV1 = aligned1.up[i]
            else:
                i += 1

        # Reset for second station
        i = 0;
        while math.isnan(initialDispE2):
            # Start from first day in time series. If tested time series date has data, set this to be the reference (position = 0).
            if math.isnan(aligned2.east[i]) == False:
                initialDispE2 = aligned2.east[i]
                initialDispN2 = aligned2.north[i]
                initialDispV2 = aligned2.up[i]
            else:
                i += 1


        print('Intial displacements for ' + aligned1.station_name[i] + ': ' + str(initialDispE1) + ', ' + str(initialDispN1) + ', ' + str(initialDispV1))
        print('Intial displacements for ' + aligned2.station_name[i] + ': ' + str(initialDispE2) + ', ' + str(initialDispN2) + ', ' + str(initialDispV2))


        # Create new list for incremental GPS displacements
        baselineDates = []
        dispE1 = []
        dispE2 = []
        dispN1 = []
        dispN2 = []
        dispV1 = []
        dispV2 = []

        # Subtract initial displacement values for each
        for i in range(len(days_in_ts)):
            # If there is GPS data for both stations on a given date, subtract the initial displacement value and append date and net displacement to new lists
            if math.isnan(aligned1.east[i]) == False and math.isnan(aligned2.east[i]) == False:
                baselineDates.append(days_in_ts[i])
                dispE1.append(aligned1.east[i] - initialDispE1)
                dispE2.append(aligned2.east[i] - initialDispE2)
                dispN1.append(aligned1.north[i] - initialDispN1)
                dispN2.append(aligned2.north[i] - initialDispN2)
                dispV1.append(aligned1.up[i] - initialDispV1)
                dispV2.append(aligned2.up[i] - initialDispV2)

                # print('Change in ' + aligned1.station_name[i] + '-' + aligned2.station_name[i] + ' baseline length from ' + days_in_ts[0].strftime('%Y%m%d') + ' to ' + days_in_ts[i].strftime('%Y%m%d') + ': ' + str(baselineChange[-1]) + ' m')


        # Compute baseline lengths for given period.
        baselineChange = []
        for i in range(len(baselineDates)):
            if component == 'true':
                baselineChange.append(np.sqrt((dispE1[i] - dispE2[i])**2 + (dispN1[i] - dispN2[i])**2 + (dispV1[i] - dispV2[i])**2))

            elif component == 'east':
                baselineChange.append(dispE1[i] - dispE2[i])

            elif component == 'north':
                baselineChange.append(dispN1[i] - dispN2[i])

            elif component == 'up':
                baselineChange.append(dispV1[i] - dispV2[i])

                print(dispV1[i], dispV2[i], baselineChange[-1])

        return baselineDates, baselineChange
    """
