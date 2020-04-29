import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import glob as glob
import netCDF4 as nc
import subprocess
import datetime as dt
import requests
from matplotlib import cm
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable

"""
A library of Python tools for generating InSAR time series with GMTSAR

"""


# -------------------- DOWNLOADING --------------------

def getOrbits(listURL, orbitURL, dirList, saveDir):
    """
    getOrbits - download Sentinel-1 orbits for a given set of SAFE folders.

    INPUTS:
    listURL = Typically "https://s1qc.asf.alaska.edu/aux_poeorb"
    orbitURL = 'http://aux.sentinel1.eo.esa.int/POEORB' for precise orbit files
    dirList = list of SAFE directories to get orbit files for (i.e. SAFE_filelist)
    saveDir = directory to save orbits in (usually in {asc/des}/data)

    OUTPUTS:
    orbitList = list containing links to download orbit files
    downloadList = List of orbit files to download

    """

    # listURL = "https://s1qc.asf.alaska.edu/aux_poeorb"  # May need to get edited in the future
    # orbitURL = 'http://aux.sentinel1.eo.esa.int/POEORB'
    # dirList = 'SAFE_filelist'
    # saveDir = '.'

    def getOrbitList(url):
        # Gets list of all .EOF filenames from specified URL

        # Scrape current list of S1A/B orbit filenames
        print('Getting list of current orbit files from ' + url + ' ...')
        orbitHTML = requests.get(url)
        tempList = list(orbitHTML.text.split("\n")[4:-7])

        # Save names to list
        orbitList = []
        for line in tempList:
            orbitList.append(line[9:86])

        return orbitList

    def getOrbitURL(orbitList, dirList):
        # Get list of EOF file URLs based on input list of directories
        # orbitList format:
            # S1A_OPER_AUX_POEORB_OPOD_20180228T120602_V20180207T225942_20180209T005942.EOF
            # S1A_OPER_AUX_POEORB_OPOD_20180312T120552_V20180219T225942_20180221T005942.EOF
            # S1A_OPER_AUX_POEORB_OPOD_20180324T120757_V20180303T225942_20180305T005942.EOF
            # ...

        # dirList format:
            # S1A_IW_SLC__1SDV_20191013T135939_20191013T140006_029441_035951_9DBD.SAFE
            # S1A_IW_SLC__1SDV_20191025T135939_20191025T140006_029616_035F52_BCBC.SAFE
            # S1A_IW_SLC__1SDV_20191106T135939_20191106T140006_029791_03657D_4FEF.SAFE
            # ...

        print('Matching filenames from ' + dirList + ' ...')

        # Create reference list of directory satellite IDs and dates
        refList = []
        with open(dirList) as file:
            for line in file:
                refList.append([line[0:3], dt.datetime.strptime(line[17:25], '%Y%m%d')])

        # Find filename for each aquisition in refList
        downloadList = []

        for file in refList:

            for orbit in orbitList:
                # Create string to validate with orbit filenames (does not include upload date)
                searchStr = '_V' + (file[1] - dt.timedelta(days=1)).strftime('%Y%m%d') + 'T225942_' + (file[1] + dt.timedelta(days=1)).strftime('%Y%m%d') + 'T005942.EOF'

                if searchStr in orbit:
                    if file[0] in orbit:
                        downloadList.append(orbit)
                        print(file[0] + ' ' + file[1].strftime('%Y%m%d') + ': Matched')
                        tag = 1

            if tag == 1:
                tag = 0
            else:
                # print('Orbit file not available for ' + file[0] + ' ' + file[1].strftime('%Y%m%d'))
                print(file[0] + ' ' + file[1].strftime('%Y%m%d') + ': NO FILE FOUND')
                tag = 0

        return downloadList

    def downloadOrbits(url, downloadList, saveDir):
        """
        Takes a list of URLS (see description for getOrbitURL) and downloads the appropriate files through the Sentinel-1  Quality Control data portal.
        """

        for file in downloadList:
            print('Downloading ' + file + '...')
            subprocess.call(['wget', url + "/" + file[25:29] + "/" + file[29:31] + "/" + file[31:33] + "/" + file], shell=False)

    orbitList = getOrbitList(listURL)
    downloadList = getOrbitURL(orbitList, dirList)
    downloadOrbits(orbitURL, downloadList, saveDir)

    return orbitList, downloadList


# def getDEM():


def getData(start, end, region, dir, subtype, framerange):
    """
    Download Sentinel-1 SAR data using Alaska Satellite Facility API
    """
    print()

# -------------------- READING --------------------


def readIntfTable(filename):
    """
    Read in interferogram metadata table
    """

    # Read specified file
    intfTable = pd.read_csv(fileName, sep=' ', header=None)
    intfTable.columns = ['Path', 'DateStr', 'Master', 'Repeat', 'TempBaseline', 'OrbitBaseline', 'MeanCorr']

    # Convert date columns to datetime
    intfTable['Master'] = pd.to_datetime(intfTable['Master'], format='%Y%m%d')
    intfTable['Repeat'] = pd.to_datetime(intfTable['Repeat'], format='%Y%m%d')

    # Display some lines
    intfTable.head()

    return intfTable


# -------------------- WRITING --------------------
def makeIntfTable(baselineTable, corrPaths, **kwargs):
    """
    Create Pandas DataFrame with interferogram metadata

    Columns:
    Path - path to intf directory
    DateStr - name of intf directory (e.g. 2019123_2020098)
    Master - datetime object for master scene
    Repeat - datetime object for repeat scene
    tempBaseline - temporal baseline (days)
    OrbitBaseline - orbital baseline (m)
    MeanCorr - mean coherence of interferogram

    ------ INPUT ------
    baselineTable = baseline table DataFrame
    corrPaths = search string to corr.grd files, i.e. '/Users/ellisvavra/LongValley/insar/des/f2/intf_all/*/corr.grd'
    writeTable = write table to 'intf_table.dat'. Hardwired to not overwrite existing files (default = True)
    printTable = print table to command line (default = False)
    region = list containing min/max indicies to define subregion to use in mean calculation, i.e. [0,1500,0,1250] (default = whole interferogram)

    ------ OUTPUT ------
    intfTable
    baselineTable (if loading for the first time)
    """

    # Load baseline table
    baselineTable = pd.read_csv(baselineTable, header=None, sep=' ')

    # Get coherence grid paths
    paths = glob.glob(corrPaths)
    paths.sort()

    # Handle kwargs
    if 'writeTable' in kwargs:
        writeTable = kwargs['writeTable']
    else:
        writeTable = False

    if 'printTable' in kwargs:
        printTable = kwargs['printTable']
    else:
        printTable = False

    if 'region' in kwargs:
        region = kwargs['printTable']
    else:
        # Get dimensions from first file in list
        example = nc.Dataset(paths[0], 'r+', format="NETCDF4")
        region = [0, example.dimensions['y'].size,
                  0, example.dimensions['x'].size]
        example.close()

    # Initiate dataframe and start adding columns
    intfTable = pd.DataFrame()
    intfTable['Path'] = [line[:-9] for line in paths]
    intfTable['DateStr'] = [line[-24:-9] for line in paths]
    intfTable['Master'] = [dt.datetime.strptime(line[-24:-17], '%Y%j') + dt.timedelta(days=1) for line in paths]
    intfTable['Repeat'] = [dt.datetime.strptime(line[-16:-9], '%Y%j') + dt.timedelta(days=1) for line in paths]
    intfTable['TempBaseline'] = (intfTable['Repeat'] - intfTable['Master']).dt.days

    # Loop through all intfs to calculate orbit baselines
    bl = []

    print('Calculating baselines...')
    for i in range(len(intfTable)):
        # Search baselineTable for master baseline
        for j, namestr in enumerate(baselineTable[0]):
            if intfTable['Master'][i].strftime('%Y%m%d') in namestr:
                mbl = baselineTable[4][j]
                break
        # Search baselineTable for repeat baseline
        for j, namestr in enumerate(baselineTable[0]):
            if intfTable['Repeat'][i].strftime('%Y%m%d') in namestr:
                rbl = baselineTable[4][j]
                break

        bl.append(mbl - rbl)

    intfTable['OrbitBaseline'] = bl

    # Now calculate mean coherences
    meancorr = []

    print('Calculating mean coherence...')
    for path in paths:
        meancorr.append(np.nanmean(np.array(nc.Dataset(path, 'r+', format="NETCDF4").variables['z'][region[0]:region[1], region[2]:region[3]])))

    intfTable['MeanCorr'] = meancorr

    # Write table
    if writeTable == True:

        # Write table to file if it does not already exist
        if len(glob.glob('intf_table.dat')) == 0:
            intfTable.to_csv('intf_table.dat', sep=' ', index=False)
            print("Interferogram table written to 'intf_table.dat'")

        # Append index to filename and try again
        else:
            print("'intf_table.dat' already exists...")
            written = False
            i = 1

            while written == False:
                if len(glob.glob('intf_table.dat.{}'.format(i))) == 0:
                    intfTable.to_csv('intf_table.dat.{}'.format(i), sep=' ', index=False)
                    print("Interferogram table written to 'intf_table.dat.{}'".format(i))
                    written = True

                else:
                    i += 1

    return intfTable, baselineTable


def filtIntfTable(intfTable, minMaster, maxMaster, minRepeat, maxRepeat, minTempBaseline, maxTempBaseline, minOrbitBaseline, maxOrbitBaseline, minMeanCorr, maxMeanCorr):
    """
    Filter interferogram table using input parameters
    ---- INPUT ----------------------------------------------
        intfTable - input interferogram table
        minMaster/maxMaster - min/max interferogram master date
        minRepeat/maxRepeat - min/max interferogram Repeat date
        minTempBaseline/maxTempBaseline - min/max temporal baseline length (days)
        minOrbitBaseline/maxOrbitBaseline - min/max temporal baseline length (m)
        minMeanCorr/maxMeanCorr - min/max mean intereferogram coherence
    ---- OUTPUT ---------------------------------------------
        filtIntfTable - table of interferograms meeting specified input parameters
    """

    filtIntfTable = intfTable[(intfTable['Master'] >= minMaster) &
                              (intfTable['Master'] <= maxMaster) &
                              (intfTable['Repeat'] >= minRepeat) &
                              (intfTable['Repeat'] <= maxRepeat) &
                              (intfTable['TempBaseline'] >= minTempBaseline) &
                              (intfTable['TempBaseline'] <= maxTempBaseline) &
                              (intfTable['OrbitBaseline'].abs() >= minOrbitBaseline) &
                              (intfTable['OrbitBaseline'].abs() <= maxOrbitBaseline) &
                              (intfTable['MeanCorr'] >= minMeanCorr) &
                              (intfTable['MeanCorr'] <= maxMeanCorr)]

    # Reset index to 0,1,2,..., n-1
    filtIntfTable = filtIntfTable.reset_index(drop=True)

    return filtIntfTable


def getSceneTable(intfTable):
    """

    """
    # Cut out master/Repeat and coherence columns for concatenating
    df1 = intfTable[['Master', 'TempBaseline', 'OrbitBaseline', 'MeanCorr']]
    df1.columns = ['Scene', 'TempBaseline', 'OrbitBaseline', 'MeanCorr']
    df2 = intfTable[['Repeat', 'TempBaseline', 'OrbitBaseline', 'MeanCorr']]
    df2.columns = ['Scene', 'TempBaseline', 'OrbitBaseline', 'MeanCorr']

    # Combine interferogram columns
    df3 = pd.concat([df1, df2])

    # Get mean scene coherence and intf counts
    time = df3.groupby('Scene')['TempBaseline'].mean()
    orbit = df3.groupby('Scene')['OrbitBaseline'].mean()
    corr = df3.groupby('Scene')['MeanCorr'].mean()
    counts = df3.groupby('Scene').count()['MeanCorr']
    sceneTable = pd.merge(time, orbit, how='inner', on='Scene')
    sceneTable = pd.merge(sceneTable, corr, how='inner', on='Scene')
    sceneTable = pd.merge(sceneTable, counts, how='inner', on='Scene').reset_index()
    sceneTable.columns = ['Date', 'TempBaseline', 'OrbitBaseline', 'MeanCorr', 'Count']

    return sceneTable


# -------------------- ANALYSIS --------------------
def selectIntfs(tablePath, method, tMin, tMax, **kwargs):
    """
    ========================== INPUTs: ==========================
    tablePath - Path to baseline_table.dat generated by GMTSAR
    method - 'sequential' for nth nearest neighbor pair(s) or 'baseline' for temporal baseline

    If 'sequential' is selected:
        tMin - minimum nearest-neighbor pair threshold
        tMax - maximum nearest-neighbor pair threshold

    If 'baseline' is selected:
        tMin - minimum allowable temporal baseline (days)
        tMax - maximum allowable temporal baseline (days)

    Optional:
    requiredDates -
    # orbitMin - minimum allowable orbital baseline (m)
    # orbitMax - maximum allowable orbital baseline (m)
    plotMatrix - set to True to visualize interferogram pairs
    printList - print intfIn to command line
    writeList - write intfIn to file named 'intf.in'

    ========================== OUTPUTS: ==========================
        intfIn - list of interferogram pair filestems formatted for intf_tops.csh
                 ex: 'S1_20141108_ALL_F2:S1_20150823_ALL_F2'


    """
    # Handle kwarg options

    plotMatrix = False
    printList = False
    writeList = False

    if 'plotMatrix' in kwargs:
        plotMatrix = kwargs['plotMatrix']

    if 'printList' in kwargs:
        printList = kwargs['printList']

    if 'writeList' in kwargs:
        writeList = kwargs['writeList']

    # Load data
    pd.set_option('display.float_format', lambda x: '%f' % x)  # Display without exponential
    baselineTable = pd.read_csv(tablePath, header=None, sep=' ')  # Read table
    baselineTable.columns = ['Stem', 'numDate', 'sceneID', 'parBaseline', 'perpBaseline']
    baselineTable['Dates'] = pd.to_datetime(baselineTable['Stem'].str.slice(start=15, stop=23))  # Scrape dates
    baselineTable = baselineTable.sort_values(by='numDate')

    N = len(baselineTable)  # Number of aquisitions
    ID = np.zeros((N, N))  # Interferogram pair key matrix (1 to make intf, 0 for no intf)

    # Print info
    if method == 'sequential':
        print('Creating list of {} to {} nearest-neighbor interferograms...'.format(tMin, tMax))

    elif method == 'baseline':
        print('Creating list of interferograms with baselines between {} to {} days...'.format(tMin, tMax))
    else:
        print("Please set method to 'sequential' or 'baseline'")
        return

    # Use input nearest-neighbor order range tMin and tMax to specify which interferogram keys to 'turn on'
    for masterID, row in enumerate(ID):
        for repeatID, value in enumerate(row):

            # Select pairs based on specified method:
            if method == 'sequential':

                # If difference in numerical scene ID is within allowed range, mark as true
                if abs(masterID - repeatID) != 0 and abs(masterID - repeatID) >= tMin and abs(masterID - repeatID) <= tMax:
                    ID[masterID, repeatID] = 1

            elif method == 'baseline':

                # Get absolute value of perpendicular baseline
                baseline = abs(baselineTable['perpBaseline'][repeatID] - baselineTable['perpBaseline'][masterID])

                # If difference in temporal baseline is within allowed range, mark as true
                if baseline >= tMin and baseline <= tMax:
                    ID[masterID, repeatID] = 1

    # Create master and repeat matricies of dimension N x N
    Masters = np.array(list(baselineTable['Dates'])).repeat(N).reshape(N, N)
    Repeats = np.array(list(baselineTable['Dates'])).repeat(N).reshape(N, N).T

    # Loop through indicies to get pair dates
    intfIn = []
    for i in range(len(ID)):
        for j in range(len(ID[0])):
            if ID[i, j] == 1 and Masters[i, j] < Repeats[i, j]:  # We only want the lower half of the matrix, so ignore intf pairs where 'master' comes after 'repeat'
                intfIn.append('S1_' + Masters[i, j].strftime('%Y%m%d') + '_ALL_F2:S1_' + Repeats[i, j].strftime('%Y%m%d') + '_ALL_F2')

    # Get number of interferogams to make
    n = len(intfIn)
    print('Number of interferograms to be made: {}'.format(n))

    # Print list
    if printList == True:
        for intf in intfIn:
            print(intf)

    # Plot interferogram matrix
    if plotMatrix == True:
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
        plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

        # colors = ID

        # for i in range(len(colors)):
        #     for j in range(len(colors[0]))
        #         if baselineTable['Stem'][i][2] == 'a':
        #             colors

        fig = plt.figure(1, (10, 10))
        ax = plt.gca()
        ax.imshow(ID, 'binary')
        ax.set_ylabel('Master')
        ax.set_title('Repeat', size=10)

    # Write intfIn to file
    if writeList == True:
        print('Writing list of interferograms to intf.in')

        with open('intf.in', 'w') as file:
            for line in intfIn:
                file.write(line + '\n')

    return intfIn


# -------------------- PLOTTING --------------------
def plotNetwork(intfTable, baselineTable):
    """
    Make interferogram network/baseline plot
    """
    # Establish figure
    fig = plt.figure(1, (15, 8))
    ax = plt.gca()
    master = intfTable['Master']
    repeat = intfTable['Repeat']
    mbl = []
    rbl = []

    # Get relative baselines
    for i in range(len(intfTable)):

        # Search baselineTable for master baseline
        for j, namestr in enumerate(baselineTable[0]):
            if intfTable['Master'][i].strftime('%Y%m%d') in namestr:
                mbl.append(baselineTable[4][j])
                break

        # Search baselineTable for repeat baseline
        for j, namestr in enumerate(baselineTable[0]):
            if intfTable['Repeat'][i].strftime('%Y%m%d') in namestr:
                rbl.append(baselineTable[4][j])
                break

    # Manualy set supermaster baseline
    superbl = -48.578476

    # Plot interferogram pairs as lines
    for i in range(len(intfTable)):
        plt.plot([master[i], repeat[i]], [mbl[i] - superbl, rbl[i] - superbl], 'k', lw=0.5)

    # Plot scenes over pair lines
    plt.scatter(master, np.array(mbl) - superbl, s=30, c='C0', zorder=3)
    plt.scatter(repeat, np.array(rbl) - superbl, s=30, c='C0', zorder=3)

    # Figure features

    plt.grid(axis='x', zorder=1)
    plt.xlim(min(master) - dt.timedelta(days=50), max(repeat) + dt.timedelta(days=50))
    # plt.ylim(int(np.ceil((min(baselineTable[4]) - superbl - 50) / 50.0) ) * 50, int(np.floor((max(baselineTable[4]) - superbl + 50) / 50.0)) * 50)
    plt.xlabel('Year')
    plt.ylabel('Baseline relative to master (m)')
    plt.show()

    # return im


def plotScenes(sceneTable, dataType, **kwargs):
    """
    Plot mean coherence of each SAR scene for a given set of interferograms
    ---- INPUT ----------------------------------------------
        intfTable - interferogram list with baseline and coherence data
    ---- OPTIONAL --------------------------------------------
        ax - axis handle for plotting
        cmap - colormap handle
    """
    # Check for passes axis handle
    if 'ax' in kwargs:
        ax = kwargs['ax']
    else:
        ax = plt.gca()

    # Check for passed colorbar axis handle
    if 'cax' in kwargs:
        cax = kwargs['cax']
    else:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad=0.05)

    # Plot settings
    ax.set_xlim(sceneTable['Date'].min(), sceneTable['Date'].max())
    ax.set_xlabel('Date')

    # Make actual plot

    if dataType == 'MeanCorr':
        ax.set_ylabel('Mean coherence')

    elif dataType == 'OrbitBaseline':
        ax.set_ylabel('Mean orbital baseline (m)')

    elif dataType == 'TempBaseline':
        ax.set_ylabel('Mean temporal baseline (days)')

    # Normalize data to make ImageGrid happy
    normData = abs(sceneTable[dataType] / sceneTable[dataType].max())

    # Plot data
    im = ax.scatter(sceneTable['Date'], normData, c=sceneTable['Count'])

    # Get tick labels that correspond with original data
    ticks = np.linspace(0, np.ceil(sceneTable[dataType].max()), 5)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_yticklabels(ticks)

    plt.colorbar(im, cax=cax, label='Number of interferograms')

    return im


def baselineCorrPlot(intfTable, **kwargs):
    """
    Plot temporal baseline versus mean coherence
    ---- INPUT ----------------------------------------------
        intfTable - interferogram list with baseline and coherence data
    ---- OPTIONAL --------------------------------------------
        ax - axis handle for plotting
        cmap - colormap handle
    """

    # Check for passed axis handle
    if 'ax' in kwargs:
        ax = kwargs['ax']
    else:
        ax = plt.gca()

    # Check for passed colorbar axis handle
    if 'cax' in kwargs:
        cax = kwargs['cax']
    else:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad=0.05)

    # Check for passed colormap handle
    if 'cmap' in kwargs:
        cmap = kwargs['cmap']
    else:
        cmap = 'viridis'

    # Plot settings
    ax.set_xlim(intfTable['Master'].min(), intfTable['Repeat'].max())
    # ax.set_ylim(0, 1)
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean coherence')
    # ax.set_aspect(.01)

    # Colorbar business
    baselineRange = list(range(0, int(np.ceil(intfTable['OrbitBaseline'].max() / 10) * 10)))
    n = len(baselineRange)
    cmap = cm.get_cmap(cmap, n)
    Z = [[0, 0], [0, 0]]
    levels = range(0, n)
    CS3 = ax.contourf(Z, levels, cmap=cmap)
    plt.colorbar(CS3, cax=cax, label='Orbital baseline (m)')

    # Make actual plot
    for i in range(len(intfTable)):
        lineColor = np.floor(intfTable['OrbitBaseline'][i]) / n
        im = ax.plot([intfTable['Master'][i], intfTable['Repeat'][i]], [intfTable['MeanCorr'][i], intfTable['MeanCorr'][i]], color=cmap(lineColor))

    return im


# -------------------- DRIVERS --------------------

def analyzeCatalog(sceneTable, intfTable):
    """
    Perform catalog coherence analysis for given interferogram table
    """

    fig = plt.figure(figsize=(15, 9))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(2, 2),
                     axes_pad=.65,
                     aspect=False,
                     cbar_mode='each',
                     cbar_location='top',
                     cbar_pad=0,
                     cbar_size='2.5%',
                     share_all=False
                     )

    plotScenes(sceneTable, 'MeanCorr', ax=grid[0], cax=grid.cbar_axes[0])
    baselineCorrPlot(intfTable, ax=grid[1], cax=grid.cbar_axes[1])
    plotScenes(sceneTable, 'TempBaseline', ax=grid[2], cax=grid.cbar_axes[2])
    plotScenes(sceneTable, 'OrbitBaseline', ax=grid[3], cax=grid.cbar_axes[3])
