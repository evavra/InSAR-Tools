import pandas as pd
import numpy as np
import netCDF4 as nc
import glob as glob
import datetime as dt

"""
Methods for reading, writing, and editing special Pandas Dataframes for GMTSAR time series analysis
"""


def readBaselineTable(fileName):
    """
    Read in baseline table from GMTSAR
    """
    print('Reading baseline table...')
    print()

    baselineTable = pd.read_csv(fileName, header=None, sep=' ')  # Read table
    baselineTable.columns = ['Stem', 'numDate', 'sceneID', 'parBaseline', 'OrbitBaseline']
    baselineTable['Date'] = pd.to_datetime(baselineTable['Stem'].str.slice(start=15, stop=23))  # Scrape dates
    baselineTable = baselineTable.sort_values(by='numDate')
    baselineTable = baselineTable.reset_index(drop=True)

    return baselineTable


def readIntfTable(fileName):
    """
    Read in interferogram metadata table
    """
    print('Reading interferogram table...')
    print()

    # Read specified file
    intfTable = pd.read_csv(fileName, sep=' ', header=0)
    intfTable.columns = ['Path', 'DateStr', 'Master', 'Repeat', 'TempBaseline', 'OrbitBaseline', 'MeanCorr']

    # Convert date columns to datetime
    intfTable['Master'] = pd.to_datetime(intfTable['Master'], format='%Y-%m-%d')
    intfTable['Repeat'] = pd.to_datetime(intfTable['Repeat'], format='%Y-%m-%d')

    # Convert numpy.float64 to float
    # orbitBaseline = intfTable['OrbitBaseline']
    # meanCorr = intfTable['MeanCorr']
    # intfTable['OrbitBaseline'] = [np.float64(bl).item() for bl in orbitBaseline]
    # intfTable['MeanCorr'] = [np.float64(c).item() for c in meanCorr]

    # Display some lines
    # intfTable.head()

    return intfTable


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
        region = kwargs['region']
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


def makeSceneTable(intfTable):
    """
    Generate table with information about each SAR aqquisition based off of input interferogram catalog.

    FIELDS:
    Date - acquisition date
    TempBaseline - mean temporal baseline of all interferograms using scene
    OrbitBaseline - mean orbital baseline of all interferograms using scene
    MeanCorr - mean coherence of all interferograms using scene
    TotalCount - number of interferograms using scene
    MasterCount - number of interferograms using scene as a master
    RepeatCount - number of interferograms using scene as a repeat
    Masters - list of interferograms using scene as a master
    Repeats - list of interferograms using scene as a repeat
    """

    print('Getting scene information...')
    print()

    # Cut out master/Repeat and coherence columns for concatenating
    df1 = intfTable[['Master', 'TempBaseline', 'OrbitBaseline', 'MeanCorr']]
    df1.columns = ['Scene', 'TempBaseline', 'OrbitBaseline', 'MeanCorr']
    df2 = intfTable[['Repeat', 'TempBaseline', 'OrbitBaseline', 'MeanCorr']]
    df2.columns = ['Scene', 'TempBaseline', 'OrbitBaseline', 'MeanCorr']

    # Combine interferogram table columns
    df3 = pd.concat([df1, df2])

    # Aggregate lists of master/repeat interferograms. So sorry for the horrible stacked Dataframe methods.
    masters = intfTable.set_index('Master', append='True').groupby(level=[0, 1], sort=False)['DateStr'].apply(list).reset_index('Master').groupby('Master')['DateStr'].apply(list).reset_index('Master')
    masters.columns = ['Scene', 'Masters']

    repeats = intfTable.set_index('Repeat', append='True').groupby(level=[0, 1], sort=False)['DateStr'].apply(list).reset_index('Repeat').groupby('Repeat')['DateStr'].apply(list).reset_index('Repeat')
    repeats.columns = ['Scene', 'Repeats']

    # Account for start/end scenes not having repeat/master instances
    for date in repeats['Scene']:
        if date not in list(masters['Scene']):
            masters = masters.append({'Scene': date, 'Masters': []}, ignore_index=True)

    for date in masters['Scene']:
        if date not in list(repeats['Scene']):
            repeats = repeats.append({'Scene': date, 'Repeats': []}, ignore_index=True)

    # Reset indicies in date-ascending order
    masters = masters.sort_values('Scene').reset_index(drop=True)
    repeats = repeats.sort_values('Scene').reset_index(drop=True)

    # Get mean scene coherence and intf counts
    time = df3.groupby('Scene')['TempBaseline'].mean()
    orbit = df3.groupby('Scene')['OrbitBaseline'].mean()
    corr = df3.groupby('Scene')['MeanCorr'].mean()
    totalcounts = df3.groupby('Scene').count()['MeanCorr']

    # Merge everything together
    sceneTable = pd.merge(time, orbit, how='inner', on='Scene')
    sceneTable = pd.merge(sceneTable, corr, how='inner', on='Scene')
    sceneTable = pd.merge(sceneTable, totalcounts, how='inner', on='Scene').reset_index()
    sceneTable['MasterCount'] = [len(intfList) for intfList in masters['Masters']]
    sceneTable['RepeatCount'] = [len(intfList) for intfList in repeats['Repeats']]
    sceneTable = pd.merge(sceneTable, masters, how='inner', on='Scene')
    sceneTable = pd.merge(sceneTable, repeats, how='inner', on='Scene')
    sceneTable.columns = ['Date', 'TempBaseline', 'OrbitBaseline', 'MeanCorr', 'TotalCount',
                          'MasterCount', 'RepeatCount', 'Masters', 'Repeats']

    return sceneTable


def filtIntfTable(intfTable, **kwargs):
    """
    Filter interferogram table using input parameters
    ---- INPUT ----------------------------------------------
    intfTable - input interferogram table

    Kwargs:
    - Keys should be column names of intfTable.
    - Arguments should be lists containing minimum/maximum values.
    Master - min/max interferogram master date
    Repeat - min/max interferogram Repeat date
    TempBaseline - min/max temporal baseline length (days)
    OrbitBaseline - min/max temporal baseline length (m)
    MeanCorr - min/max mean intereferogram coherence

    ---- OUTPUT ---------------------------------------------
        filtIntfTable - table of interferograms meeting specified input parameters

    ---- EXAMPLE --------------------------------------------
    filtIntfTable = filtIntfTable(intfTable, Master=[dt.datetime(2014,1,1,0,0,0), dt.datetime(2021,1,1,0,0,0)],
                                Repeat=[dt.datetime(2014,1,1,0,0,0), dt.datetime(2021,1,1,0,0,0)],
                                TempBaseline=[0, 10**10],
                                OrbitBaseline=[-1000, 1000],
                                MeanCorr=[0, 1],
                                Order=[1, 100])
    """
    filtIntfTable = intfTable

    print('Filtering with following constraints:')

    for arg in kwargs:
        # Print message
        print('{}: {} to {}'.format(arg, kwargs[arg][0], kwargs[arg][1]))
        # Perform filtering
        filtIntfTable = filtIntfTable[(intfTable[arg] >= kwargs[arg][0]) &
                                      (intfTable[arg] <= kwargs[arg][1])]

    # Reset index to 0,1,2,..., n-1
    filtIntfTable = filtIntfTable.reset_index(drop=True)

    # Print
    print()
    print('{} interferograms selected'.format(len(filtIntfTable)))
    print()

    return filtIntfTable


def addOrder(intfTable, baselineTable):
    """
    Calulate and append "nearest neighbor order" to each interferogram in intfTable.
    For example, if 20191202 is the 1st available aquisition and 20200312 is the 5th,
    then 20191202_20200312 is a 4th-order interferogram.
    """
    order = []

    for i in range(len(intfTable)):
        # Get indicies of scenes in intf
        mi = baselineTable[baselineTable['Date'] == intfTable['Master'][i]].index
        ri = baselineTable[baselineTable['Date'] == intfTable['Repeat'][i]].index
        order.append((ri - mi)[0])

    # Append column to imnput intfTable
    newIntfTable = intfTable
    newIntfTable['Order'] = order

    return newIntfTable


def makeCandisTables(intfTable, baselineTable, **kwargs):

    intf_list = pd.DataFrame([intfTable['Master'][i].strftime('%Y%m%d') + '_' + intfTable['Repeat'][i].strftime('%Y%m%d') for i in range(len(intfTable))])
    baseline_info = pd.DataFrame()
    baseline_info['Date'] = [date.strftime('%Y%m%d') for date in baselineTable['Date']]
    baseline_info['OrbitBaseline'] = baselineTable['OrbitBaseline']
    dates_to_use = baseline_info['Date']

    if 'fileNames' in kwargs:
        print()
        for i, table in enumerate([intf_list, baseline_info, dates_to_use]):
            print('Writing intf_list to ' + kwargs['fileNames'][i])
            table.to_csv(kwargs['fileNames'][i], sep=' ', index=False, header=False)

    return intf_list, baseline_info, dates_to_use
