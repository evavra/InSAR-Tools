import sys
import glob
import numpy as np
import netCDF4 as nc
import datetime as dt

"""
Read and write InSAR data files (mostly NetCDF4 for images and ASCII for time series)
"""


def getFileNames(fileDir, fileType):
    """
    Get paths to 'fileType' files in 'fileDir' directory
    """

    # Seatches directory fileDir for files of format fileType
    print('Searching ' + fileDir + fileType)
    filePaths = glob.glob(fileDir + fileType)
    filePaths.sort()

    if len(filePaths) == 0:
        print("Error! No files matching search pattern.")
        sys.exit(1)

    print("Reading data from " + str(len(filePaths)) + " files.")

    return filePaths


def readGrd(filePath):
    """
    Read in NetCDF grid-formattted InSAR data

    INPUT:
    filepath - full path to file

    OUTPUT:
    xdata - data contained in 'x' variable
    ydata - data contained in 'y' variable
    zdata - data contained in 'z' variable
    """

    grid = nc.Dataset(filePath, 'r')

    # Anticipate radar coordinates
    if '_ra.' in filePath:
        xdata = np.array(grid.variables['x'])
        ydata = np.array(grid.variables['y'])
        zdata = np.flip(np.array(grid.variables['z']), 1)

    # Anticipate geographic coordinates
    elif '_ll.' in filePath:
        try:
            xdata = np.array(grid.variables['lon'])
            ydata = np.array(grid.variables['lat'])
            zdata = np.array(grid.variables['z'])

            # Convert longitude data to Prime Meridian = 0 deg
            xdata = xdata - 360
            zdata = np.flip(zdata, 0)

        except KeyError:
            # Assume variable names have regular format
            xdata = np.array(grid.variables['x'])
            ydata = np.array(grid.variables['y'])
            zdata = np.array(grid.variables['z'])

            xdata = xdata - 360
            zdata = np.flip(zdata, 0)

    else:
        # Assume radar format (GMTSAR and CANDIS default)
        xdata = np.array(grid.variables['x'])
        ydata = np.array(grid.variables['y'])
        zdata = np.flip(np.array(grid.variables['z']), 1)

        print('Please remember to specify radar (ra) or lat/lon (ll) in the future!')

    grid.close()

    return xdata, ydata, zdata


def readStack(filePaths):
    """
    Read in stack of InSAR data
    Uses readGrd

    INPUT:
    filePaths - list of paths to files to read

    OUTPUT:
    x - data contained in 'x' variable
    y - data contained in 'y' variable
    stack - stack of 'z' data
    dates - dates of SAR acquisitions
    """

    stack = []
    dates = []

    # Read in stack of
    for file in filePaths:
        try:
            # File name with ra/ll suffix (LOS_20161121_INT3_ll.grd)
            dates.append(dt.datetime.strptime(file[-20:-12], '%Y%m%d'))

        except ValueError:
            # Default CANDIS format (# LOS_20161121_INT3.grd)
            dates.append(dt.datetime.strptime(file[-17:-9], '%Y%m%d'))

        print('Reading ' + dates[-1].strftime('%Y-%m-%d') + ' ...')
        x, y, z = readGrd(file)

        stack.append(z)

    return x, y, stack, dates
