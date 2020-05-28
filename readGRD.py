import numpy as np
import netCDF4


def readInSAR(filePath):
    # Input:
    # filePath: full path to NetCDF file
    nc = netCDF4.Dataset(filePath)

    if '_ra.' in filePath:
        xdata = np.array(nc.variables['x'])
        ydata = np.array(nc.variables['y'])
        zdata = np.flip(np.array(nc.variables['z']), 1)

    elif '_ll.' in filePath:
        try:
            xdata = np.array(nc.variables['lon'])
            ydata = np.array(nc.variables['lat'])
            zdata = np.array(nc.variables['z'])

            # Convert longitude data to Prime Meridian = 0 deg
            xdata = xdata - 360
            zdata = np.flip(zdata, 0)

        except KeyError:
            # Assume variable names have regular format
            xdata = np.array(nc.variables['x'])
            ydata = np.array(nc.variables['y'])
            zdata = np.array(nc.variables['z'])

            xdata = xdata - 360
            zdata = np.flip(zdata, 0)

    else:
        # Assume radar format (CANDIS default)
        xdata = np.array(nc.variables['x'])
        ydata = np.array(nc.variables['y'])
        zdata = np.flip(np.array(nc.variables['z']), 1)

        print('Please remember to specify radar (ra) or lat/lon (ll) in the future!')

    return xdata, ydata, zdata
