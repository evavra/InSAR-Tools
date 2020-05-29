import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import data


# def getIndex(x, y, grid, coordinates):
#     """
#     Get grid indicies of given range/azimuth or longitude/latitude point.
#     (Remember that Numpy convention is [row, column], not [x, y])

#     INPUT:
#     x - row coordinate (azimuth or latitude)
#     y - column coordinate (range or longitude)
#     grid - path to NetCDF grid to extract coordinates from
#     coordinates - 'ra' for radar (rng/azi) or 'll' for geographic (lon/lat)

#     OUTPUT:
#     ij - indicies of array cell equal or closest to input point
#     """

#     # Load data
#     data = nc.Dataset(file, )

#     return ij

def getPoint(grid):
    """
    Retrieve point from data grid via point-and-click

    INPUT:
    grid - Numpy array containing InSAR data
    """
    ij = []

    def select_coordinates(event):
        """
        Helper function for selecting point.
        """

        ij.append(int(event.xdata))
        ij.append(int(event.ydata))
        print('Point indicies: ({}, {})'.format(ij[0], ij[1]))
        plt.close()

        return ij

    # Plot deformation map with selected region/pixels overlain
    fig = plt.figure()
    ax = plt.gca()
    im = ax.imshow(grid)
    cbar = fig.colorbar(im)
    cbar.set_label('LOS change (m)')
    fig.canvas.mpl_connect('button_press_event', select_coordinates)

    plt.show()

    return ij


def getTimeSeries(stack, point, track, coordinates, **kwargs):
    """
    Get InSAR time series at speficied point.
    ox-averaged

    INPUT:
    stack - 3D numpy array of InSAR stack (stack[Aquisition][Azimuth][Range])
    point - incidies of point
    track - direction of satellite heading ('asc' or 'des')



    Optional:
    boxDim - averaging box side length in pixels (Default = 10)


    """

    # Handle kwarg settings
    if 'boxDim' in kwargs:
        boxDim = kwargs['boxDim']
    else:
        boxDim = 10

    # Compute time series for a specified point
    # First, get averaging box indicies
    box = [point[0] - int(boxDim / 2), point[0] + int(boxDim / 2), point[0] - int(boxDim / 2), point[0] + int(boxDim / 2)]

    print()
    print('Box azimuth: ({}, {})'.format(box[0], box[1]))
    print('Box range: ({}, {})'.format(box[2], box[3]))
    print()

    # Now we've got our averaging box. We will use the mean value of this box to formulate each point in the timeseries
    rangeChange = [np.nanmean(grid[box[0]:box[1], box[2]:box[3]]) for grid in stack]

    return rangeChange


if __name__ == '__main__':
    # Point time series test
    # --------------------------------------------------------------------------------------
    fileDir = '/Users/ellisvavra/Desktop/LongValley/LV-InSAR/Results/Run01/'
    fileType = 'LOS_20*_INT3.grd'

    track = 'des'
    coordinates = 'ra'
    colors = 'Spectral_r'
    boxDim = 10

    save = 'no'
    outDir = fileDir
    outputName_ts = "timeseries_test.eps"

    # --------------------------------------------------------------------------------------
    # Get list of filenames
    fileNames = data.getFileNames(fileDir, fileType)

    # Read data
    [xdata, ydata, stack, dates] = data.readStack(fileNames)

    # Get time series point
    point = getPoint(stack[-1])

    # Get time series data
    rangeChange = getTimeSeries(stack, point, track, coordinates)

    plt.scatter(dates, rangeChange)
    plt.show()
