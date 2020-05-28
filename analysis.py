import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt


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
        print('Point indicies: {}'.format(ij[0], ij[1]))
        plt.close()

        return region

    fig = plt.figure()

    # Plot deformation map with selected region/pixels overlain
    ax = plt.gca()
    im = ax.imshow(grid)
    cbar = fig.colorbar(im)
    cbar.set_label('LOS change (m)')

    fig.canvas.mpl_connect('button_press_event', select_coordinates)

    plt.show()

    return region


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
    print()
    print('Point indices: ' + str(point))
    print('Z-array dimensions: ' + str(stack[0].shape))
    print()

    # Since the z data are organized by numerical index, not geographic coordinate, we count the number of pixels
    # which are less than the target point x/y coordinates to find its location in the grid.
    # We must subtract zero from the sums due to Python's zero-indexing

    ptID = [sum(xData < point[0]) - 1, sum(yData > point[1]) - 1]

    boxID = [ptID[0] - boxDim,
             ptID[0] + boxDim,
             ptID[1] - boxDim,
             ptID[1] + boxDim]

    # If initial box extends outside of the grid, shift over by one pixel until it fits
    if boxID[0] < min(xData) or boxID[1] > max(xData) or boxID[3] < min(yData) or boxID[4] > max(yData):

        print('Box extends outside of grid domain')

    print('Point indicies: ' + str(ptID))
    print('Box indicies: ' + str(boxID))
    print()

    # plt.imshow(zCube[0])
    # plt.scatter(ptID[0], ptID[1], marker='o', s=20, c='k')
    # plt.plot([boxID[0], boxID[1], boxID[1], boxID[0], boxID[0]], np.array([boxID[2], boxID[2], boxID[3], boxID[3], boxID[2]]), c='k')
    # plt.axis([600,1000,600,1000])
    # plt.show()

    # Now we've got our averaging box. We will use the mean value of this box to formulate each point in the timeseries
    dates = []
    rangeChange = []

    for scene in zCube:

        nanCount = 0
        boxSum = []
        numPix = 0

        for x in range(boxID[0], boxID[1]):
            for y in range(boxID[2], boxID[3]):
                if math.isnan(scene[y][x]):
                    nanCount += 1
                else:
                    boxSum.append(scene[y][x])
                    numPix += 1

        # dates.append(dt.datetime.strptime(titles[i], "%Y%m%d"))
        rangeChange.append(np.mean(boxSum))

    return rangeChange


if __name__ == '__main__':
    #  GENERATE POINT TIME SERIES
    # -- INPUT --------------------------------------------------------------------------------------
    # Files
    fileDir = '/Users/ellisvavra/Desktop/LongValley/LV-InSAR/Results/Run01/'
    fileType = 'LOS_2014*_INT3.grd'

    track = 'des'
    coordinates = 'ra'
    colors = 'Spectral_r'
    boxDim = 10

    save = 'no'
    outDir = fileDir
    outputName_ts = "timeseries_test.eps"

    # -- EXECUTE --------------------------------------------------------------------------------------
    # Get list of filenames
    fileNames = getFileNames(fileDir, fileType)

    # Read data
    [xdata, ydata, stack, dates] = readStack(fileNames)

    # Get point
    point = getPoint(stack[-1])

    # Get time series
    # getTimeSeries(stack, track, coordinates, boxDim)
