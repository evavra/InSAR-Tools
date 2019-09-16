import matplotlib.pyplot as plt
import numpy as np
import glob as glob
import sys
import datetime as dt
import subprocess
import netcdf_read_write
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import utilities_GPS
from readGRD import readInSAR

# Original version by Kathryn Materna
# Modified by Ellis Vavra


# ------------------------- DRIVER ------------------------- 

def driver():


    # # ------------------------- GENERATE INSAR PANELS AND POINT TIME SERIES -------------------------
    # # -- INPUT --------------------------------------------------------------------------------------
    # # Files
    # fileDir = "/Users/ellisvavra/Thesis/insar/des/f2/intf_all/Attempt15-Cmin-0.20/SBAS_SMOOTH_0.0000e+00/"  # Lorax
    # fileType = "LOS_*_INT3.grd"

    # output_name = "panels_Attempt15.eps"
    # output_name_ts = "ts_P650_Attempt15.eps"
    # output_name_swath = ""
    # output_name_compare = "InSAR-RDOM-Attempt15.eps"
    # save = 'no'

    # # Figure settings
    # num_plots_x = 6
    # num_plots_y = 14
    # colors = 'jet'

    # # Region settings
    # box_width = 50         # Time-series region width in pixels
    # box_height = 50        # Time-series region height in pixels

    # # -- EXECUTE --------------------------------------------------------------------------------------
    # # Get list of filenames
    # file_names = getFileNames(fileDir, fileType)

    # # Extract data
    # [xdata, ydata, data_all, titles] = inputs(file_names)

    # # Plot InSAR time-series panels
    # # insar_panels(xdata, ydata, data_all, num_plots_x, num_plots_y, titles, colors, fileDir, output_name, save)

    # # Get coordinates for point analysis
    # region = getRegion(data_all, titles, colors, box_width, box_height)
    # # region = [505, 510, 675, 685] # RDOM
    # # region = [285, 305, 1265, 1285] 
    # # region = [75, 85, 1230, 1240] # P649

    # # Make timeseries from selected point
    # # point_ts(data_all, titles, region, colors, fileDir, output_name_ts, save


    # # -------------------------------- OVERLAY INSAR TIME SERIES ---------------------------------
    # # -- INPUT -----------------------------------------------------------------------------------
    # output_dir = "/Users/ellisvavra/Thesis/insar/des/f2/intf_all/Figures/"  # Lorax

    # fileDir11 = "/Users/ellisvavra/Thesis/insar/des/f2/intf_all/Attempt11-Deramp-3-Ref-1/SBAS_SMOOTH_0.0000e+00/"  
    # fileDir12 = "/Users/ellisvavra/Thesis/insar/des/f2/intf_all/Attempt12-Deramp-2-Ref-1/SBAS_SMOOTH_0.0000e+00/"  
    # fileDir14 = "/Users/ellisvavra/Thesis/insar/des/f2/intf_all/Attempt14-Updated-ts-Cmin-0.19/SBAS_SMOOTH_0.0000e+00/"  
    # fileDir15 = "/Users/ellisvavra/Thesis/insar/des/f2/intf_all/Attempt15-Cmin-0.20/SBAS_SMOOTH_0.0000e+00/"  
    # file_names11 = getFileNames(fileDir11, fileType)
    # file_names12 = getFileNames(fileDir12, fileType)
    # file_names14 = getFileNames(fileDir14, fileType)
    # file_names15 = getFileNames(fileDir15, fileType)

    # output_name_ts2 = "ts-Attempt11-12-14-15.eps"

    # # -- EXECUTE ---------------------------------------------------------------------------------
    # [xdata, ydata, data_all11, titles11] = inputs(file_names11)
    # [xdata, ydata, data_all12, titles12] = inputs(file_names12)
    # [xdata, ydata, data_all14, titles14] = inputs(file_names14)
    # [xdata, ydata, data_all15, titles15] = inputs(file_names15)
    # master_data = [data_all11, data_all12, data_all14, data_all15]
    # master_titles = [titles11, titles12, titles14, titles15]
    # region = [400, 450, 700, 750]

    # labels = ['Attempt 12', 'Attempt 13', 'Attempt 14', 'Attempt 15']

    # overlay_ts(master_data, master_titles, region, colors, fileDir12, output_name_ts2, labels, save)



    # # ------------------------- COMPARE INSAR AND GPS TIME SERIES -------------------------
    # gps_filename = 'RDOM.IGS08.tenv3.txt'
    # data_format = 'env'
    # component = 'N'
    # region = [400, 450, 700, 750]

    # # Get list of filenames
    # file_names = getFileNames(fileDir, fileType)

    # # Extract data
    # [xdata, ydata, data_all, titles] = inputs(file_names)

    # gps_insar_ts(data_all, titles, region, colors, fileDir, output_name_compare, save, gps_filename, data_format, component)
    


    # # ------------------------- CALCULATE INSAR BASELINE -------------------------
    colors = 'jet'
    fileDir = "/Users/ellisvavra/Thesis/insar/des/f2/intf_all/Geocoding/Attempt15/"  # Lorax
    fileType = "LOS_*_INT3_ll.grd"

    # Get list of filenames
    filePaths = getFileNames(fileDir, fileType)

    # Extract data
    lon, lat, zCube, dates = readStack(filePaths)

    # Compute time series for RDOM
    RDOM = timeSeries(lon, lat, zCube, [-118.898, 37.677], 10)

    # Compute time series for P649
    P649= timeSeries(lon, lat, zCube, [-118.736, 37.903], 10)

    # Compute differential range change
    RDOM_P649 = diffTimeSeries(RDOM, P649)

    # Plot
    ax1 = plt.subplot(211)
    plotTimeSeries(dates, RDOM)
    plotTimeSeries(dates, P649)
    plt.grid()

    ax2 = plt.subplot(212)
    plotTimeSeries(dates, RDOM_P649)

    plt.show()

# ------------------------- READING ------------------------- 

def getFileNames(fileDir, fileType):
     # Seatches directory fileDir for files of format fileType
    print('Searching ' + fileDir + fileType)
    filePaths = glob.glob(fileDir + fileType)

    if len(filePaths) == 0:
        print("Error! No files matching search pattern.")
        sys.exit(1)

    print("Reading data from " + str(len(filePaths)) + " files.")

    return filePaths


def readStack(filePaths):
    
    zCube = []
    dates = []

    # Read in stack of 
    for file in filePaths:

         # LOS_20161121_INT3_ll.grd
        dates.append(dt.datetime.strptime(file[-20:-12], '%Y%m%d'))
        print('Reading ' + dates[-1].strftime('%Y-%m-%d') + ' ...')
        x, y, z = readInSAR(file)
        zCube.append(z)

    return x, y, zCube, dates

# ------------------------- SELECT ------------------------- 

def getRegion(data_all, titles, colors, box_width, box_height):

    point = []
    region = []

    def select_coordinates(event):
        # # For mutiple points
        # points.append([event.xdata, event.ydata])
        # print(points[-1])

        point.append(int(event.xdata))
        point.append(int(event.ydata))

        region.append(int(point[0] - box_width/2))
        region.append(int(point[0] + box_width/2))
        region.append(int(point[1] - box_height/2))
        region.append(int(point[1] + box_height/2))

        print()
        print('Centroid: ' + str(point[-1]))
        print('Min. x: ' + str(region[0]))
        print('Max. x: ' + str(region[1]))
        print('Min. y: ' + str(region[2]))
        print('Max. y: ' + str(region[3]))
        print()

        plt.close()

        return region

    fig = plt.figure(figsize=(15, 15))

    # Plot deformation map with selected region/pixels overlain
    ax1 = plt.subplot(111)
    im = ax1.imshow(data_all[-1], cmap=colors, vmin=-0.05, vmax=0.05, aspect=0.75)
    ax1.set_title(titles[-1], fontsize=12, color='black')
    ax1.invert_yaxis()
    ax1.invert_xaxis()
    cbar = fig.colorbar(im)
    cbar.set_label('LOS change (m)')

    fig.canvas.mpl_connect('button_press_event', select_coordinates)

    plt.show()

    return region


def getBaseline(data_all, titles, colors):

    points = []

    def select_coordinates(event):
        # For mutiple points
        points.append([event.xdata, event.ydata])
        print(points[-1])

        if len(points) == 2: 
            plt.close()

        return points

    fig = plt.figure(figsize=(15, 15))

    # Plot deformation map with selected region/pixels overlain
    ax1 = plt.subplot(111)
    im = ax1.imshow(data_all[-1], cmap=colors, vmin=-0.05, vmax=0.05, aspect=0.75)
    ax1.set_title(titles[-1], fontsize=12, color='black')
    ax1.invert_yaxis()
    ax1.invert_xaxis()
    cbar = fig.colorbar(im)
    cbar.set_label('LOS change (m)')

    fig.canvas.mpl_connect('button_press_event', select_coordinates)

    plt.show()

    return region


# ------------------------- ANALYSIS ------------------------- 

def timeSeries(xData, yData, zCube, point, boxDim):

    # Compute time series for a specified point
    print()
    print('Point: ' + str(point))
    print()
    print('x-array dimensions: ' + str(xData.shape))
    print('y-array dimensions: ' + str(yData.shape))
    print('z-array dimensions: ' + str(zCube[0].shape))
    print()

    # Since the z data are organized by numerical index, not geographic coordinate, we count the number of pixels
    # which are less than the target point x/y coordinates to find its location in the grid.
    # We must subtract zero from the sums due to Python's zero-indexing
    ptID = [sum(xData < point[0]) - 1, sum(yData > point[1]) - 1]

    boxID = [ptID[0] - boxDim,
             ptID[0] + boxDim,
             ptID[1] - boxDim,
             ptID[1] + boxDim ]

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
                    nanCount+=1
                else:
                    boxSum.append(scene[y][x])
                    numPix += 1

        # dates.append(dt.datetime.strptime(titles[i], "%Y%m%d"))
        rangeChange.append(np.mean(boxSum))

    return rangeChange

def diffTimeSeries(ts1, ts2):
    # Calculate differential change between two point time series
    diff = []

    for i in range(len(ts1)):
        diff.append(ts1[i] - ts2[i])

    return diff

# ------------------------- PLOTTING ------------------------- 

def insar_panels(xdata, ydata, data_all, num_plots_x, num_plots_y, titles, colors, fileDir, output_name, save):

    rows = num_plots_y
    cols = num_plots_x
    count = 0

    fig = plt.figure(figsize=(rows * 12, cols * 10))

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(rows, cols),
                    axes_pad=0.4,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.2
                    )

    print('Number of files: ' + str(len(data_all)))

    for ax in grid:
        if count == len(data_all):
            break

        ax.set_axis_off()
        # im = ax.imshow(data_all[count], cmap='jet', aspect=8)
        im = ax.imshow(data_all[count], vmin=-0.05, vmax=0.05, cmap=colors, aspect=0.75)
        # ax.plot(400, 700, marker='o', markersize=5, color='black', zorder=10000)
        ax.set_title(titles[count], fontsize=20, color='black')
        ax.invert_yaxis()
        ax.invert_xaxis()

        print(titles[count])
        count += 1
 
    cbar = ax.cax.colorbar(im)
    # cbar = grid.cbar_axes[0].colorbar(im)

    cbar.ax.set_yticks(np.arange(-0.05, 0.05))
    # cbar.ax.set_yticklabels(['low', 'medium', 'high'])

    
    if save == 'yes':
        print('Saving InSAR panels to ' + fileDir + output_name)
        plt.savefig(fileDir + output_name)
        plt.close()
        subprocess.call("open " + fileDir + output_name, shell=True)

    else:
        plt.show()


def plotTimeSeries(dates, timeSeries):
    # Plot time-series for selected pixels
    plt.scatter(dates, timeSeries, zorder=100)
    plt.grid()
    # ax2.set_aspect(2.15 * 10**4)

    # ax2.set_xlim(min(dates) - dt.timedelta(days=30), max(dates) + dt.timedelta(days=30))

    plt.xlabel('Date')
    # plt.ylabel('LOS range change (m)')


def overlay_ts(master_data, master_titles, region, colors, fileDir, output_name_ts2, labels, save):
    # master_data = list of data_all# lists

    dates = []
    range_change = []

    fig = plt.figure(figsize=(25, 10))

    # Plot deformation map with selected region/pixels overlain
    ax1 = plt.subplot(121)
    # ax1 = plt.subplot(111)
    # grid[0].set_axis_off()
    im = ax1.imshow(master_data[0][-1], cmap=colors, vmin=-0.05, vmax=0.05, aspect=0.75)

    ax1.plot([region[0], region[0], region[1], region[1], region[0]], [region[2], region[3], region[3], region[2], region[2]], color='blue', zorder=10000)
    ax1.set_title(master_titles[0][-1], fontsize=12, color='black')
    ax1.invert_yaxis()
    ax1.invert_xaxis()
    cbar = fig.colorbar(im)
    # cbar.set_label('LOS change (m)')

    # Plot time-series for selected pixels
    ax2 = plt.subplot(122)
    ax2.grid()

    # Get averaged timeseries for given region 
    count = 0
    for i in range(len(master_data)):
        for j in range(len(master_data[i])):
            nanCount = 0
            total = []
            numPix = 0
            for x in range(region[0], region[1]):
                for y in range(region[2], region[3]):
                    if math.isnan(master_data[i][j][y][x]):
                        nanCount+=1
                    else:
                        total.append(master_data[i][j][y][x])
                        numPix += 1

            # print('nanCount = ' + str(nanCount))
            # print('Count = ' + str(len(sum)))
            # print('Mean value = ' + str(np.mean(sum)))

            dates.append(dt.datetime.strptime(master_titles[i][j], "%Y%m%d"))
            range_change.append(np.mean(total))

        ax2.plot(dates, range_change, linestyle='--', marker='.', zorder=10)
        count += 1
        
        # Reset if not last iteration
        if count < len(master_data):
            dates = []
            range_change = []

    ax2.legend(labels, loc='lower right')
    ax2.set_aspect(2.15 * 10**4)
    ax2.set_xlim(min(dates) - dt.timedelta(days=30), max(dates) + dt.timedelta(days=30))
    ax2.set_ylim(-0.01, 0.05)

    plt.xlabel('Date')
    plt.ylabel('LOS range change (m)', rotation=0, labelpad=58)

    if save == 'yes':
        print('Saving time series to ' + fileDir + output_name_ts2)
        plt.savefig(fileDir + output_name_ts2)
        plt.close()
        subprocess.call("open " + fileDir + output_name_ts2, shell=True)

    else:
        plt.show()


def gps_insar_ts(data_all, titles, region, colors, fileDir, output_name_compare, save, gps_filename, data_format, component):

    dates_insar = []
    range_change = []

    fig = plt.figure(figsize=(25, 10))

    # Plot deformation map with selected region/pixels overlain
    ax1 = plt.subplot(121)
    im = ax1.imshow(data_all[-1], cmap=colors, vmin=-0.05, vmax=0.05, aspect=0.75)

    ax1.plot([region[0], region[0], region[1], region[1], region[0]], [region[2], region[3], region[3], region[2], region[2]], color='blue', zorder=10000)
    ax1.set_title(titles[-1], fontsize=12, color='black')
    ax1.invert_yaxis()
    ax1.invert_xaxis()
    cbar = fig.colorbar(im)
    # cbar.set_label('LOS change (m)')

    # Get averaged timeseries for given region 
    for i in range(len(data_all)):
        nanCount=0
        sum = []
        numPix = 0
        for x in range(region[0], region[1]):
            for y in range(region[2], region[3]):
                if math.isnan(data_all[i][y][x]):
                    nanCount+=1
                else:
                    sum.append(data_all[i][y][x])
                    numPix += 1

        # print('nanCount = ' + str(nanCount))
        # print('Count = ' + str(len(sum)))
        # print('Mean value = ' + str(np.mean(sum)))

        dates_insar.append(dt.datetime.strptime(titles[i], "%Y%m%d"))
        range_change.append(np.mean(sum))

    # Plot InSAR time-series for selected pixels
    ax2 = plt.subplot(122)
    ax2.grid()
    ax2.scatter(dates_insar, range_change, zorder=100)
    ax2.set_aspect(2.15 * 10**4)


    # Load GPS data
    gps_data = utilities_GPS.readUNR(gps_filename, data_format)
    
    # Get displacements
    plot_displacements = []

    # First find start date
    search_date = dates_insar[0]
    z_init = 666

    while z_init == 666:
        print('Looking for ' + search_date.strftime('%Y-%m-%d'))
        for i in range(len(gps_data.dates)):
            print(search_date.strftime('%Y%m%d') + ' = ' + gps_data.dates[i].strftime('%Y%m%d') + '?')
            print(search_date.strftime('%Y%m%d') == gps_data.dates[i].strftime('%Y%m%d'))
            if search_date.strftime('%Y%m%d') == gps_data.dates[i].strftime('%Y%m%d'):
                plot_dates = gps_data.dates[i:]
                print('GPS time series start: ' + str(plot_dates[0]))
                plot_data = gps_data.up[i:]
                z_init = gps_data.up[i]
                break

        # Try next day
        search_date += dt.timedelta(days=1)

    search_date -= dt.timedelta(days=1)
    print('Using ' + search_date.strftime('%Y-%m-%d'))
    print("Initial value: " + str(z_init))

    for value in plot_data:
        plot_displacements.append(value - z_init)
        print(plot_displacements[-1])

    plt.grid()
    plt.scatter(plot_dates, plot_displacements, marker='.', zorder=101)
    # ax2.set_xlim(min(dates_insar) - dt.timedelta(days=30), max(dates_insar) + dt.timedelta(days=30))
    ax2.set_xlim(min(dates_insar), max(dates_insar))
    # ax2.set_ylim(-0.01, 0.05)

    plt.xlabel('Date')
    plt.ylabel('LOS range change (m)', rotation=0, labelpad=58)


    if save == 'yes':
        print('Saving time series to ' + fileDir + output_name_compare)
        plt.savefig(fileDir + output_name_compare)
        plt.close()
        subprocess.call("open " + fileDir + output_name_compare, shell=True)

    else:
        plt.show()


# ------------------------- EXECUTE ------------------------- 

if __name__ == "__main__":
    driver()


"""
DEFUNCT CODE

    # ------------------------------- GENERATE DISPLACEMENT SWATH ---------------------------------
    # -- INPUT ------------------------------------------------------------------------------------
    # Files
    fileDir = "/Users/ellisvavra/Thesis/insar/des/f2/intf_all/Geocoding/Attempt15/"  # Lorax
    fileType = "LOS_*_INT3_ll.grdnc3"
    fileType = "LOS_2019*_INT3_ll.grdnc3"

    # Output filenames
    output_name = "panels_Attempt15.eps"
    output_name_swath = "swath_Attempt15"

     # Figure settings
    proj = 'll'
    colors = 'jet'
    save = 'no'


def point_ts(data_all, titles, region, colors, fileDir, output_name_ts, save):

    dates = []
    range_change = []

    fig = plt.figure(figsize=(25, 10))

    # Plot deformation map with selected region/pixels overlain
    ax1 = plt.subplot(121)
    # ax1 = plt.subplot(111)
    # grid[0].set_axis_off()
    im = ax1.imshow(data_all[-3], cmap=colors, vmin=-0.05, vmax=0.05, aspect=0.75)

    ax1.plot([region[0], region[0], region[1], region[1], region[0]], [region[2], region[3], region[3], region[2], region[2]], color='blue', zorder=10000)
    ax1.set_title(titles[-3], fontsize=12, color='black')
    ax1.invert_yaxis()
    ax1.invert_xaxis()
    cbar = fig.colorbar(im)
    # cbar.set_label('LOS change (m)')


    # Get averaged timeseries for given region 
    for i in range(len(data_all)):
        nanCount=0
        sum = []
        numPix = 0
        for x in range(region[0], region[1]):
            for y in range(region[2], region[3]):
                if math.isnan(data_all[i][y][x]):
                    nanCount+=1
                else:
                    sum.append(data_all[i][y][x])
                    numPix += 1

        print('nanCount = ' + str(nanCount))
        print('Count = ' + str(len(sum)))
        print('Mean value = ' + str(np.mean(sum)))

        dates.append(dt.datetime.strptime(titles[i], "%Y%m%d"))
        range_change.append(np.mean(sum))


    # Plot time-series for selected pixels
    ax2 = plt.subplot(122)
    ax2.grid()
    ax2.scatter(dates, range_change, zorder=100)
    ax2.set_aspect(2.15 * 10**4)

    ax2.set_xlim(min(dates) - dt.timedelta(days=30), max(dates) + dt.timedelta(days=30))
    # ax2.set_ylim(-0.01, 0.05)


    plt.xlabel('Date')
    plt.ylabel('LOS range change (m)', rotation=0, labelpad=58)

    if save == 'yes':
        print('Saving time series to ' + fileDir + output_name_ts)
        plt.savefig(fileDir + output_name_ts)
        plt.close()
        subprocess.call("open " + fileDir + output_name_ts, shell=True)

    else:
        plt.show()

def inputs(file_names):

    # Make exception if .grd files are in lat/lon coordinates and not yet converted to old NetCDF format
    if '_ll.' in file_names[0]:
        print('LOS files have been geocoded.')
        print()

        xdata = []
        ydata = []
        data_all = []
        titles = []
        file_names = sorted(file_names)  # To force into date-ascending order.

        if 'grdnc3' in file_names[0]:
            print('Files have already been converted to NetCDF3 format')

            for ifile in file_names:   # Read the data
                print('Reading ' + ifile + ' ...')
                [x, y, z] = netcdf_read_write.read_grd_lonlatz(ifile)

                xdata.append(x)
                ydata.append(y)
                data_all.append(z)

                # LOS_20161121_INT3_ll.grdnc3
                titles.append(ifile[-23:-15])
        else:
            print('Need to convert files to NetCDF3 format')
            for ifile in file_names:   # Read the data
                print('Reading ' + ifile + ' ...')
                [x, y, z] = netcdf_read_write.read_netcdf4_variables(ifile, 'lon', 'lat', 'z')

                xdata.append(x)
                ydata.append(y)
                data_all.append(z)

                # LOS_20161121_INT3_ll.grd
                titles.append(ifile[-23:-15])

    else:
        try:
            [xdata, ydata] = netcdf_read_write.read_grd_xy(file_names[0])  # can read either netcdf3 or netcdf4.
        except TypeError:
            [xdata, ydata] = netcdf_read_write.read_netcdf4_xy(file_names[0])

        data_all = []
        file_names = sorted(file_names)  # To force into date-ascending order.
        titles = []

        for ifile in file_names:  # Read the data
            print('Reading ' + ifile + ' ...')
            try:
                data = netcdf_read_write.read_grd(ifile)
            except TypeError:
                data = netcdf_read_write.read_netcdf4(ifile)
            data_all.append(data)
            # LOS_20161121_INT3.grd
            titles.append(ifile[-17:-9])

    return [xdata, ydata, data_all, titles]

def getSwath(data_all, titles, colors, proj):

    points = []
    region = []

    def select_coordinates(event):

        points.append([int(event.xdata), int(event.ydata)]) 
        print(points[-1])
        
        if len(points) == 2:
            region.append(min([points[0][0], points[1][0]]))
            region.append(max([points[0][0], points[1][0]]))
            region.append(min([points[0][1], points[1][1]]))
            region.append(max([points[0][1], points[1][1]]))
            plt.close()

        return region

    fig = plt.figure(figsize=(15, 15))

    # Plot deformation map with selected region/pixels overlain
    ax1 = plt.subplot(111)
    im = ax1.imshow(data_all[-2], cmap=colors, vmin=-0.05, vmax=0.05, aspect=0.75, zorder=1)
    ax1.set_title(titles[-2], fontsize=12, color='black')

    if proj == 'ra':
        ax1.invert_yaxis()
        ax1.invert_xaxis()

    ax1.set_aspect(1)

    cbar = fig.colorbar(im)
    cbar.set_label('LOS change (m)')

    fig.canvas.mpl_connect('button_press_event', select_coordinates)

    plt.show()

    return region

def swath(data_all, titles, region, colors, fileDir, output_name_swath, save, proj):

    range_change = []

    # fig = plt.figure(figsize=(25, 10))

    # Plot deformation map with selected region/pixels overlain
    ax1 = plt.subplot(121)

    plt.grid()
    im = ax1.imshow(data_all[-2], vmin=-0.05, vmax=0.05, cmap=colors, aspect=0.75)

    ax1.plot([region[0], region[0], region[1], region[1], region[0]], [region[2], region[3], region[3], region[2], region[2]], color='blue', zorder=10000)
    ax1.set_title(titles[-2], fontsize=12, color='black')
    
    if proj == 'ra':
        ax1.invert_yaxis()
        ax1.invert_xaxis()
    
    ax1.set_aspect(1)
    cbar = fig.colorbar(im)
    # cbar.set_label('LOS change (m)')

    print()
    print('Number of dates = ' + str(len(data_all)))
    print('Range = ' + str(len(data_all[-1])))
    print('Azimuth = ' + str(len(data_all[-1][0])))
    print('Swath = ' + str(region))
    print()
    swath_x = []
    swath_y = []

    displacements = data_all[-2]

    # Calculate mean swath displacements for East-West trending swath
    if len(range(region[0], region[1])) > len(range(region[2], region[3])):
        for i in range(region[0], region[1]):
            total = 0
            numPix = 0
            
            for row in displacements[region[2]:region[3]]:

                if math.isnan(row[i]) != True:
                    total += row[i]
                    numPix += 1

            if numPix != 0:     
                swath_x.append(i)
                swath_y.append(total/numPix)

        ax2 = plt.subplot(122)
        ax2.grid()
        ax2.scatter(swath_x, swath_y, marker='.', zorder=100)
        plt.xlabel('Longitude')
        plt.ylabel('LOS range change (m)', rotation=0, labelpad=58)

    # Or, calculate mean swath displacements for North-South trending swath
    else:
        for i in range(region[2], region[3]):
            total = 0
            numPix = 0
            for pixel in displacements[i][region[0]:region[1]]:

                if math.isnan(pixel) != True:
                    total += pixel
                    numPix += 1

            if numPix != 0:     
                swath_x.append(i)
                swath_y.append(total/numPix)

        ax2 = plt.subplot(122)
        ax2.grid()
        ax2.invert_xaxis()
        ax2.scatter(swath_x, swath_y, marker='.', zorder=100)
        plt.xlabel('Latitude')
        plt.ylabel('LOS range change (m)', rotation=0, labelpad=58)

    
    

    if save == 'yes':
        print('Saving swath plot to ' + fileDir + output_name_swath)
        plt.savefig(fileDir + output_name_swath)
        plt.close()
        subprocess.call("open " + fileDir + output_name_swath, shell=True)

    else:
        plt.show()

"""