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

# Original version by Kathryn Materna
# Modified by Ellis Vavra


# ------------------------- DRIVER ------------------------- 

def driver():

    # """
    # ------------------------- GENERATE INSAR PANELS AND POINT TIME SERIES -------------------------
    # InSAR time series files

    # file_dir = "/Users/ellisvavra/Thesis/insar/des/f2/intf_all/Attempt11-Deramp-3-Ref-1/SBAS_SMOOTH_0.0000e+00/"  # Lorax
    # file_dir = "/Users/ellisvavra/Thesis/insar/des/f2/intf_all/Attempt12-Deramp-2-Ref-1/SBAS_SMOOTH_0.0000e+00/"  # Lorax
    file_dir = "/Users/ellisvavra/Thesis/insar/des/f2/intf_all/Attempt14-Updated-ts-Cmin-0.19/SBAS_SMOOTH_0.0000e+00/"  # Lorax
    file_type = "LOS_*_INT3.grd"

    output_name = "panels_Attempt14.eps"
    output_name_ts = "ts_paoha_Attempt14.eps"
    output_name_swath = ""
    output_name_compare = "InSAR-RDOM-Attempt14.eps"
    save = 'yes'

    # Figure settings
    num_plots_x = 6
    num_plots_y = 14
    colors = 'jet'

    # Region settings
    box_width = 50         # Time-series region width in pixels
    box_height = 50        # Time-series region height in pixels

    """
    # Get list of filenames
    file_names = getFileNames(file_dir, file_type)

    # Extract data
    [xdata, ydata, data_all, titles] = inputs(file_names)

    # Plot InSAR time-series panels
    # insar_panels(xdata, ydata, data_all, num_plots_x, num_plots_y, titles, colors, file_dir, output_name, save)

    # Get coordinates for point analysis
    region = getRegion(data_all, titles, colors, box_width, box_height)
    # region = [400, 450, 700, 750]

    # Make timeseries from selected point
    point_ts(data_all, titles, region, colors, file_dir, output_name_ts, save)

    # Get coordinates for swath analysis
    # region = getSwath(data_all, titles, colors)

    # Make displacement swath
    # swath(data_all, titles, region, colors, file_dir, output_name_swath, save)
    """


    """
    # ------------------------- OVERLAY INSAR TIME SERIES -------------------------
    output_dir = "/Users/ellisvavra/Thesis/insar/des/f2/intf_all/Figures/"  # Lorax

    file_dir11 = "/Users/ellisvavra/Thesis/insar/des/f2/intf_all/Attempt11-Deramp-3-Ref-1/SBAS_SMOOTH_0.0000e+00/"  
    file_dir12 = "/Users/ellisvavra/Thesis/insar/des/f2/intf_all/Attempt12-Deramp-2-Ref-1/SBAS_SMOOTH_0.0000e+00/"  
    file_dir14 = "/Users/ellisvavra/Thesis/insar/des/f2/intf_all/Attempt14-Updated-ts-Cmin-0.19/SBAS_SMOOTH_0.0000e+00/"  
    file_names11 = getFileNames(file_dir11, file_type)
    file_names12 = getFileNames(file_dir12, file_type)
    file_names14 = getFileNames(file_dir14, file_type)

    output_name_ts2 = "ts-Attempt11-Attempt-12-Attempt14.eps"

    [xdata, ydata, data_all11, titles] = inputs(file_names11)
    [xdata, ydata, data_all12, titles] = inputs(file_names12)
    [xdata, ydata, data_all14, titles] = inputs(file_names14)

    master_data = [data_all11, data_all12, data_all14]
    region = [400, 450, 700, 750]

    overlay_ts(master_data, titles, region, colors, file_dir12, output_name_ts2, save)
    """


    """
    # ------------------------- COMPARE INSAR AND GPS TIME SERIES -------------------------
    gps_filename = 'RDOM.IGS08.tenv3.txt'
    data_format = 'env'
    component = 'N'
    region = [400, 450, 700, 750]

    # Get list of filenames
    file_names = getFileNames(file_dir, file_type)

    # Extract data
    [xdata, ydata, data_all, titles] = inputs(file_names)


    gps_insar_ts(data_all, titles, region, colors, file_dir, output_name_compare, save, gps_filename, data_format, component)
    """

    # ------------------------- COMPARE GPS AND INSAR BASELINES -------------------------
    colors = 'jet'
    file_dir = "/Users/ellisvavra/Thesis/insar/des/f2/intf_all/Attempt11-Deramp-3-Ref-1/SBAS_SMOOTH_0.0000e+00/"  # Lorax
    file_type = "LOS_*_INT3.grd"

    # Get list of filenames
    file_names = getFileNames(file_dir, file_type)

    # Extract data
    [xdata, ydata, data_all, titles] = inputs(file_names)
    getBaseline(data_all, titles, colors)


# ------------------------- CONFIGURE ------------------------- 

def getFileNames(file_dir, file_type):

    print('Searching ' + file_dir + file_type)
    file_names = glob.glob(file_dir + file_type)
    if len(file_names) == 0:
        print("Error! No files matching search pattern.")
        sys.exit(1)
    print("Reading " + str(len(file_names)) + " files.")

    return file_names


def inputs(file_names):
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


def getSwath(data_all, titles, colors):

    points = []
    region = []

    def select_coordinates(event):

        points.append([int(event.xdata), int(event.ydata)]) 
       
        # ax1.plot(points[-1][0], points[-1][1], markersize=10000, marker='.', color='blue', zorder=10000)
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
    im = ax1.imshow(data_all[-1], cmap=colors, vmin=-0.05, vmax=0.05, aspect=0.75, zorder=1)
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





# ------------------------- PLOTTING ------------------------- #
def insar_panels(xdata, ydata, data_all, num_plots_x, num_plots_y, titles, colors, file_dir, output_name, save):

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
        print('Saving InSAR panels to ' + file_dir + output_name)
        plt.savefig(file_dir + output_name)
        plt.close()
        subprocess.call("open " + file_dir + output_name, shell=True)

    else:
        plt.show()


def point_ts(data_all, titles, region, colors, file_dir, output_name_ts, save):

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
        print('Saving time series to ' + file_dir + output_name_ts)
        plt.savefig(file_dir + output_name_ts)
        plt.close()
        subprocess.call("open " + file_dir + output_name_ts, shell=True)

    else:
        plt.show()


def swath(data_all, titles, region, colors, file_dir, output_name_swath, save):

    range_change = []

    fig = plt.figure(figsize=(25, 10))

    # Plot deformation map with selected region/pixels overlain
    ax1 = plt.subplot(121)
    # ax1 = plt.subplot(111)
    # grid[0].set_axis_off()
    plt.grid()
    im = ax1.imshow(data_all[-1], vmin=-0.05, vmax=0.05, cmap=colors, aspect=0.75)

    ax1.plot([region[0], region[0], region[1], region[1], region[0]], [region[2], region[3], region[3], region[2], region[2]], color='blue', zorder=10000)
    ax1.set_title(titles[-1], fontsize=12, color='black')
    ax1.invert_yaxis()
    ax1.invert_xaxis()
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

    displacements = data_all[-1]

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
    ax2.scatter(swath_x, swath_y, marker='.', zorder=100)
    ax2.invert_xaxis()

    # ax2.set_aspect(2.15 * 10**4)
    # ax2.set_xlim(min(dates) - dt.timedelta(days=30), max(dates) + dt.timedelta(days=30))


    plt.xlabel('Date')
    plt.ylabel('LOS range change (m)', rotation=0, labelpad=58)
    

    if save == 'yes':
        print('Saving swath plot to ' + file_dir + output_name_swath)
        plt.savefig(file_dir + output_name_swath)
        plt.close()
        subprocess.call("open " + file_dir + output_name_swath, shell=True)

    else:
        plt.show()


def overlay_ts(master_data, titles, region, colors, file_dir, output_name_ts2, save):
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
    ax1.set_title(titles[-1], fontsize=12, color='black')
    ax1.invert_yaxis()
    ax1.invert_xaxis()
    cbar = fig.colorbar(im)
    # cbar.set_label('LOS change (m)')

    # Plot time-series for selected pixels
    ax2 = plt.subplot(122)
    ax2.grid()

    # Get averaged timeseries for given region 
    count = 0
    for data_all in master_data:
        for i in range(len(data_all)):
            nanCount = 0
            total = []
            numPix = 0
            for x in range(region[0], region[1]):
                for y in range(region[2], region[3]):
                    if math.isnan(data_all[i][y][x]):
                        nanCount+=1
                    else:
                        total.append(data_all[i][y][x])
                        numPix += 1

            # print('nanCount = ' + str(nanCount))
            # print('Count = ' + str(len(sum)))
            # print('Mean value = ' + str(np.mean(sum)))

            dates.append(dt.datetime.strptime(titles[i], "%Y%m%d"))
            range_change.append(np.mean(total))

        ax2.scatter(dates, range_change, marker='.', zorder=10)
        count += 1
        
        # Reset if not last iteration
        if count < len(master_data):
            dates = []
            range_change = []

    # ax2.legend(loc='lower right')
    ax2.set_aspect(2.15 * 10**4)
    ax2.set_xlim(min(dates) - dt.timedelta(days=30), max(dates) + dt.timedelta(days=30))
    ax2.set_ylim(-0.01, 0.05)

    plt.xlabel('Date')
    plt.ylabel('LOS range change (m)', rotation=0, labelpad=58)

    if save == 'yes':
        print('Saving time series to ' + file_dir + output_name_ts2)
        plt.savefig(file_dir + output_name_ts2)
        plt.close()
        subprocess.call("open " + file_dir + output_name_ts2, shell=True)

    else:
        plt.show()


def gps_insar_ts(data_all, titles, region, colors, file_dir, output_name_compare, save, gps_filename, data_format, component):

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
        print('Saving time series to ' + file_dir + output_name_compare)
        plt.savefig(file_dir + output_name_compare)
        plt.close()
        subprocess.call("open " + file_dir + output_name_compare, shell=True)

    else:
        plt.show()


if __name__ == "__main__":
    driver()
