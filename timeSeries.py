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

# Original version by Kathryn Materna
# Modified by Ellis Vavra


# TOP LEVEL DRIVER
def top_level_driver():
    [file_names, outdir, num_plots_x, num_plots_y] = configure()

    [xdata, ydata, data_all, titles] = inputs(file_names)

    # insar_panels(xdata, ydata, data_all, outdir, num_plots_x, num_plots_y, titles)

    point_ts(xdata, ydata, data_all, outdir, num_plots_x, num_plots_y, titles)



# ------------- CONFIGURE ------------ #
def configure():
    # file_dir = "/Users/ellisvavra/Desktop/Thesis/S1_Processing/NSBAS/INT3/"  # Laptop
    file_dir = "/Users/ellisvavra/Thesis/insar/des/f2/intf_all/Attempt6/SBAS_SMOOTH_0.0000e+00/INT3/"  # Lorax
    file_type = "LOS_*_INT3.grd"
    # file_type = "LOS_20190709_INT3.grd"
    outdir = 'preview'

    subprocess.call(['mkdir', '-p', outdir], shell=False)

    file_names = glob.glob(file_dir + file_type)
    if len(file_names) == 0:
        print("Error! No files matching search pattern.")
        sys.exit(1)
    print("Reading " + str(len(file_names)) + " files.")
    num_plots_x = 6
    num_plots_y = 13
    return [file_names, outdir, num_plots_x, num_plots_y]


# ------------- INPUTS ------------ #
def inputs(file_names):
    try:
        [xdata, ydata] = netcdf_read_write.read_grd_xy(file_names[0])  # can read either netcdf3 or netcdf4.
    except TypeError:
        [xdata, ydata] = netcdf_read_write.read_netcdf4_xy(file_names[0])
    data_all = []

    file_names = sorted(file_names)  # To force into date-ascending order.

    titles = []

    for ifile in file_names:  # Read the data
        try:
            data = netcdf_read_write.read_grd(ifile)
        except TypeError:
            data = netcdf_read_write.read_netcdf4(ifile)
        data_all.append(data)
        # LOS_20161121_INT3.grd
        titles.append(ifile[-17:-9])

    return [xdata, ydata, data_all, titles]



# ------------- PLOTTING ------------ #


def insar_panels(xdata, ydata, data_all, outdir, num_plots_x, num_plots_y, titles):

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
        im = ax.imshow(data_all[count], cmap='jet', aspect=0.75)
        # ax.plot(400, 700, marker='o', markersize=5, color='black', zorder=10000)
        ax.set_title(titles[count], fontsize=20, color='black')
        ax.invert_yaxis()
        ax.invert_xaxis()

        print(titles[count])
        count += 1
 
    cbar = ax.cax.colorbar(im)
    # cbar = grid.cbar_axes[0].colorbar(im)

    # cbar.ax.set_yticks(np.arange(0, 1.1, 0.5))
    # cbar.ax.set_yticklabels(['low', 'medium', 'high'])

    # plt.show()
    plt.savefig("time-series-1.eps")
    plt.close()




def point_ts(xdata, ydata, data_all, outdir, num_plots_x, num_plots_y, titles):

    dates = []
    range_change = []

    # region = [400, 475, 675, 775] # Resurgent dome
    region = [775, 825, 1275, 1350] # Pahoa

    fig = plt.figure(figsize=(15, 6))


    # Plot deformation map with selected region/pixels overlain
    ax1 = plt.subplot(121)
    # grid[0].set_axis_off()
    im = ax1.imshow(data_all[-1], cmap='jet', aspect=0.75)

    ax1.plot([region[0], region[0], region[1], region[1], region[0]], [region[2], region[3], region[3], region[2], region[2]], color='black', zorder=10000)
    ax1.set_title(titles[-1], fontsize=12, color='black')
    ax1.invert_yaxis()
    ax1.invert_xaxis()
    cbar = fig.colorbar(im)



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
    plt.xlabel('Date')
    plt.ylabel('LOS range change (m)')

    plt.show()
    # plt.savefig("time-series-region.eps")
    # plt.close()







if __name__ == "__main__":
    top_level_driver()
