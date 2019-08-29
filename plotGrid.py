import matplotlib.pyplot as plt
import numpy as np
import glob as glob
import sys
import datetime as dt
import subprocess
import netcdf_read_write
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Original version by Kathryn Materna
# Modified by Ellis Vavra


# TOP LEVEL DRIVER
def top_level_driver():
    [file_names, outdir, num_plots_x, num_plots_y] = configure()

    [xdata, ydata, data_all, titles] = inputs(file_names)

    # make_plots(xdata, ydata, data_all, outdir, num_plots_x, num_plots_y, titles)

    insar_panels(xdata, ydata, data_all, outdir, num_plots_x, num_plots_y, titles)
    return


# ------------- CONFIGURE ------------ #
def configure():
    # file_dir = "/Users/ellisvavra/Desktop/Thesis/S1_Processing/NSBAS/INT3/"  # Laptop
    file_dir = "/Users/ellisvavra/Thesis/insar/des/f2/intf_all/Attempt6/SBAS_SMOOTH_0.0000e+00/INT3/"  # Lorax
    file_type = "LOS_2018*_INT3.grd"
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


def make_plots(xdata, ydata, data_all, outdir, num_plots_x, num_plots_y, titles):

    for i in range(len(data_all)):

        if len(data_all) == 1:

            # The actual plotting
            f, axarr = plt.subplots(figsize=(30, 40))
            axarr.imshow(data_all[0], cmap='jet', aspect=0.75)
            axarr.invert_yaxis()
            axarr.invert_xaxis()
            axarr.get_xaxis().set_ticks([])
            axarr.get_yaxis().set_ticks([])

            axarr.set_title(titles[0], fontsize=8, color='black')
            print(titles[0])

            plt.savefig("time-series-1.eps")
            # plt.show()
            plt.close()

        elif np.mod(i, num_plots_y * num_plots_x) == 0:
            count = i

            fignum = i / (num_plots_y * num_plots_x)  # counting figures up 0 to 1 to 2....

            # Looping forward and plotting the next 12 plots...
            f, axarr = plt.subplots(num_plots_y, num_plots_x, figsize=(15, 30))
            for k in range(num_plots_y):
                for m in range(num_plots_x):
                    if count == len(data_all):
                        break

                    # The actual plotting
                    axarr[k][m].imshow(data_all[count], cmap='jet', aspect=0.75)
                    plt.colorbar()

                    axarr[k][m].invert_yaxis()
                    axarr[k][m].invert_xaxis()
                    axarr[k][m].get_xaxis().set_ticks([])
                    axarr[k][m].get_yaxis().set_ticks([])

                    axarr[k][m].set_title(titles[count], fontsize=8, color='black')
                    print(titles[count])

                    count = count + 1

            plt.savefig("time-series-1.eps")
            # plt.show()
            plt.close()

    return


def insar_panels(xdata, ydata, data_all, outdir, num_plots_x, num_plots_y, titles):

    rows = num_plots_y
    cols = num_plots_x
    count = 0

    fig = plt.figure(figsize=(rows * 12, cols * 10))

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(rows, cols),
                    axes_pad=0.05,
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
        ax.set_title(titles[count], fontsize=8, color='black')
        ax.invert_yaxis()
        ax.invert_xaxis()

        print(titles[count])
        count += 1




    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="1%", pad=0.05)

    # plt.colorbar(im, cax=cax)

    # when cbar_mode is 'single', for ax in grid, ax.cax = grid.cbar_axes[0]


    # cbar_axis = plt.axis([0.09, 0.06, 0.84, 0.02])



    # cbar = grid.cbar_axes[0].colorbar(im, cax=cax)

    cbar = ax.cax.colorbar(im)
    # cbar = grid.cbar_axes[0].colorbar(im)


    # cbar.ax.set_yticks(np.arange(0, 1.1, 0.5))
    # cbar.ax.set_yticklabels(['low', 'medium', 'high'])

    plt.show()
    # plt.savefig("time-series.eps")


if __name__ == "__main__":
    top_level_driver()
