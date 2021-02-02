#!/bin/python -f

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
import glob
import subprocess
import os


def main():

    if len(sys.argv) < 3:
        print("""

    Make interferogram plots for a given datraset

    Usage: python intf_plots.py grd_path out_dir dpi width height cmap

    INPUTS:
    grd_path - wildcard path for grd files to plot
    out_dir  - name of directory to save files to
    dpi      - dpi for resultant images
    width, height - x, y dimensions of figure in inches
    cmap     - Matplotlib colormap for plotting

    Example: python intf_plots.py phase phase_figs 500 (10, 3)
              """)


    else:
        # Define argument variables
        grd_path = sys.argv[1]
        out_dir  = sys.argv[2] 
        dpi      = sys.argv[3]
        h        = sys.argv[4]
        w        = sys.argv[5]
        cmap     = sys.argv[6]

        # Make output directory
        if len(glob.glob(out_dir)) == 0
            os.mkdir(out_dir)

        # Get intf directories
        # intf_dir = np.sort(glob.glob('20*_20*'))
        intf_list = np.sort(glob.glob(grd_path))

        if len(intf_dir) == 0:
            print('Error: No interferograms identified in {}'.format(subprocess.call('pwd')))
            sys.exit()

        # Loop through files and make plots
        for intf_path in intf_list:

            pieces = intf_path.split('/')
            new_name = pieces[-2] + '_' + pieces[-1]

            with xr.open_dataset(intf_path) as grd:

                fig, ax = plt.subplots(figsize=(w, h))
                ax.imshow(grd['z'], cmap=cmap)
                
                save_path = out_dir + '/' + new_name
                # save_path = out_dir + '/' + dir + '_' + grd_type + '.grd'
                fig.savefig(save_path, dpi=dpi)



if __name__ == '__main__':
    main()

    # print(sys.argv[1].split())