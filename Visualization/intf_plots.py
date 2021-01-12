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

    Usage: python intf_plots.py grd_type out_dir dpi width height cmap

    INPUTS:
    grd_type - grd file stem for grids to visualize
    out_dir  - name of directory to save files to
    dpi      - dpi for resultant images
    width, height - x, y dimensions of figure in inches
    cmap     - Matplotlib colormap for plotting

    Example: python intf_plots.py phase phase_figs 500 (10, 3)
              """)


    else:
        # Define argument variables
        grd_type = sys.argv[1]
        out_dir  = sys.argv[2] 
        dpi      = sys.argv[3]
        h        = sys.argv[4]
        w        = sys.argv[5]
        cmap     = sys.argv[6]

        # Make output directory
        os.mkdir('out_dir')

        # Get intf directories
        intf_dir = np.sort(glob.glob('20*_20*'))

        if len(intf_dir) == 0:
            print('Error: No interferograms identified in {}'.format(subprocess.call('pwd')))
            sys.exit()

        # Loop through directories and make plots
        for dir in intf_dir:
            intf_path = dir + '/' + grd_type + '.grd'
            with xr.open_dataset(intf_path) as grd:

                fig, ax = plt.subplots(figsize=(w, h))
                ax.imshow(grd['z'], cmap=cmap)
                
                save_path = out_dir + '/' + dir + '_' + grd_type + '.grd'
                fig.savefig(save_path, dpi=dpi)



if __name__ == '__main__':
    main()

    # print(sys.argv[1].split())