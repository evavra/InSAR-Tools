#!/usr/bin/python
import numpy as np
import glob as glob
import sys
import datetime as dt
import subprocess
import netcdf_read_write
import readmytupledata
import matplotlib.pyplot as plt


"""
Written by Ellis Vavra, July 21, 2019

Want to be able to take a list of all interferogram directories and calculate the (1) mean coherence for each intf and (2) how many pixels are above/below a certain threshold (make a plot?)

INPUT:
- List of interferogram directories
- List of interferograms to skip
- Coherence threshold
- Count Nans

OUTPUT:
- list of mean coheence for each interferogram pair
-
"""



# -------------------------------- TOP LEVEL DRIVER -------------------------------- #

def driver():
    homedir = '/Users/ellisvavra/Thesis/insar/des/f2/CANDIS_test/'
    intf_list = 'dates.run'
    skip_list = []
    area = [0,20000,0,6000]

    test = readmytupledata.reader(homedir + intf_list)

    print(test)


# -------------------------------- CONFIGURE -------------------------------- #
# def configure():
#     for directory in intf_list:
#         # Move into interferogram directory
#         subprocess.call(['cd', directory ], shell=False)
#         # Read corr.grd
#         # readCor()
#         # Perform analysis

#         # Save files

#         # Move back out
#         subprocess.call(['cd', '..' ], shell=False)


# -------------------------------- INPUT -------------------------------- #
# def readCorr(homedir, file_names): # Removed skip_file
#     try:
#         [xdata, ydata] = netcdf_read_write.read_grd_xy(homedir + file_names[0])  # can read either netcdf3 or netcdf4.
#     except TypeError:
#         [xdata, ydata] = netcdf_read_write.read_netcdf4_xy(homedir + file_names[0])
#     data_all = []
#     date_pairs = []

#     file_names = sorted(file_names)  # To force into date-ascending order.

#     for ifile in file_names:  # Read the data
#         try:
#             data = netcdf_read_write.read_grd(ifile)
#         except TypeError:
#             data = netcdf_read_write.read_netcdf4(ifile)
#         data_all.append(data)
#         pairname = ifile.split('/')[-2][0:19];
#         date_pairs.append(pairname)  # returning something like '2016292_2016316' for each intf
#         print(pairname)

#     # skip_intfs = []
#     # if len(skip_file) > 0:
#     #     ifile = open(skip_file, 'r')
#     #     for line in ifile:
#     #         skip_intfs.append(line.split()[0])
#     #     ifile.close()

#     return [xdata, ydata, data_all, date_pairs, skip_intfs]


# -------------------------------- ANALYSIS -------------------------------- #






if __name__ == "__main__":
    driver()
