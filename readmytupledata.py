#!/usr/bin/python
# Written by Saeed Mohanna, July 2019
import numpy as np
import collections
from datetime import datetime
import netcdf_read_write

data = collections.namedtuple('data', ['filepaths', 'dates_correct', 'date_deltas',  'xvalues', 'yvalues', 'zvalues'])

def reader(filepathslist):
    """This function takes in a list of filepaths that each contain a 2d array of data, effectively taking
    in a cuboid of data. It splits and stores this data in a named tuple which is returned. This can then be used
    to extract key pieces of information."""
    filepaths  = []
    dates_correct , date_deltas = [], []
    xvalues, yvalues, zvalues = [], [], []


    for i in range(len(filepathslist)):
        print(filepathslist[i])
        filepaths.append(filepathslist[i])
        datesplit = filepathslist[i].split('/')

        print(datesplit)

        date_new = datesplit[-2].replace(datesplit[-2][0:7], str(int(datesplit[-2][0:7]) + 1))
        date_new = date_new.replace(date_new[8:15], str(int(date_new[8:15]) + 1))
        dates_correct.append(date_new)
        delta = abs(datetime.strptime(dates_correct[i][0:7], '%Y%j') - datetime.strptime(dates_correct[i][8:15], '%Y%j'))
        date_deltas.append(delta.days/365.24)
        xdata, ydata, zdata = netcdf_read_write.read_grd_xyz(filepathslist[i])
        xvalues=xdata
        yvalues=ydata
        zvalues.append(zdata)
        if i == round(len(filepathslist)/2):
            print('halfway done reading files...')

    # for i in range(len(filepathslist)):
    #     print(filepathslist[i])
    #     filepaths.append(filepathslist[i])
    #     datesplit = filepathslist[i].split('/')
    #     date_new = datesplit[-2].replace(datesplit[-2][0:7], str(int(datesplit[-2][0:7]) + 1))
    #     date_new = date_new.replace(date_new[8:15], str(int(date_new[8:15]) + 1))
    #     dates_correct.append(date_new)
    #     delta = abs(datetime.strptime(dates_correct[i][0:7], '%Y%j') - datetime.strptime(dates_correct[i][8:15], '%Y%j'))
    #     date_deltas.append(delta.days/365.24)
    #     xdata, ydata, zdata = netcdf_read_write.read_grd_xyz(filepathslist[i])
    #     xvalues=xdata
    #     yvalues=ydata
    #     zvalues.append(zdata)
    #     if i == round(len(filepathslist)/2):
    #         print('halfway done reading files...')

    mydata = data(filepaths=np.array(filepaths), dates_correct=np.array(dates_correct), date_deltas=np.array(date_deltas), xvalues=np.array(xvalues), yvalues=np.array(yvalues), zvalues=np.array(zvalues))
