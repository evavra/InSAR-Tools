import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import collections
import netcdf_read_write
import glob

def get_date_from_xml(xml_name):
    """
    xml file has name like s1a-iw1-slc-vv-20150121t134413-20150121t134424-004270-005317-001.xml
    We want to return 20150121.
    """
    xml_name = xml_name.split('/')[-1]
    mydate = xml_name[15:23];
    return mydate



def getDatesFromList(fileName, fileType, numDates):
    """
    Retrive name from input .SAFE directory name
    ex:
    Input: S1A_IW_SLC__1SDV_20151209T135925_20151209T135955_008966_00CD9B_CA9F.SAFE
    Output: 20151209
    """

    if fileType == 'SAFE':
        print("Reading " + fileName)

        with open(fileName, 'r') as file:
            names = file.readlines()

        dates = []
        for line in names:
            dates.append(line[17:25])
            print(line[17:25])

        with open('SAFE-dates.txt', 'w') as newFile:
            for line in dates:
                newFile.write("%s\n" % line)

        print(dates)
        print('SAFE-dates.txt has been created')
        print()
        print()

        return dates

    elif fileType == "ASF":
        """
        Read in dates from Alaska Satellite Facility download file
        """
        print("Reading " + fileName)

        with open(fileName, 'r') as file:
            names = file.readlines()

        datesASF = []

        for i in range(78, 78 + numDates - 1):
            datesASF.append(names[i][80:88])
            print(names[i][80:88])

        with open('ASF-dates.txt', 'w') as newFile:
            for line in datesASF:
                newFile.write("%s\n" % line)

        print(datesASF)
        print('ASF-dates.txt has been created')

        return datesASF

    else:
        print("List must be SAFE directory names")



def plotSceneDates(dateList1, dateList2, xLabelList):
    dates1, dates2 = [], []

    for i in dateList1:
        dates1.append(dt.datetime.strptime(i, "%Y%m%d"))
        # print(dates1)

    for i in dateList2:
        dates2.append(dt.datetime.strptime(i, "%Y%m%d"))
        # print(dates2)

    # Make plot
    fig = plt.figure()

    plt.scatter(dates1, np.ones(len(dates1)) * 4.5, marker='.')
    plt.scatter(dates2, np.ones(len(dates2)) * 5.5, marker='.')
    plt.xlabel('Date')
    plt.yticks([4.5, 5.5], xLabelList)
    plt.ylim(0, 10)
    plt.grid(True)
    plt.show()
    # ax.set_aspect(aspect=0.2)



def checkItems(listIn, listSearch):
    listOut = []
    for item in listIn:
        if item not in listSearch:
            listOut.append(item)

    print(listOut)

    with open("Missing-items.txt","w") as newFile:
        for line in listOut:
            newFile.write("%s\n" % line)

    return listOut



def rename_intf_in(intf_in, swath):

    with open(intf_in) as file_list:
        tempList = file_list.readlines()

    new_list = []

    for i in range(len(tempList)):
        # s1a-iw2-slc-vv-20170707t135930-20170707t135950-017366-01d006-005:s1a-iw2-slc-vv-20170812t135932-20170812t135952-017891-01e00b-005
        new_list.append('S1_' + tempList[i][15:23] + '_ALL_' + swath + ':' + 'S1_' + tempList[i][80:88] + '_ALL_' + swath )
        print(new_list[i])

    return new_list



def readGRD(file_type):
    # Read in SAR data from .grd formatted file
    # INPUT FILES MUST BE LOCATED IN A GMTSAR DIRECTORY!
    # Path_list format:
    #   20170426_20170520/corr.grd
    #   20170426_20170601/corr.grd
    #   20170426_20170613/corr.grd
    #   ...

    # Get list of file paths
    path_list = glob.glob("*/" + file_type)

    print('Number of files to read: ' + str(len(path_list)))
    # Establish tuple
    tupleGRD = collections.namedtuple('GRD_data', ['path_list', 'xdata', 'ydata', 'zdata'])

    # Get dimensional data
    try:
        [xdata, ydata] = netcdf_read_write.read_grd_xy(path_list[0])  # can read either netcdf3 or netcdf4.
    except TypeError:
        [xdata, ydata] = netcdf_read_write.read_netcdf4_xy(path_list[0])

    # Loop through path_list to read in target datafiles
    zdata = []
    date_pairs = []
    i = 0
    for file in path_list:
        try:
            data = netcdf_read_write.read_grd(file)
        except TypeError:
            data = netcdf_read_write.read_netcdf4(file)

        zdata.append(data)
        pairname = file.split('/')[-2][0:19];
        date_pairs.append(pairname)  # returning something like '2016292_2016316' for each intf

        if i == floor(len(path_list)/2):
            print('Halfway done reading...')

    myData = tupleGRD(path_list=np.array(path_list), xdata=np.array(xdata), ydata=np.array(ydata), zdata=np.array(zdata))

    return myData



if __name__ == "__main__":
    readGRD('corr.grd')


