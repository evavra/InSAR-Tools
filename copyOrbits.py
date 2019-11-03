import sys
import glob as glob
import subprocess
import os
import shutil
import datetime as dt


def copyOrbits(SAFE_filelist):
    """
    Example SAFE_filelist:
    /Users/ellisvavra/Thesis/insar/des/data/S1A_IW_SLC__1SSV_20160326T135920_20160326T135947_010541_00FA9F_2B05.SAFE

    Example output format:
    S1A_OPER_AUX_POEORB_OPOD_20160415T121448_V20160325T225943_20160327T005943.EOF
    """

    # Set home directory where orbit files are stored
    homedir = '~/S1_orbits/'

    # Read SAFE_list into a Python list
    with open(SAFE_filelist, "r") as tempFile:
        tempList = tempFile.readlines()

    print('SAFE_filelist:')
    SAFE_list = []
    for line in tempList:
        SAFE_list.append(line)
        print(line)

    # Create list of dates to search orbit directory using UNIX commands
    print('Search strings...')
    searchList = []
    for line in SAFE_list:

        date = line[57:65]
        date1 = dt.datetime.strptime(date, '%Y%m%d') - dt.timedelta(day=1)
        date2 = dt.datetime.strptime(date, '%Y%m%d') + dt.timedelta(day=1)

        searchList.append(homedir + '/' + line[40:43] + '*V' + date2.strftime('%Y%m%d') + '*.EOF')

    # lets try something else...
    for item in searchList:
        shutil.copy(item, '.')


if __name__ == '__main__':
    copyOrbits(sys.argv[0])
