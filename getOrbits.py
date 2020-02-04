import sys
import glob as glob
import subprocess
import datetime as dt
import requests

"""
getOrbits:
    Python script for downloading Sentinel-1 orbits for a given set of SAFE folders.
"""


def getOrbitList(url):
    # Gets list of all .EOF filenames from specified URL

    # Scrape current list of S1A/B orbit filenames
    print('Getting list of current orbit files from ' + url + ' ...')
    orbitHTML = requests.get(url)
    tempList = list(orbitHTML.text.split("\n")[4:-7])

    # Save names to list
    orbitList = []
    for line in tempList:
        orbitList.append(line[9:86])

    return orbitList


def getOrbitURL(orbitList, dirList):
    # Get list of EOF file URLs based on input list of directories
    # orbitList format:
        # S1A_OPER_AUX_POEORB_OPOD_20180228T120602_V20180207T225942_20180209T005942.EOF
        # S1A_OPER_AUX_POEORB_OPOD_20180312T120552_V20180219T225942_20180221T005942.EOF
        # S1A_OPER_AUX_POEORB_OPOD_20180324T120757_V20180303T225942_20180305T005942.EOF
        # ...

    # dirList format:
        # S1A_IW_SLC__1SDV_20191013T135939_20191013T140006_029441_035951_9DBD.SAFE
        # S1A_IW_SLC__1SDV_20191025T135939_20191025T140006_029616_035F52_BCBC.SAFE
        # S1A_IW_SLC__1SDV_20191106T135939_20191106T140006_029791_03657D_4FEF.SAFE
        # ...

    print('Matching filenames from ' + dirList + ' ...')

    # Create reference list of directory satellite IDs and dates
    refList = []
    with open(dirList) as file:
        for line in file:
            refList.append([line[0:3], dt.datetime.strptime(line[17:25], '%Y%m%d')])

    # Find filename for each aquisition in refList
    downloadList = []

    for file in refList:

        for orbit in orbitList:
            # Create string to validate with orbit filenames (does not include upload date)
            searchStr = '_V' + (file[1] - dt.timedelta(days=1)).strftime('%Y%m%d') + 'T225942_' + (file[1] + dt.timedelta(days=1)).strftime('%Y%m%d') + 'T005942.EOF'

            if searchStr in orbit:
                if file[0] in orbit:
                    downloadList.append(orbit)
                    print(file[0] + ' ' + file[1].strftime('%Y%m%d') + ': Matched')
                    tag = 1

        if tag == 1:
            tag = 0
        else:
            # print('Orbit file not available for ' + file[0] + ' ' + file[1].strftime('%Y%m%d'))
            print(file[0] + ' ' + file[1].strftime('%Y%m%d') + ': NO FILE FOUND')
            tag = 0

    return downloadList


def downloadOrbits(url, downloadList, saveDir):
    """
    Takes a list of URLS (see description for getOrbitURL) and downloads the appropriate files through the Sentinel-1  Quality Control data portal.
    """

    for file in downloadList:
        print('Downloading ' + file + '...')
        subprocess.call(['wget', url + "/" + file[25:29] + "/" + file[29:31] + "/" + file[31:33] + "/" + file], shell=False)


if __name__ == '__main__':
    listURL = "https://s1qc.asf.alaska.edu/aux_poeorb"  # May need to get edited in the future
    orbitURL = 'http://aux.sentinel1.eo.esa.int/POEORB'
    dirList = 'SAFE_filelist'
    saveDir = '/Users/ellisvavra/Downloads'

    orbitList = getOrbitList(listURL)
    downloadList = getOrbitURL(orbitList, dirList)
    downloadOrbits(orbitURL, downloadList, saveDir)

# OLD
"""
# def getOrbitURL(filename):

#     # Set data portal root URL
#     rootURL = 'http://aux.sentinel1.eo.esa.int/POEORB'

#     # Read orbitList into a Python list
#     with open(filename, "r") as tempFile:
#         EOF_list = tempFile.readlines()

#     orbitList = []
#     for line in EOF_list:
#         orbitList.append(line)
#         print()

#     # Create list of URLS from list of EOF filenames
#     urls = []

#     for fileName in orbitList:
#         urls.append(rootURL + "/" + fileName[25:29] + "/" + fileName[29:31] + "/" + fileName[31:33] + "/" + fileName[:-1])
#         print(rootURL + "/" + fileName[25:29] + "/" + fileName[29:31] + "/" + fileName[31:33] + "/" + fileName[:-1])

#     return urls
"""
