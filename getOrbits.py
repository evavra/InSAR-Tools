import sys
import glob as glob
import subprocess


"""
getOrbits:
    Python script for downloading Sentinel-1 orbits for a given set of SAFE folders.
"""


def makeOrbitList():
    print('Translating makeOrbitList from Organize_files_tops.csh still underway...')


def getOrbitURL(filename):
    """
    From list of Sentinel-1 orbit file names, get the corresponding URLs for the Sentinel-1 Quality Control data portal (https://qc.sentinel1.eo.esa.int/)
    Input filename list:
    S1B_OPER_AUX_POEORB_OPOD_20190619T110540_V20190529T225942_20190531T005942.EOF
    S1A_OPER_AUX_POEORB_OPOD_20190619T120814_V20190529T225942_20190531T005942.EOF
    ...

    Output URL list:
    http://aux.sentinel1.eo.esa.int/POEORB/2019/06/19/S1B_OPER_AUX_POEORB_OPOD_20190619T110540_V20190529T225942_20190531T005942.EOF
    http://aux.sentinel1.eo.esa.int/POEORB/2019/06/19/S1A_OPER_AUX_POEORB_OPOD_20190619T120814_V20190529T225942_20190531T005942.EOF
    ...
    """

    # Set data portal root URL
    rootURL = 'http://aux.sentinel1.eo.esa.int/POEORB'

    # Read orbitList into a Python list
    with open(filename, "r") as tempFile:
        EOF_list = tempFile.readlines()

    orbitList = []
    for line in EOF_list:
        orbitList.append(line)
        print()


    # Create list of URLS from list of EOF filenames
    urls = []

    for fileName in orbitList:
        urls.append(rootURL + "/" + fileName[25:29] + "/" + fileName[29:31] + "/" + fileName[31:33] + "/" +  fileName[:-1])
        print(rootURL + "/" + fileName[25:29] + "/" + fileName[29:31] + "/" + fileName[31:33] + "/" +  fileName[:-1])

    return urls


def downloadOrbits(urls):
    """
    Takes a list of URLS (see description for getOrbitURL) and downloads the appropriate files through the Sentinel-1  Quality Control data portal.
    """

    for item in urls:
        print('Downloading ' + item[50:])
        subprocess.call(['wget', item], shell=False)


if __name__ == '__main__':

    urls = getOrbitURL('/Users/ellisvavra/Thesis/insar/des/data/orbits.list')
    print(urls)
    downloadOrbits(urls)


