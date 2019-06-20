import subprocesss
import glob as glob

"""
getOrbits:
    Python script for downloading Sentinel-1 orbits for a given set of SAFE folders.
"""


def makeOrbitList():
    print('Translating makeOrbitList from Organize_files_tops.csh still underway...')


def getOrbitURL(orbitList):
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

    rootURL = 'http://aux.sentinel1.eo.esa.int/POEORB'

    urls = []

    for fileName in orbitList:
        urls.append(rootURL + "/" + fileName[25:29] + "/" + fileName[29:31] + "/" + fileName[31:33] + "/" +  fileName)
        print(rootURL + "/" + fileName[25:29] + "/" + fileName[29:31] + "/" + fileName[31:33] + "/" +  fileName))

    return urls


def downloadOrbits(urls):
    """
    Takes a list of URLS (see description for getOrbitURL) and downloads the appropriate files through the Sentinel-1  Quality Control data portal.
    """

    for item in urls:
        subprocess.call(['wget', item], shell=False)












if __name__ == '__main__':

    testList = ['S1B_OPER_AUX_POEORB_OPOD_20190619T110540_V20190529T225942_20190531T005942.EOF','S1A_OPER_AUX_POEORB_OPOD_20190619T120814_V20190529T225942_20190531T005942.EOF']

    getOrbitURL(testList)


