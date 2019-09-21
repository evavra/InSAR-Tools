import insarPlots
import subprocess
from new_baseline_table import readIntfList
from readGRD import readInSAR
import matplotlib.pyplot as plt
import seismoPlots

# Specify intf list file
intfList = '/Users/ellisvavra/Thesis/insar/des/f2/intf_all/dates.run'
intfDir = '/Users/ellisvavra/Thesis/insar/des/f2/intf_all/'
gridType = 'phase'

# Read in list of interferogram directories
intfs = readIntfList(intfList, 'date_pairs')

# Iteratavely display list of grids
qualityList = []

for file in intfs:
    # Identify full path to target grid and read in it's data
    tempGrid = intfDir + file[0].strftime('%Y%m%d') + '_' + file[1].strftime('%Y%m%d') + '/' + gridType + '.grd'
    x, y, z = readInSAR(tempGrid)

    # Establish figure and plot grid
    fig = plt.figure(figsize=(10,15))
    ax = plt.gca()

    insarPlots.map(x, y, z, 'jet', [-3.1459, 3.1459], 'des', 'ra', ax)

    plt.title(file[0].strftime('%Y/%m/%d') + ' - ' + file[1].strftime('%Y/%m/%d'))

    plt.show()

    # qualityList.append(input('Label interferogram as good (g) or bad (b)'))
    # print(len(qualityList))



# if __name__ == '__main__':
