import insarPlots
import subprocess
from new_baseline_table import readIntfList
from readGRD import readInSAR
import matplotlib.pyplot as plt
import seismoPlots

# Specify intf list file
intfList = '/Users/ellisvavra/Thesis/insar/des/f2/intf_all/intfs_for_CANDIS_09252019.CANDIS'
intfDir = '/Users/ellisvavra/Thesis/insar/des/f2/intf_all/'
gridType = 'corr'

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

    if 'phase' in gridType:
        im = insarPlots.map(x, y, z, 'jet', [-3.1459, 3.1459], 'des', 'orig', ax)
        print(min(x))

    elif 'unwrap' in gridType:
        im = insarPlots.map(x, y, z, 'Spectral', [], 'des', 'orig', ax)
        print(min(x))

    elif 'corr' in gridType: 
        im = insarPlots.map(x, y, z, 'viridis_r', [0, 1], 'des', 'orig', ax) 
        im.set_clim([0, 0.1])

    plt.colorbar(im)
    plt.title(file[0].strftime('%Y%m%d') + '_' + file[1].strftime('%Y%m%d'))
    plt.plot([1000, 13000, 13000, 1000, 1000], [2000, 2000, 4500, 4500, 2000])
    plt.axis([20000, 0, 6000, 0])
    plt.show()

    # qualityList.append(input('Label interferogram as good (g) or bad (b)'))
    # print(len(qualityList))



# if __name__ == '__main__':
# 20150612 20