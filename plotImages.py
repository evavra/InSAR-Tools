import matplotlib.pyplot as plt
import matplotlib.image as img
import sys
import glob as glob
import subprocess
import numpy as np
# Original version by Kathryn Materna

def topLevelDriver():
    fileList, outdir = configure()
    makePlots(fileList, outdir)

def configure():
    file_dir = 'data'
    file_type = 'preview/quick-look.png'
    outdir = 'Preview_Summary'

    subprocess.call(['mkdir', '-p', outdir], shell=False)

    fileList = glob.glob(file_dir + "/*/" + file_type)


    if len(fileList) == 0:
        print("Error! No files matching search pattern.")
        sys.exit(1)

    print("Reading " + str(len(fileList)) + " files.")
    print(fileList)
    print("Output directory '" + outdir + "' created.")

    return fileList, outdir

"""
def plotImages(fileList, plotIndex):
   # imageList should contain paths relative to  asc/des data directories, i.e.: data/S1A_IW_SLC__1SDV_20171128T135926_20171128T135953_019466_021079_09A4.SAFE/preview/quick-look.png

    fig = plt.figure
    plt.figure(figsize=(14.13, 8.2))
    plt.rc('font', size=5)          # controls default text sizes


    for i in range(12):
        image = img.imread(fileList[i])
        plt.subplot(2,6,i+1)
        plt.title(fileList[i][22:30])
        plt.imshow(image)

    plt.show()
    plt.savefig(outdir + "/selected_data_" + str(int(plotIndex)) + ".eps")
    print("Plot " + (plotIndex+1) + "saved as " + outdir + "/selected_data_" + str(int(i)) + ".eps")
"""

def makePlots(fileList, outdir):
    # Find number of images to plot
    n = len(fileList)

    if (n % 12) == 0:
        nPlots = int(n/12)
    else:
        nPlots = int(np.floor(n/12) + 1)

    for h in range(nPlots):
        # old: plotImages(fileList[i:i+12], i)
        newList = fileList[h:h+12]

        fig = plt.figure
        plt.figure(figsize=(14.13, 8.2))
        plt.rc('font', size=5)          # controls default text sizes

        for i in range(12):
            image = img.imread(fileList[i])
            plt.subplot(2,6,i+1)
            plt.title(fileList[i][22:30])
            plt.imshow(image)

        plt.show()
        plt.savefig(outdir + "/selected_data_" + str(int(h)) + ".eps")
        print("Plot " + str(int(h+1)) + " saved as /" + outdir + "/selected_data_" + str(int(h)) + ".eps")
        plt.close()

    return



if __name__ == "__main__":
    topLevelDriver()
