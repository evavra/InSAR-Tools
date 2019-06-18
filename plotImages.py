import matplotlib.pyplot as plt
import matplotlib.image as img
import sys
import glob as glob
import subprocess
import numpy as np

# Original version by Kathryn Materna
# Modified by Ellis Vavra

def topLevelDriver():
    fileList, outdir = configure()
    makePlots(fileList, outdir, 10)
    subprocess.call(['open', outdir], shell=False)

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


def makePlots(fileList, outdir, numPlots):
    # Find number of images to plot
    n = len(fileList)

    if (n % numPlots) == 0:
        nPlots = int(n/numPlots)
    else:
        nPlots = int(np.floor(n/numPlots) + 1)

    for h in range(nPlots):
        newList = fileList[(h * numPlots):(h + 1) * numPlots]

        fig = plt.figure
        plt.figure(figsize=(25, 18))
        plt.rc('font', size=8)          # controls default text sizes

        for i in range(len(newList)):
            image = img.imread(newList[i])
            plt.subplot(2,numPlots/2,i+1)
            plt.title(newList[i][5:8] + "*" + newList[i][22:30] + "*" + newList[i][68:77])
            plt.imshow(image)

        plt.savefig(outdir + "/selected_data_" + str(int(h+1)) + ".eps")
        print("Plot " + str(int(h+1)) + " saved as /" + outdir + "/selected_data_" + str(int(h+1)) + ".eps")
        plt.close()

    return



if __name__ == "__main__":
    topLevelDriver()
