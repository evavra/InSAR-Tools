import datetime as dt
import glob as glob
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from PyGMTSAR import addOrder
from PyGMTSAR import filtIntfTable
from PyGMTSAR import getSceneTable
from PyGMTSAR import readBaselineTable
from PyGMTSAR import readIntfTable
from mpl_toolkits.axes_grid1 import ImageGrid


# ----------- DRIVERS ----------
def driver():
    # Read metadata
    # baselineTableFile = '/Users/ellisvavra/Desktop/LongValley/LV-InSAR/baseline_table_des.dat'
    # intfTableFile = '/Users/ellisvavra/Desktop/LongValley/LV-InSAR/intf_table_NN15.dat'
    # pathStem = '/Users/ellisvavra/Desktop/LongValley/Tests/des/intf_all/'
    # baselineTable = readBaselineTable(baselineTableFile)
    # intfTable = readIntfTable(intfTableFile)
    # intfTable = addOrder(intfTable, baselineTable)

    intf_path = '/Users/ellisvavra/Desktop/LongValley/Tests/des/intf_all/2014311_2014335/'
    temp = nc.Dataset(intf_path + '/unwrap.grd', 'r+', format='NETCDF4')
    intf = np.array(temp.variables['z'])
    temp.close()

    intf_m = rad2m(13.2860, units='m')

    # plt.hist(intf_m, bins=100)
    # plt.show()

def plot_all_driver():

    # ---- N = 1 ------------------------------------
    # Read metadata
    baselineTableFile = '/Users/ellisvavra/Desktop/LongValley/LV-InSAR/baseline_table_des.dat'
    intfTableFile = '/Users/ellisvavra/Desktop/LongValley/LV-InSAR/intf_table_NN15.dat'
    baselineTable = readBaselineTable(baselineTableFile)
    intfTable = readIntfTable(intfTableFile)
    intfTable = addOrder(intfTable, baselineTable)  # Add interferogram order to table (should eventually get integrated with makeIntfTable)

    # OPTIONAL: select interferograms to use in CSS based off of baseline, order,
    maxOrder = 1
    intfTable = intfTable[intfTable['Order'] <= maxOrder].reset_index(drop=True)

    print('Number of interferograms to use: ' + str(len(intfTable)))
    print()

    # Get individual scene information
    sceneTable = getSceneTable(intfTable)

    # Print valid/invalid dates
    APS1, balance1 = css(sceneTable, intfTable, '/Users/ellisvavra/Desktop/LongValley/Tests/des/intf_all/', 2, 6)

    # ---- N = 2 ------------------------------------
    baselineTable = readBaselineTable(baselineTableFile)
    intfTable = readIntfTable(intfTableFile)
    intfTable = addOrder(intfTable, baselineTable)

    # OPTIONAL: select interferograms to use in CSS based off of baseline, order,
    maxOrder = 2
    intfTable = intfTable[intfTable['Order'] <= maxOrder].reset_index(drop=True)

    print('Number of interferograms to use: ' + str(len(intfTable)))
    print()

    # Get individual scene information
    sceneTable = getSceneTable(intfTable)

    # Print valid/invalid dates
    APS2, balance2 = css(sceneTable, intfTable, '/Users/ellisvavra/Desktop/LongValley/Tests/des/intf_all/', 2, 6)

    # ---- N = 8 ------------------------------------
    baselineTable = readBaselineTable(baselineTableFile)
    intfTable = readIntfTable(intfTableFile)
    intfTable = addOrder(intfTable, baselineTable)

    # OPTIONAL: select interferograms to use in CSS based off of baseline, order,
    maxOrder = 10
    intfTable = intfTable[intfTable['Order'] <= maxOrder].reset_index(drop=True)

    print('Number of interferograms to use: ' + str(len(intfTable)))
    print()

    # Get individual scene information
    sceneTable = getSceneTable(intfTable)

    # Print valid/invalid dates
    APSN, balance8 = css(sceneTable, intfTable, '/Users/ellisvavra/Desktop/LongValley/Tests/des/intf_all/', 2, 6)

    # ---- PLOT ------------------------------------
    # Plot APS

    # fig3 = plt.figure(1, (13.75, 8.75))
    # grid3 = ImageGrid(fig3, 111,
    #               nrows_ncols=(9, 17),
    #               axes_pad=0.,
    #               share_all=True,
    #               aspect=1,
    #               cbar_mode='single'
    #               )

    # for i, ax in enumerate(grid3):
    #     if i >= len(APS1):
    #         ax.imshow(np.zeros(APS1[0].shape), 'Spectral_r')
    #     else:
    #         im = ax.imshow(np.flip(np.flip(APS1[i]), 0), 'Spectral_r')

    # cbar = plt.colorbar(im, label='Phase', cax=grid3.cbar_axes[0])
    # plt.savefig('/Users/ellisvavra/Desktop/LongValley/Tests/des/CSS/test.ps')

    for i in range(len(APS1)):
        fig = plt.figure(1, (13.75, 8.75))

        grid = ImageGrid(fig, 111,
                         nrows_ncols=(1, 3),
                         axes_pad=0.,
                         share_all=True,
                         aspect=1,
                         cbar_mode='single'
                         )

        # vrange = [min([np.nanmin(APS1[i]), np.nanmin(APS2[i]), np.nanmin(APSN[i])]), max([np.nanmax(APS1[i]), np.nanmax(APS2[i]), np.nanmax(APSN[i])]) ]

        fig.suptitle(sceneTable['Date'][i].strftime('%Y-%m-%d'), fontsize=16)
        grid[0].imshow(APS1[i] * 55.6 / (4 * np.pi), cmap='Spectral')
        grid[1].imshow(APS2[i] * 55.6 / (4 * np.pi), cmap='Spectral')
        im = grid[2].imshow(APSN[i] * 55.6 / (4 * np.pi), cmap='Spectral')
        grid[0].set_title('N = 1 (Net {} days)'.format(balance1[i]))
        grid[1].set_title('N = 2 (Net {} days)'.format(balance2[i]))
        grid[2].set_title('N = {} '.format(maxOrder) + ' (Net {} days)'.format(balance8[i]))
        grid[0].invert_xaxis()
        grid[1].invert_xaxis()
        grid[2].invert_xaxis()
        plt.colorbar(im, label='LOS (mm)', cax=grid.cbar_axes[0])
        plt.show()

    # for i, aps in enumerate(APS):
        # fig = plt.figure(1, (9,6))
        # ax = plt.gca()
        # ax.imshow(aps, cmap='Spectral_r')
        # ax.set_title(sceneTable['Date'][i].strftime('%Y-%m-%d') + ' (Net {} days)'.format(balance[i]))
        # ax.invert_xaxis()
        # plt.show()

# ---------- CALCULATIONS ----------


def anc(apsList):
    """
    Calculate atmosphereic noise coefficients (ANC) for input InSAR scenes.

    INPUT:
    apsList - list containing atmospheric phase screen estimates

    OUTPUT:
    ancList - list of atmospheric noise coefficients
    """
    print('Calculating Atmospheric Noise Coefficients...')
    print()

    aps_rms = []

    for aps in apsList:
        # Correct for nans
        aps_corrected = aps.flatten()[np.isnan(aps.flatten()) == False]

        # Get mean phase value of
        aps_mean = np.ones(len(aps_corrected)) * np.mean(aps_corrected.flatten())

        # Calculate APS RMS
        aps_rms.append(np.sqrt(np.sum((aps_corrected - aps_mean)**2) / (len(aps.flatten()))))

    # Normalize ANC to max. value(Rmax term in Tymofyeyeva & Fialko, 2015)
    ancList = list(10 * np.array(aps_rms)) / max(aps_rms)

    return ancList


def rad2m(grid_rad, **kwargs):
    """
    Convert number or grid from radians to meters

    INPUT:
    data_rad - data in radians

    OUTPUT:
    data_m - data in metric units

    Optional keyword arguments:
    wavelength - sensor wavelength in meters (default is 0.0556m for C-band)
    units - 'm', 'cm', or 'mm'
    """

    # Set wavelength
    if 'wavelength' in kwargs:
        wavelength = kwargs['wavelength']
    else:
        wavelength = 0.0556

    # Set unit scalind (default is meters)
    if 'units' in kwargs:
        if kwargs['units'] == 'm':
            units = 1
        elif kwargs['units'] == 'cm':
            units = 100
        elif kwargs['units'] == 'mm':
            units = 1000
    else:
        units = 1

    grid_m = grid_rad * wavelength / (4 * np.pi) * units

    return grid_m


def m2rad(data_m, **kwargs):
    """
    Convert number or grid from meters to radians

    INPUT:
    data_m - data in metric units

    OUTPUT:
    data_rad - data in radians

    Optional keyword arguments:
    wavelength - sensor wavelength in meters (default is 0.0556m for C-band)
    units - 'm', 'cm', or 'mm'
    """

    # Set wavelength
    if 'wavelength' in kwargs:
        wavelength = kwargs['wavelength']
    else:
        wavelength = 0.0556

    # Set unit scalind (default is meters)
    if 'units' in kwargs:
        if kwargs['units'] == 'm':
            units = 1
        elif kwargs['units'] == 'cm':
            units = 100
        elif kwargs['units'] == 'mm':
            units = 1000
    else:
        units = 1

    data_rad = 4 * np.pi * data_m / (wavelength * units)

    return data_rad


# ---------- ANALYSIS ----------
def css(sceneTable, intfTable, pathStem, stack_min, stack_max):
    """
    Perform common scene stacking for input list of scenes and interferograms.

    INPUT:
    sceneTable - sceneTable for stack of SAR acquisitions
    intfTable - intfTable for interferograms to use in CSS
    pathStem - path to directory contatining interferograms
    stack_min - minimum number of valid interferograms to use in stacking
    stack_max - maximum number of valid interferograms to use in stacking

    OUTPUT:
    APS - list containting estimated atmospheric phase screens (APS) as Numpy grids
    balance - list containing temporal balance count (days). Positive values indicate scene is used more frequently as a repeat, negative for master.

    References:
    Tymofyeyeva, E., and Y. Fialko (2015), Mitigation of atmospheric phase delays in InSAR data, with application to the eastern California shear zone, J. Geophys. Res. Solid Earth, 120, 5952–5963, doi:10.1002/2015JB011886.

    Wang, K., & Fialko, Y. (2018). Observations and modeling of coseismic and postseismic deformation due to the 2015 Mw 7.8 Gorkha (Nepal) earthquake. Journal of Geophysical Research: Solid Earth, 123. https://doi.org/10.1002/2017JB014620
    """

    print('Starting Common Scene Stacking...')
    print()

    APS = []
    balance = []

    for i, date in enumerate(sceneTable['Date']):

        # Pass on dates that don't satisfy the input parameters
        if sceneTable['Count'][i] < stack_min:
            print('Skipping ' + date.strftime('%Y%m%d') + ', only used in {} interferograms'.format(sceneTable['Count'][i]))
            APS.append(False)
            balance.append(np.nan)

        else:
            # print('Getting list of interferograms using ' + date.strftime('%Y%m%d'))

            masters = []
            masterDays = []
            repeats = []
            repeatDays = []

            # Get list of intfs containing date
            for j in range(len(intfTable)):
                if date == intfTable['Master'][j]:
                    masters.append(intfTable['DateStr'][j])
                    masterDays.append(intfTable['TempBaseline'][j])
                elif date == intfTable['Repeat'][j]:
                    repeats.append(intfTable['DateStr'][j])
                    repeatDays.append(intfTable['TempBaseline'][j])

            # Do actual stacking
            print('Estmating APS on ' + date.strftime('%Y%m%d') + '...')

            # Initiate array using dimensions from first intf

            temp = nc.Dataset(pathStem + (masters + repeats)[0] + '/unwrap.grd', 'r+', format='NETCDF4')
            aps = np.zeros((temp.dimensions['y'].size, temp.dimensions['x'].size))
            temp.close()

            # Sum together repeat instances
            for array in repeats:
                temp = nc.Dataset(pathStem + array + '/unwrap.grd', 'r+', format='NETCDF4')
                aps += np.array(temp.variables['z'])
                temp.close()

            # Then difference master instances
            for array in masters:
                temp = nc.Dataset(pathStem + array + '/unwrap.grd', 'r+', format='NETCDF4')
                aps -= np.array(temp.variables['z'])
                temp.close()

            aps /= (len(masters) + len(repeats))

            APS.append(aps)
            balance.append(sum(repeatDays) - sum(masterDays))

            # Determine APS balance
            # print('Balance score: {} days'.format(balance[-1]))

    # Retroactively assign blank APS to dates with not enough available interferograms
    temp = nc.Dataset(pathStem + repeats[0] + '/unwrap.grd', 'r+', format='NETCDF4')
    blank_aps = np.zeros((temp.dimensions['y'].size, temp.dimensions['x'].size))
    temp.close()

    for i, aps in enumerate(APS):
        if aps is False:
            APS[i] = blank_aps

    return APS, balance


# ---------- PLOTS ----------
def plot_APS():

    return im


def plot_ANC(dates, ancList):

    return im


if __name__ == '__main__':
    driver()
