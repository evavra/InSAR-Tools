import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import datetime as dt


"""
Plotting methods for InSAR time series with GMTSAR
"""


def network(intfTable, baselineTable, **kwargs):
    """
    Make interferogram network/baseline plot
    """
    # Establish figure
    fig = plt.figure(1, (10, 5))
    ax = plt.gca()
    master = intfTable['Master']
    repeat = intfTable['Repeat']
    mbl = []
    rbl = []

    # Get relative baselines
    for i in range(len(intfTable)):

        # Search baselineTable for master baseline
        for j, namestr in enumerate(baselineTable['Stem']):
            if intfTable['Master'][i].strftime('%Y%m%d') in namestr:
                mbl.append(baselineTable['OrbitBaseline'][j])
                break

        # Search baselineTable for repeat baseline
        for j, namestr in enumerate(baselineTable['Stem']):
            if intfTable['Repeat'][i].strftime('%Y%m%d') in namestr:
                rbl.append(baselineTable['OrbitBaseline'][j])
                break

    # Manualy set supermaster baseline
    superbl = -48.578476

    # Plot interferogram pairs as lines
    for i in range(len(intfTable)):
        plt.plot([master[i], repeat[i]], [mbl[i] - superbl, rbl[i] - superbl], c='k', lw=0.5)

    # Plot scenes over pair lines
    if 'sceneTable' in kwargs:
        sceneTable = kwargs['sceneTable']
        im = plt.scatter(baselineTable['Dates'], baselineTable['OrbitBaseline'].subtract(superbl), s=30, c=sceneTable['MeanCorr'], zorder=3, cmap='Spectral_r', vmin=0, vmax=1)
        plt.colorbar(im, label='Mean coherence')

    else:
        plt.scatter(master, np.array(mbl) - superbl, s=30, c='C0', zorder=3)
        plt.scatter(repeat, np.array(rbl) - superbl, s=30, c='C0', zorder=3)

    # Figure features
    plt.grid(axis='x', zorder=1)
    plt.xlim(min(master) - dt.timedelta(days=50), max(repeat) + dt.timedelta(days=50))
    # plt.ylim(int(np.ceil((min(baselineTable[4]) - superbl - 50) / 50.0) ) * 50, int(np.floor((max(baselineTable[4]) - superbl + 50) / 50.0)) * 50)
    plt.xlabel('Year')
    plt.ylabel('Baseline relative to master (m)')
    plt.show()

    if 'figName' in kwargs:
        print('Saving to {}...'.format(kwargs['figName']))
        fig.savefig(kwargs['figName'] + '.eps')
        plt.close()


def scenes(sceneTable, dataType, **kwargs):
    """
    Plot mean coherence of each SAR scene for a given set of interferograms
    ---- INPUT ----------------------------------------------
        intfTable - interferogram list with baseline and coherence data
    ---- OPTIONAL --------------------------------------------
        ax - axis handle for plotting
        cmap - colormap handle
    """
    # Check for passes axis handle
    if 'ax' in kwargs:
        ax = kwargs['ax']
    else:
        ax = plt.gca()

    # Check for passed colorbar axis handle
    if 'cax' in kwargs:
        cax = kwargs['cax']
    else:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad=0.05)

    # Plot settings
    ax.set_xlim(sceneTable['Date'].min(), sceneTable['Date'].max())
    ax.set_xlabel('Date')

    # Make actual plot

    if dataType == 'MeanCorr':
        ax.set_ylabel('Mean coherence')

    elif dataType == 'OrbitBaseline':
        ax.set_ylabel('Mean orbital baseline (m)')

    elif dataType == 'TempBaseline':
        ax.set_ylabel('Mean temporal baseline (days)')

    # Normalize data to make ImageGrid happy
    normData = abs(sceneTable[dataType] / sceneTable[dataType].max())

    # Plot data
    im = ax.scatter(sceneTable['Date'], normData, c=sceneTable['TotalCount'])

    # Get tick labels that correspond with original data
    ticks = np.linspace(0, np.round(np.ceil(sceneTable[dataType].max()), 1), 5)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_yticklabels(ticks)

    plt.colorbar(im, cax=cax, label='Number of interferograms')

    return im


def baselineCorr(intfTable, **kwargs):
    """
    Plot temporal baseline versus mean coherence
    ---- INPUT ----------------------------------------------
        intfTable - interferogram list with baseline and coherence data
    ---- OPTIONAL --------------------------------------------
        ax - axis handle for plotting
        cmap - colormap handle
    """

    # Check for passed axis handle
    if 'ax' in kwargs:
        ax = kwargs['ax']
    else:
        ax = plt.gca()

    # Check for passed colorbar axis handle
    if 'cax' in kwargs:
        cax = kwargs['cax']
    else:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad=0.05)

    # Check for passed colormap handle
    if 'cmap' in kwargs:
        cmap = kwargs['cmap']
    else:
        cmap = 'viridis'

    # Plot settings
    ax.set_xlim(intfTable['Master'].min(), intfTable['Repeat'].max())
    # ax.set_ylim(0, 1)
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean coherence')
    # ax.set_aspect(.01)

    # Colorbar business
    baselineRange = list(range(0, int(np.ceil(intfTable['OrbitBaseline'].max() / 10) * 10)))
    n = len(baselineRange)
    cmap = cm.get_cmap(cmap, n)
    Z = [[0, 0], [0, 0]]
    levels = range(0, n)
    CS3 = ax.contourf(Z, levels, cmap=cmap)
    plt.colorbar(CS3, cax=cax, label='Orbital baseline (m)')

    # Make actual plot
    for i in range(len(intfTable)):
        lineColor = np.floor(intfTable['OrbitBaseline'][i]) / n
        im = ax.plot([intfTable['Master'][i], intfTable['Repeat'][i]], [intfTable['MeanCorr'][i], intfTable['MeanCorr'][i]], color=cmap(lineColor))

    return im
