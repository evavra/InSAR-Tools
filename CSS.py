import glob as glob
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from PyGMTSAR import readBaselineTable
from PyGMTSAR import readIntfTable
from PyGMTSAR import getOrder
from PyGMTSAR import filtIntfTable


def driver():
    # Read metadata
    baselineTableFile = '/Users/ellisvavra/Desktop/LongValley/LV-InSAR/baseline_table_des.dat'
    intfTableFile = '/Users/ellisvavra/Desktop/LongValley/LV-InSAR/intf_table_NN15.dat'
    baselineTable = readBaselineTable(baselineTableFile)
    intfTable = readIntfTable(intfTableFile)
    intfTable = addOrder(intfTable, baselineTable)

    # OPTIONAL: select interferograms to use in CSS based off of baseline, order,
    intfTable = intfTable[intfTable['Order'] == 1]

    print(intfTable):
        # Print valid/invalid dates


def ANC():
    """
    Calculate atmosphereic noise coefficients for input InSAR scenes.
    """


def CSS(stack_min, stack_max):
    """
    Perform common scene stacking for input list of scenes and interferograms
    """


if __name__ == '__main__':
    driver()
