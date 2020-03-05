
import numpy as np
import matplotlib.pyplot as plt
from insarClasses import insarData
import subprocess

# NOTES --------------------------------------------------------------------------------------------------
# Need to reconfigure clipping method in insarData class.
#   - Need to be able to clip ALL of the sub-arrays within the data class (i.e. not just specified one)
#
# Or can make alignment its own method (which should happen either way) and have it make two new, clipped
# versions of the asc and des data... might be easier

# SETTINGS -----------------------------------------------------------------------------------------------
makePlots = False


# FUNCTION LIBRARY ---------------------------------------------------------------------------------------

def align(asc, des):
    print()
    print('----------------- ALIGNING GRIDS -----------------')
    ascOrg = [asc.e[0][0], asc.n[0][0]]
    print('Origin for ascending data:')
    print(ascOrg)

    print()
    desOrg = [des.e[0][0], des.n[0][0]]
    print('Origin for descending data:')
    print(desOrg)

    # Starting from origin of descending data, determine the distance between lon and lat positions
    dLon = ascOrg[0] - desOrg[0]
    dLat = ascOrg[1] - desOrg[1]  # Starting distance between ascending origin and initial descending point

    print()
    print('dLon: ' + str(dLon))
    print('dLat: ' + str(dLat))

    # Intialize new position indicies
    iLon = 1

    # When convergence has occured, this flag will be changed to true
    convergence = False

    # Start with longitude (or column index)
    while convergence == False:

        # Determine if lon. point needs to increase or decrease
        if dLon > 0:
            direction = 1
        elif dLon < 0:
            direction = -1

        # Incrementally increase/decrease longitude index to find best match
        while dLon * direction > 0:
            iLon = iLon + direction
            dLon = ascOrg[0] - des.e[0, iLon]

        iLon = iLon - direction
        dLon = ascOrg[0] - des.e[0, iLon]
        convergence = True

    # Now do latitude (or row index)
    iLat = 1
    convergence = False

    while convergence == False:

        # Determine if lat point needs to increase or decrease
        if dLat > 0:
            direction = 1
        elif dLat < 0:
            direction = -1

        # Incrementally increase/decrease latitude index to find best match
        while dLat * direction > 0:
            iLat = iLat + direction
            dLat = ascOrg[1] - des.n[iLat, 0]

        iLat = iLat - direction
        dLat = ascOrg[1] - des.n[iLat, 0]
        convergence = True

    newIndex = [iLon, iLat]

    # Clip descending track data accordingly
    index0 = np.arange(newIndex[1], min([len(asc.e), len(des.e)]))
    index1 = np.arange(newIndex[0], min([len(asc.e[0]), len(des.e[0])]))

    clip_e, clip_n, clip_z = asc.clip(index0, index1)
    ascAlign = insarData(clip_e, clip_n, clip_z)

    clip_e, clip_n, clip_z = asc.look.clip(index0, index1)
    ascAlign.addLook(clip_e, clip_n, clip_z)
    ascAlign.inc = asc.inc[index0, :]
    ascAlign.inc = ascAlign.inc[:, index1]
    ascAlign.head = asc.head[index0, :]
    ascAlign.head = ascAlign.head[:, index1]

    clip_e, clip_n, clip_z = des.clip(index0, index1)
    desAlign = insarData(clip_e, clip_n, clip_z)

    clip_e, clip_n, clip_z = des.look.clip(index0, index1)
    desAlign.addLook(clip_e, clip_n, clip_z)
    desAlign.inc = des.inc[index0, :]
    desAlign.inc = desAlign.inc[:, index1]
    desAlign.head = des.head[index0, :]
    desAlign.head = desAlign.head[:, index1]

    return ascAlign, desAlign


def delft(losAsc, losDes, incAsc, incDes, headAsc, headDes):
    # ------------------------------------------------------------------------
    # Method for decomposing InSAR LOS measurements from two different viewing
    # geometries into to the vertical component and one horizontal component
    # From Samieie-Esfahany et al. (2009), Fringe 2009 Workshop, Frascati,
    # Italy, 30 November - 4 December 2009
    # ------------------------------------------------------------------------
    # InputS:
    # All are n x 1 vectors, where n is the number of InSAR pixels/points
    #   losAsc - deformation in ascending LOS
    #   losDes - deformation in descending LOS
    #   incAsc - ascending track incident angle
    #   incDes - descending track incident angle
    #   headAsc - ascending track heading angle
    #   headDes - descending track heading angle
    # ------------------------------------------------------------------------
    # Output:
    #   up - vertical deformation
    #   azi - projection of horizontal deformation in descending azimuth
    #           look direction
    # ------------------------------------------------------------------------
    # Problem statement:
    # [ ascLos ]    [ up  ]      [ ascLos ]   [ up  ]
    # [        ] = A[     ] => A\[        ] = [     ]
    # [ losDes ]    [ azi ]      [ losDes ]   [ azi ]
    # ------------------------------------------------------------------------
    # Example implementation:
    # [up, azi] = delft(0.0357, 0.0476, 35, 38, 10, 190);

    # Check for equal dimensions
    if (len(losAsc) + len(losDes) + len(incAsc) + len(incDes) + len(headAsc) + len(headDes)) / 6 is not len(losAsc):
        print('All matrices must be of same n x 1 dimension!')
        print('losAsc: ' + str(losAsc.shape))
        print('losDes: ' + str(losDes.shape))
        print('incAsc: ' + str(incAsc.shape))
        print('incDes: ' + str(incDes.shape))
        print('headAsc: ' + str(headAsc.shape))
        print('headDes: ' + str(headAsc.shape))

    # Find satellite heading difference between ascending and descending mode
    deltaHead = headAsc - headDes
    n = len(deltaHead.flatten())

    # Initialize transformation tensor
    A = np.zeros((n, 2, 2))

    # Make transformation tensor
    print()
    print('Generating ' + str(n) + ' x 2 x 2 transformation tensor...')
    rad = np.pi / 180

    import time
    start = time.time()

    for i in range(n):
        A[i] = np.array([[np.cos(incAsc.flatten()[i] * rad), np.sin(incAsc.flatten()[i] * rad) / np.cos(deltaHead.flatten()[i] * rad)],
                         [np.cos(incDes.flatten()[i] * rad), np.sin(incDes.flatten()[i] * rad)]])

    print('Time elapsed: ' + str((time.time() - start) / 60) + ' minutes')

    B = np.array([losAsc.flatten(), losDes.flatten()])

    # Check dimensions
    print(A.shape)
    print(B.shape)

    # Perform transformation
    print()
    print('Solving for vertical and horizontal azimuth components...')
    start = time.time()
    x = np.linalg.solve(A, B)
    print('Time elapsed: ' + str((time.time() - start) / 60) + ' minutes')

    up = x[0]
    azi = x[1]

    return up, azi


def decompDriver(asc, des):
    dim = asc.e.shape

    # Initialize output component arrays
    up = np.zeros([dim[0] * dim[1], 1])
    azi = np.zeros([dim[0] * dim[1], 1])

    # For each pixel, perform vector decomposition into vertical and horizontal components
    print('Input array dimensions:')
    print('losAsc: ' + str(asc.z.shape))
    print('losDes: ' + str(des.z.shape))
    print('incAsc: ' + str(asc.inc.shape))
    print('incDes: ' + str(des.inc.shape))
    print('headAsc: ' + str(asc.head.shape))
    print('headDes: ' + str(des.head.shape))
    print()
    print('Performing vector decomposition...')
    print()
    up, azi = delft(asc.z, des.z, asc.inc, des.inc, asc.head, des.head)

    # OLD
    # # Set up % tracker
    # benchmarks = np.array(np.arange(0, 1, 0.05) * dim[0] * dim[1]).round(0)
    # for i in range(dim[0] * dim[1]):
    #     up[i], azi[i] = delft(asc.z.flatten()[i], des.clip_z.flatten()[i], asc.inc[i], des.inc[i], asc.head[i], des.head[i])
    # if i in benchmarks:
    #     print(str(np.round(i / (dim[0] * dim[1]) * 100), 0) + "% complete")

    # Print some info to see that decomposition worked
    print('Number of matching pixels: ' + str(dim[0] * dim[1] - sum(np.isnan(up.flatten()))))
    print(up.min(), up.max())
    print(azi.min(), azi.max())

    return up, azi


# PROCESSING -------------------------------------------------------------------------------------------------------
subprocess.call('clear')

# ------ PREP ASCENDING DATA ---------------------------------------------------------------------------------------
ascDataFile = '/Users/ellisvavra/Desktop/LongValley/Vector_Decomposition/asc/LOS_20191008_asc_ll.grd'
ascLookTable = '~/Desktop/LongValley/Vector_Decomposition/topo/look_asc.dat'
asc = insarData([], [], [])
asc.readFile(ascDataFile)

print()
print('Ascending data:')
print('E: ' + str(np.round(asc.e.flatten().min(), 4)) + ' ' + str(np.round(asc.e.flatten().max(), 4)) + ' ' + str(asc.e.shape))
print('N: ' + str(np.round(asc.n.flatten().min(), 4)) + ' ' + str(np.round(asc.n.flatten().max(), 4)) + ' ' + str(asc.n.shape))
print('Z: ' + str(np.round(asc.z.flatten().min(), 4)) + ' ' + str(np.round(asc.z.flatten().max(), 4)) + ' ' + str(asc.z.shape))

asc.readLook(ascLookTable)

# Plot
if makePlots is True:
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle('Ascending LOS data')
    im = plt.imshow(asc.z)
    plt.colorbar(im, label='Displacement (m)')
    plt.show()

# ------ PREP DESCENDING DATA ---------------------------------------------------------------------------------------
desDataFile = '/Users/ellisvavra/Desktop/LongValley/Vector_Decomposition/des/LOS_20191007_des_ll.grd'
desLookTable = '/Users/ellisvavra/Desktop/LongValley/Vector_Decomposition/topo/look_des.dat'
des = insarData([], [], [])
des.readFile(desDataFile)

print()
print('Descending data dimensions:')
print('E: ' + str(np.round(des.e.flatten().min(), 4)) + ' ' + str(np.round(des.e.flatten().max(), 4)) + ' ' + str(des.e.shape))
print('N: ' + str(np.round(des.n.flatten().min(), 4)) + ' ' + str(np.round(des.n.flatten().max(), 4)) + ' ' + str(des.n.shape))
print('Z: ' + str(np.round(des.z.flatten().min(), 4)) + ' ' + str(np.round(des.z.flatten().max(), 4)) + ' ' + str(des.z.shape))

des.readLook(desLookTable)

# Plot
if makePlots is True:
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle('Descending LOS data')
    im = plt.imshow(des.z)
    plt.colorbar(im, label='Displacement (m)')
    plt.show()

# ------ ALIGN GRIDS ---------------------------------------------------------------------------------------
ascAlign, desAlign = align(asc, des)

# ------ PERFORM DECOMPOSITION ---------------------------------------------------------------------------------------
up, azi = decompDriver(ascAlign, desAlign)

print(up)
