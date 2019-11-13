import matplotlib.pyplot as plt
import numpy as np
import math
import readGRD
import seismoPlots
import matplotlib.cm as cm


def getPoint(xdata, ydata, zdata, colors, track, boxWidth, boxHeight):


    region = []

    def select_coordinates(event):

        point = [int(event.xdata), int(event.ydata)]
        region.append(int(point[0] - boxWidth/2))
        region.append(int(point[0] + boxWidth/2))
        region.append(int(point[1] - boxHeight/2))
        region.append(int(point[1] + boxHeight/2))
        

        print()
        print('Centroid: ' + str(point))
        print('Min. x: ' + str(region[0]))
        print('Max. x: ' + str(region[1]))
        print('Min. y: ' + str(region[2]))
        print('Max. y: ' + str(region[3]))
        print()

        plt.close()

        return region

    fig = plt.figure(figsize=(15, 20))

    # Plot deformation map with selected region/pixels overlain
    # Use a SEPARATE method from normak 'map' function in order to get grid x & y indicies, not values

    if track == 'asc': 
        # print("You haven't written ascending data plot method yet")

        extent = [0, len(xdata), 0, len(ydata)]

        # Make plot
        ax = plt.subplot(111)
        im = ax.imshow(zdata, extent=extent, cmap=colors, aspect=1.1/1.085)
        ax.invert_yaxis()
        cbar = plt.colorbar(im)
        cbar.set_label('LOS change (m)')


    elif track == 'des':

        extent = [0, len(xdata), 0, len(ydata)]

        # Make plot
        ax = plt.subplot(111)
        im = ax.imshow(zdata, extent=extent, cmap=colors, aspect=1.1/1.085)
        cbar = plt.colorbar(im)
        cbar.set_label('LOS change (m)')

    else:
        print(track + " is not a valid track. Must be 'asc' for acsending data or 'des' for descending data")

    fig.canvas.mpl_connect('button_press_event', select_coordinates)
    plt.show()


    boxIndex = region
    boxCoords = [np.round(float(xdata[region[0]]), 4), 
                   np.round(float(xdata[region[1]]), 4), 
                   np.round(float(ydata[region[2]]), 4), 
                   np.round(float(ydata[region[3]]), 4) ]

    print('Box coordinates:  [' + str(boxCoords[0]) + ' ' + str(boxCoords[1]) + ' ' + str(boxCoords[2]) + ' ' + str(boxCoords[3]) + ']') 

    return boxIndex, boxCoords


def getSwath(xdata, ydata, zdata, colors, track):

    points = []
    region = []

    def select_coordinates(event):

        points.append([int(event.xdata), int(event.ydata)]) 
        print(points[-1])
        
        if len(points) == 2:
            region.append(min([points[0][0], points[1][0]]))
            region.append(max([points[0][0], points[1][0]]))
            region.append(min([points[0][1], points[1][1]]))
            region.append(max([points[0][1], points[1][1]]))
            plt.close()

        return region

    fig = plt.figure(figsize=(15, 20))

    # Plot deformation map with selected region/pixels overlain
    # Use a SEPARATE method from normak 'map' function in order to get grid x & y indicies, not values

    if track == 'asc': 
        print("You haven't written ascending data plot method yet")

    elif track == 'des':

        extent = [0, len(xdata), 0, len(ydata)]

        # Make plot
        ax = plt.subplot(111)
        im = ax.imshow(zdata, extent=extent, cmap=colors, aspect=1.1/1.085)
        cbar = plt.colorbar(im)
        cbar.set_label('LOS change (m)')

    else:
        print(track + " is not a valid track. Must be 'asc' for acsending data or 'des' for descending data")

    fig.canvas.mpl_connect('button_press_event', select_coordinates)
    plt.show()


    swathIndex = region
    swathCoords = [np.round(float(xdata[region[0]]), 4), 
                   np.round(float(xdata[region[1]]), 4), 
                   np.round(float(ydata[region[2]]), 4), 
                   np.round(float(ydata[region[3]]), 4) ]

    print('Swath coordinates:  [' + str(swathCoords[0]) + ' ' + str(swathCoords[1]) + ' ' + str(swathCoords[2]) + ' ' + str(swathCoords[3]) + ']') 

    return swathIndex, swathCoords


# ------------------------- PLOTTING -------------------------

def swath(xdata, ydata, zdata, region, colors, subFigID):
    # INPUTS:
    # data - GMTSAR LOS grid file. Can be in radar or lat/lon coordinates
    # region - Xmin/Xmax/Ymin/Ymax of swath

    # IMPORTANT:
    # We DON'T want to apply any figure settings here; we want to treat this method akin to 
    # standard plotting routines (i.e. plt.scatter) in order to be able to apply multiple 
    # custom routines to the same subfigure
    
    range_change = []
    swath_x = []
    swath_y = []

    displacements = zdata

    # Calculate mean swath displacements for East-West trending swath
    if len(range(region[0], region[1])) > len(range(region[2], region[3])):
        for i in range(region[0], region[1]):
            total = 0
            numPix = 0
            
            for row in displacements[region[2]:region[3]]:

                if math.isnan(row[i]) != True:
                    total += row[i]
                    numPix += 1

            if numPix != 0:     
                swath_x.append(xdata[i])
                swath_y.append(total/numPix)

        plt.scatter(swath_x, swath_y, marker='.', zorder=100)


    # Or, calculate mean swath displacements for North-South trending swath
    else:
        for i in range(region[2], region[3]):
            total = 0
            numPix = 0
            for pixel in displacements[i][region[0]:region[1]]:

                if math.isnan(pixel) != True:
                    total += pixel
                    numPix += 1

            if numPix != 0:     
                swath_x.append(ydata[i])
                swath_y.append(total/numPix)

        plt.scatter(swath_x, swath_y, marker='.', zorder=100)


def map(xdata, ydata, zdata, colormap, vlim, track, CRS, subFigAx):
    # IMPORTANT:
    # We DON'T want to apply any figure settings here; we want to treat this method akin to 
    # standard plotting routines (i.e. plt.scatter) in order to be able to apply multiple 
    # custom routines to the same subfigure
    
    if track == 'asc': 
        # print("You haven't written ascending data plot method yet")

        if CRS == 'll':
            # Create nan mask
            masked_array = np.ma.array(zdata, mask=np.isnan(zdata))

            cmap = cm.jet
            cmap.set_bad('white', 1.)

            # Determine decimation increment for lat and lon axes
            dx = (xdata[1] - xdata[0])/2
            dy = (ydata[1] - ydata[0])/2
            extent = [xdata[0]-dx, xdata[-1]+dx, ydata[0]-dy, ydata[-1]+dy]
            im = subFigAx.imshow(masked_array, extent=extent, cmap=colormap, aspect=1.15, vmin=vlim[0], vmax=vlim[1], zorder=0)


        elif CRS == 'ra':
            cmap = cm.jet

            # For radar coordinates
            extent = [xdata[0], xdata[-1], ydata[0], ydata[-1]]
            im = subFigAx.imshow(zdata, cmap=colormap, aspect=1, vmin=vlim[0], vmax=vlim[1])
            subFigAx.invert_yaxis()

        elif CRS == 'orig':

            extent = [int(min(xdata)), int(max(xdata)), int(min(ydata)), int(max(ydata))]

            if len(vlim) == 2:
                im = subFigAx.imshow(zdata, extent=extent, cmap=colormap, aspect=3.3, vmin=vlim[0], vmax=vlim[1])
            else:
                im = subFigAx.imshow(zdata, extent=extent, cmap=colormap, aspect=3.3)
            subFigAx.invert_yaxis()
            
        return im

    elif track == 'des':

        if CRS == 'll':
            # Create nan mask
            masked_array = np.ma.array(zdata, mask=np.isnan(zdata))

            cmap = cm.jet
            cmap.set_bad('white', 1.)

            # Determine decimation increment for lat and lon axes
            dx = (xdata[1] - xdata[0])/2
            dy = (ydata[1] - ydata[0])/2
            extent = [xdata[0]-dx, xdata[-1]+dx, ydata[0]-dy, ydata[-1]+dy]
            im = subFigAx.imshow(masked_array, extent=extent, cmap=colormap, aspect=1.15, vmin=vlim[0], vmax=vlim[1], zorder=0)


        elif CRS == 'ra':
            cmap = cm.jet

            # For radar coordinates
            extent = [xdata[0], xdata[-1], ydata[0], ydata[-1]]
            im = subFigAx.imshow(zdata, cmap=colormap, aspect=1, vmin=vlim[0], vmax=vlim[1])
            subFigAx.invert_xaxis()

        elif CRS == 'orig':

            extent = [int(min(xdata)), int(max(xdata)), int(max(ydata)), int(min(ydata))]

            if len(vlim) == 2:
                im = subFigAx.imshow(zdata, extent=extent, cmap=colormap, aspect=3.3, vmin=vlim[0], vmax=vlim[1])
            else:
                im = subFigAx.imshow(zdata, extent=extent, cmap=colormap, aspect=3.3)

            subFigAx.invert_xaxis()

        return im

    else:
        print(track + " is not a valid track. Must be 'asc' for acsending data or 'des' for descending data")


if __name__ == '__main__':
    driver()


""" OLD
def swath(xdata, ydata, zdata, region, colors, subFigID):
    # INPUTS:
    # data - GMTSAR LOS grid file. Can be in radar or lat/lon coordinates
    # region - Xmin/Xmax/Ymin/Ymax of swath

    range_change = []
    swath_x = []
    swath_y = []

    displacements = zdata

    # Calculate mean swath displacements for East-West trending swath
    if len(range(region[0], region[1])) > len(range(region[2], region[3])):
        for i in range(region[0], region[1]):
            total = 0
            numPix = 0
            
            for row in displacements[region[2]:region[3]]:

                if math.isnan(row[i]) != True:
                    total += row[i]
                    numPix += 1

            if numPix != 0:     
                swath_x.append(xdata[i])
                swath_y.append(total/numPix)

        ax = plt.subplot(subFigID)
        ax.grid()
        ax.scatter(swath_x, swath_y, marker='.', zorder=100)
        plt.xlabel('Longitude')
        plt.ylabel('LOS range change (m)', rotation=0, labelpad=58)

    # Or, calculate mean swath displacements for North-South trending swath
    else:
        for i in range(region[2], region[3]):
            total = 0
            numPix = 0
            for pixel in displacements[i][region[0]:region[1]]:

                if math.isnan(pixel) != True:
                    total += pixel
                    numPix += 1

            if numPix != 0:     
                swath_x.append(ydata[i])
                swath_y.append(total/numPix)

        ax = plt.subplot(subFigID)
        ax.grid()
        ax.invert_xaxis()
        ax.scatter(swath_x, swath_y, marker='.', zorder=100)
        plt.xlabel('Latitude')
        plt.ylabel('LOS range change (m)', rotation=0, labelpad=58)
    # if limits != []:
"""