from readGRD import readInSAR
import matplotlib.pyplot as plt
import numpy as np
import insarPlots
from mpl_toolkits.axes_grid1 import ImageGrid
    
# -- INPUT --------------------------------------------------------------------------------------
# Files
fileDir = "/Users/ellisvavra/Thesis/insar/des/f2/intf_all/SBAS_SMOOTH_0.0000e+00/"
fileType = "LOS_20191001_INT3.grd"
outDir = '/Users/ellisvavra/Thesis/insar/des/f2/intf_all/SBAS_SMOOTH_0.0000e+00'
outputName = "temp.eps"
save = 'no'

# Figure settings
num_plots_x = 10
num_plots_y = 10
colors = 'jet'
track = 'asc'

CRS = 'orig'
vlim = [-0.05, 0.05]

[xdata, ydata, zdata] = readInSAR(fileDir + fileType)

fig = plt.figure(figsize=(10,10))

grid = ImageGrid(fig, 111,
                nrows_ncols=(1, 1),
                axes_pad=0.25,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.2,
                cbar_size='5%'
                )

im = insarPlots.map(xdata, ydata, zdata, colors, vlim, track, CRS, grid[0])
cbar = grid[0].cax.colorbar(im)
cbar.ax.set_yticks(np.arange(-0.05, 0.051, 0.01))
cbar.ax.tick_params(labelsize=20)
cbar.ax.set_label('LOS displacement (m)')
grid[0].scatter(200, 1200, 100)
plt.show()