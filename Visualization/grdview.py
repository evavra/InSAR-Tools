#!/bin/python
import sys

def main():
    """
    Plot grd file
    
    Usage: python grdview.py grd_file cmap

    grd_file  - path to grd file
    cmap - Matplotlib color map to use
    """

    if len(sys.argv) < 2:
        print(main.__doc__)
        sys.exit()

    # Perform hefty imports after making sure you need to 
    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr

    # Assign args
    grd_file = sys.argv[1]
    cmap = sys.argv[2]


    # vmin = float(sys.argv[3])
    # vmax = float(sys.argv[4])

    with xr.open_dataset(grd_file) as file:
        grd = np.array(file['z'])

    fig, ax = plt.subplots()

    if 'flip_0' in sys.argv:
        grd = np.flip(grd, 0)

    if 'flip_1' in sys.argv:
        grd = np.flip(grd, 1)

    ax.imshow(grd, cmap=cmap)
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.show()

if __name__ == '__main__':
    main()






 # Activate conda environment with PyGMT
# import subprocess
# import pygmt
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import xarray as xr
# import sys




# # Plot figure
# fig = pygmt.Figure()
# # grid = pygmt.datasets.load_earth_relief(resolution="03s", region=[-116.25, -115.25, 32.75, 34.])
# fig.grdimage(grid=grid, projection="M10c", frame="a", cmap='gray', shading=True)#'+a-45+nt3+m.3')
# # fig.basemap(region=[-117.5, -114.5, 32, 34.5], projection="U11S/10c", frame="a")
# fig.coast(resolution='f', water="skyblue")
# fig.colorbar(frame=["a1000", "x+lElevation", "y+lm"])
# fig.show()
# # fig.savefig('test.png')