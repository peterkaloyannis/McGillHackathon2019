'''
This file will simulate galaxies based on different mass star distributions, optimally over time scales longer than a star lifetime. It is to be plugged into main.py.
PK, GP, AG, AB 2019
'''
import numpy as np
import matplotlib.pyplot as plt
from header import *

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import plotly.graph_objects as go

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Need to sample from gaussian to get initial distribution of points
# Generate random points in cylindrical to have the symmetry, then convert to rectangular

num_points = 100000
rdata = np.abs(np.random.normal(0, 1, num_points))
thetadata = np.random.uniform(0, 2 * np.pi, num_points)

zdata = np.empty(num_points)
count = 0
for r in rdata:
    zdata[count] = np.random.normal(0, -np.arctan(5*r - 4) + 3*np.pi / 4, 1) / 5 / np.pi

    count += 1
# Convert to rectangular
xdata = rdata * np.cos(thetadata)
ydata = rdata * np.sin(thetadata)

# Helix equation

fig = go.Figure(data=[go.Scatter3d(x=xdata, y=ydata, z=zdata,
                                   mode='markers')])

fig.update_layout(scene = dict(
        xaxis = dict(nticks=4, range=[-3, 3],),
                     yaxis = dict(nticks=4, range=[-3, 3],),
                     zaxis = dict(nticks=4, range=[-3, 3],),),
                     width=1680,
                     margin=dict(r=20, l=10, b=10, t=10))

fig.show()
