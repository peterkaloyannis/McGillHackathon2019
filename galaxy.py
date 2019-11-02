'''
This file will simulate galaxies based on different mass star distributions, optimally over time scales longer than a star lifetime. It is to be plugged into main.py.
PK, GP, AG, AB 2019
'''
import numpy as np
import matplotlib.pyplot as plt
from header import *
from mpl_toolkits import mplot3d
import plotly.graph_objects as go

# Need to sample from gaussian to get initial distribution of points
# Generate random points in cylindrical to have the symmetry, then convert to rectangular

def NFW_potential(r,rho_0,r_s):
    '''
    - r is the radius
    - rho_0 is the mass density
    - r_s is the radius in which half the mass is inside
    '''
    return -1*(4*np.pi*rho_0*r_s**3/r)*np.log(1+r/r_s)

def generate_galaxy(num_stars, r):
    '''
    -num_stars is the number of stars
    -r is the radius of the galaxy
    '''
    rdata = np.abs(np.random.normal(0, r, num_stars))
    thetadata = np.random.uniform(0, 2 * np.pi, num_stars)

    zdata = np.random.normal(0, -np.arctan(5*rdata - 4) + 3*np.pi / 4, num_stars) / 5 / np.pi

    return rdata, thetadata, zdata


def graph(rdata, thetadata, zdata):
    '''
    - rdata is the radial positions
    - thetadata is the angular positions
    - zdata is the height positions
    '''
    # Convert to rectangular
    xdata = rdata * np.cos(thetadata)
    ydata = rdata * np.sin(thetadata)

    fig = go.Figure(data=[go.Scatter3d(x=xdata, y=ydata, z=zdata,
                                       mode='markers',
                                       marker = dict(
                                           size = 6,
                                           color = zdata,
                                           colorscale = 'viridis',
                                           opacity = 0.8
                                       )
                                       )])

    fig.update_layout(scene = dict(
            xaxis = dict(nticks=4, range=[-3, 3],),
                         yaxis = dict(nticks=4, range=[-3, 3],),
                         zaxis = dict(nticks=4, range=[-3, 3],),),
                         width=1680,
                         margin=dict(r=20, l=10, b=10, t=10))

    fig.show()

if __name__ == '__main__':

    # x = np.linspace(1,50001,50001)
    # y = NFW_potential(x,1e-21,25000)
    # plt.plot(x,y)
    # plt.xlabel("Radius [ly]")
    # plt.ylabel("Potential Energy [mks]")
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    rdata, thetadata, zdata = generate_galaxy(num_stars, r_range)

    graph(rdata, thetadata, zdata)
