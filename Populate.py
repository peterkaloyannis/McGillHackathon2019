import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.animation as animation

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Need to sample from gaussian to get initial distribution of points
# Generate random points in cylindrical to have the symmetry, then convert to rectangular


def leapfrog(i, dt, r, vr, theta, vtheta):
    '''
     - i = index of loop
     - dt = size of timestep
     - r = current radius
     - vr = current radial velocity
     - theta = current angle
     - vtheta = current angular velocity
    '''
    rNew = 0
    thetaNew =0
    vrNew = 0
    vthetaNew = 0

    if (i%2!=0): #Updates vr and vtheta for odd iterations of loop
        vrNew += 0*dt
        rNew = r + vrNew*dt
        vthetaNew += 0*dt
        thetaNew = theta + vthetaNew*dt
    else: #Does not update vr and vtheta for even iterations of loop
        rNew = r + vr*dt
        thetaNew = theta + vtheta*dt
    return rNew, thetaNew


def generate_galaxy(num_stars, radius):
    """
    - num_stars is the number of stars in the galaxy
    - radius is the radius in which around two thirds of the stars lie (one sigma)
    returns the coordinates of each star, the mass, the velocity in (r, theta) coordinates
    """
    stars = np.empty((num_stars, 6))
    # Work in cylindrical coordinates
    stars[:, 0] = np.abs(np.random.normal(0, radius, num_stars))  # Distance from center from gaussian
    stars[:, 1] = np.random.uniform(0, 2 * np.pi, num_stars)  # Uniform dist for angle
    stars[:, 2] = np.random.normal(0, radius / 6 * np.exp(-(stars[:, 0]/radius)**2), num_stars)  # Height of stars depends on r

    # Mass of stars
    stars[:, 3] = np.full(num_stars, 1)  # TODO: add the mass of stars to be sampled from a distribution

    # Velocities initialized with unit velocity in random directions
    #directions = np.random.normal(0, np.pi, )
    stars[:, 4] = np.random.normal(0, 1, num_stars)  # Velocity in radial direction
    stars[:, 5] = 1.14e-5 * stars[:, 0]**(1/3)  # Velocity in theta direction

    return stars


radius = 5000

galaxy = generate_galaxy(200, radius)

def graph(rdata, thetadata, zdata):
    # Convert to rectangular
    xdata = rdata * np.cos(thetadata)
    ydata = rdata * np.sin(thetadata)

    fig = go.Figure(data=[go.Scatter3d(x=xdata, y=ydata, z=zdata,
                                       mode='markers',
                                       marker=dict(
                                           size=2,
                                           color=zdata,  # set color to an array/list of desired values
                                           colorscale='haline',  # choose a colorscale
                                           opacity=0.8
                                       )
                                       )])

    fig.update_layout(scene = dict(
            xaxis = dict(nticks=4, range=[-3*radius, 3*radius],),
                         yaxis = dict(nticks=4, range=[-3*radius, 3*radius],),
                         zaxis = dict(nticks=4, range=[-3*radius, 3*radius],),),
                         width=1680,
                         margin=dict(r=20, l=10, b=10, t=10))

    fig.show()


#graph(galaxy[:, 0], galaxy[:, 1], galaxy[:, 2])

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Set the z axis limits so they aren't recalculated each frame.
ax.set_zlim(-3*radius, 3*radius)
ax.set_xlim(-3*radius, 3*radius)
ax.set_ylim(-3*radius, 3*radius)

# Begin plotting.
wframe = None

fps = 10

def update(idx):

    global wframe
    # If a line collection is already remove it before drawing.
    if wframe:
        ax.collections.remove(wframe)

    galaxy[:, 0], galaxy[:, 1] = leapfrog(idx, 0.1, galaxy[:, 0], galaxy[:, 5], galaxy[:, 1], galaxy[:, 4])

    # Plot the new wireframe and pause briefly before continuing.
    wframe = ax.scatter(galaxy[:, 0] * np.cos(galaxy[:, 1]), galaxy[:, 0] * np.sin(galaxy[:, 1]), galaxy[:, 2], c = galaxy[:, 0], cmap='viridis')


ani = animation.FuncAnimation(fig, update, 100, interval=1000/fps)

fn = 'gacjasasld'
ani.save(fn+'.mp4',writer='ffmpeg',fps=fps)

import subprocess
cmd = 'magick convert %s.gif -fuzz 5%% -layers Optimize %s_r.gif'%(fn,fn)
subprocess.check_output(cmd, shell=True)

plt.rcParams['animation.html'] = 'html5'

