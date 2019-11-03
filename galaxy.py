'''
This file will simulate galaxies based on different mass star distributions, optimally over time scales longer than a star lifetime. It is to be plugged into main.py.
PK, GP, AG, AB 2019
'''
import numpy as np
import matplotlib.pyplot as plt
from header import *
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d
import plotly.graph_objects as go
import matplotlib.animation as animation
from mass import *

wframe= None
motion = np.empty((100, num_stars, 3))
# Need to sample from gaussian to get initial distribution of points
# Generate random points in cylindrical to have the symmetry, then convert to rectangular
def update(idx, galaxy_parameters, ax, gradient):
    global wframe

    # If a line collection is already remove it before drawing.
    if wframe:
        ax.collections.remove(wframe)

    galaxy_parameters[:, 0], galaxy_parameters[:, 1], galaxy_parameters[:, 4], galaxy_parameters[:, 5] \
        = leapfrog(idx, dt, galaxy_parameters[:, 0],
                    galaxy_parameters[:, 4], galaxy_parameters[:, 1],galaxy_parameters[:,5],gradient, galaxy_parameters[:,3])

    # Plot the new wireframe and pause briefly before continuing.
    wframe = ax.scatter(galaxy_parameters[:,0], galaxy_parameters[:,1], galaxy_parameters[:, 2],
                        s=2, c=-galaxy_parameters[:, 7], cmap='cool')
    
    motion[idx, :, :] = galaxy_parameters[:, :3]


def leapfrog(i, dt, x, vx, y, vy, gradient, mass):
    '''
     - i = index of loop
     - dt = size of timestep
     - r = current radius
     - vr = current radial velocity
     - theta = current angle
     - vtheta = current angular velocity
    '''
    r =  np.hypot(x,y)
    vx[r<rmax] += scale*interpolatelookup(gradient, r[r<rmax])/conversion*sectoyear**2*np.cos(np.arctan2(y[r<rmax],x[r<rmax]))
    x[r<rmax] += vx[r<rmax]*dt
    vy[r<rmax] += scale*interpolatelookup(gradient, r[r<rmax])/conversion*sectoyear**2*np.sin(np.arctan2(y[r<rmax],x[r<rmax]))
    y[r<rmax] += vy[r<rmax]*dt
    return x, y, vx, vy

def interpolatelookup(table, r, r_range= r_range, r_step= r_step):
    '''
    - table is the lookup table
    - r_range is the maximum radius
    - r_step is the radial step
    - r is the current radius
    '''
    # print(max(r))
    r_lower = np.floor(np.abs(r)) #bottom radius
    bottom_index = (r_lower*r_step).astype(np.int)
    top_index = bottom_index+1
    delta_r = np.abs(r)-bottom_index

    return (table[top_index]-table[bottom_index])/r_step*delta_r+table[bottom_index] 

def genlookup(r_range, r_step, func, values,name):
    '''
    - r_range is the maximum radius
    - r_step is the radial resolution of the lookup table
    - func is the funtion to generate a lookup for
    - values are the non radius inputs of the function
    - name is the filename of the lookup
    '''
    size = r_range/r_step
    r = np.arange(1, size)*r_step
    answers = func(r, *values)
    np.save(name, answers)

def NFW_potential(r,rho_0,r_s):
    '''
    - r is the radius
    - rho_0 is the mass density
    - r_s is the radius in which half the mass is inside
    '''
    return -1*(4*np.pi*rho_0*r_s**3/r/conversion)*np.log(1+r*conversion/r_s)


def gengrad(potential, bin_width):
    '''
    :param potential: lookup table in cylindrical coordinates of potential field
            index: In which the desired force is located, only radius matters since stars are in disk.
            bin_width: Distance between scalar values in points.
    :return: inward radial force based on the radius. Scalar.
    '''

    field = - (potential[1:] - potential[:-1]) / bin_width / conversion #METRIC N

    return field

def get_density(stars):

    density = np.empty(num_stars)
    count = 0
    for star in stars:

        for star2 in stars:
            if np.abs(star[0] - star2[0]) < 1000 and np.abs(star[1] - star2[1]) < 0.5:
                density[count] += 1

        count += 1

    return density

def generate_galaxy(num_stars, radius):
    """
    - num_stars is the number of stars in the galaxy
    - radius is the radius in which around two thirds of the stars lie (one sigma)
    returns the coordinates of each star, the mass, the velocity in (r, theta) coordinates
    """
    genlookup(1000000, r_step, NFW_potential, [rho_0,r_s], "potentials.npy")
    potential = np.load('potentials.npy')
    gradient = gengrad(potential, 1)

    stars = np.empty((num_stars, 6))
    # Work in cylindrical coordinates
    stars[:, 0] = np.abs(np.random.normal(0, radius, num_stars))  # Distance from center from gaussian
    r = np.copy(stars[:,0])

    stars[:, 1] = np.random.uniform(0, 2 * np.pi, num_stars)  # Uniform dist for angle
    angs = np.copy(stars[:,1])
    stars[:, 2] = np.random.normal(0, radius / 6 * np.exp(-(r/radius)**2), num_stars)  # Height of stars depends on r

    stars[:,0] = r*np.cos(angs)
    stars[:,1] = r*np.sin(angs)
    # Mass of stars
    stars[:, 3] = np.asarray(mass_generator(num_stars)) * 1.98e+30  # Masses in metric (conversion)

    # Velocities initialized with unit velocity in random directions
    #directions = np.random.normal(0, np.pi, )
    v = np.sqrt(r * conversion * -interpolatelookup(gradient, r)) / conversion * sectoyear
    stars[:, 4] = -stars[:,1]/r*v  # Velocity in x direction
    stars[:, 5] = stars[:,0]/r*v # Velocity in y direction
    #np.sqrt(stars[:, 0] * -interpolatelookup(gradient, stars[:, 0]))
    stars[:, 6] = np.log(luminosity(stars[:, 3]))
    stars[:, 7] = get_density(stars)
    return stars, gradient


def graph(rdata, thetadata, zdata):
    # Convert to rectangular
    xdata = rdata * np.cos(thetadata)
    ydata = rdata * np.sin(thetadata)

    fig = go.Figure(data=[go.Scatter3d(x=xdata, y=ydata, z=zdata,
                                       mode='markers',
                                       marker=dict(
                                           size=6,
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

if __name__ == '__main__':

    x = np.linspace(1,50001,50001)
    # y = NFW_potential(x,1e-21,25000)
    # plt.plot(x,y)
    # plt.xlabel("Radius [ly]")
    # plt.ylabel("Potential Energy [mks]")
    # plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # rdata, thetadata, zdata = generate_galaxy(num_stars, r_range)

    # graph(rdata, thetadata, zdata)
    genlookup(r_range, r_step, NFW_potential, [rho_0,r_s], "potentials.npy")
    potential = np.load('potentials.npy')
    print(interpolatelookup(potential, 10003.7))
    plt.plot(potential)
    plt.show()
