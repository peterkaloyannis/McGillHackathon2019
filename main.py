'''
This File will be executed to combine the galaxy simulation with the star simulation in interesting ways.
PK, AB, GP, AG 2019
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d
import plotly.graph_objects as go
import matplotlib.animation as animation
from galaxy import *
from starsim import *
from header import *

if __name__=='__main__':
    # wframe = None
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    galaxy = generate_galaxy(num_stars,r_range) #creating galaxy

    ani = animation.FuncAnimation(fig, update, 100, interval=1000/fps)#, fargs = (galaxy,ax,))
    print('ass')
    fn = 'gacjasasld'
    ani.save(fn+'.mp4',fps=fps)

