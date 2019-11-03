'''
This File will be used to simulate the stars based on the flux coming in. This will be plugged into main.py
PK, GP, AB, AG 2019
'''
import numpy as np
import matplotlib.pyplot as plt
from header import *

def eddington(m,tcmb,rho):
    return 4*np.pi*G*m*c/kappa + 4*np.pi*(3*m/(4*np.pi*rho))**(2/3)*tcmb**4