'''
This file will contain a bunch of default settings and universal constants used by the code but not available from numpy or scipy to be used in all code
'''

r_range = 50000 #maximum radius ly
r_step = 1 #radial step
rmax = 9e5
dt = 1e5  #time step in years
num_stars = 2000 #number of stars

conversion = 9.4607308e15
sectoyear = 31556952

rho_0 = 1e-31 #kg/m^3
r_s = 60000 * 9.4607308e15  # metric

G = 6.67e-8 #graviational Constant
kappa = 6.65/1.67 #thomspon scattering/mp (cgs)
c = 2.99e10 #speed of light (cgs)

fps = 40 #framerate
scale = 1e4
# wframe = None
