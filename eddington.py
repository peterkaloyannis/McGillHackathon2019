import numpy as np

G = 6.67e-8
kappa = 6.65/1.67
c = 2.99e10

def eddington(m,tcmb,rho):
    return 4*np.pi*G*m*c/kappa + 4*np.pi*(3*m/(4*np.pi*rho))**(2/3)*tcmb**4