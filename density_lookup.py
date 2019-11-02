import numpy as np
import matplotlib.pyplot as plt

G = 6.67e-8
kappa = 6.65/1.67
c = 2.99e10

def NFW_potential(r,rho_0,r_s):
    return -1*(4*np.pi*rho_0*r_s**3/r)*np.log(1+r/r_s)



if __name__=="__main__":
    x = np.linspace(1,50001,50001)
    y = NFW_potential(x,1e-21,25000)
    plt.plot(x,y)
    plt.xlabel("Radius [ly]")
    plt.ylabel("Potential Energy [mks]")
    plt.show()