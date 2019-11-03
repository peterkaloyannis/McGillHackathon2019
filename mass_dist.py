import numpy as np
import matplotlib.pyplot as plt
import math

'''
The purpose of this file is the function called mass_generator(). It takes as input an integer,
and will output that integer number of masses, generated as per the mass distribution of the milky way
galaxy.
'''

G = 6.67e-8 #gravitational Constant
kappa = 6.65/1.67 #thomspon scattering/mp (cgs)
c = 2.99e10 #speed of light (cgs)

def eddington(m,tcmb,rho):
    return 4*np.pi*G*m*c/kappa + 4*np.pi*(3*m/(4*np.pi*rho))**(2/3)*tcmb**4

def lamp_adjust_lum_list(m, tcmb, rho):
    return 4*np.pi*(3*m/(4*np.pi*rho))**(2/3)*tcmb**4

def find_nearest(cum_array,mass_array,value):
    idx = np.searchsorted(cum_array, value, side="left")
    if idx > 0 and (idx == len(cum_array) or math.fabs(value - cum_array[idx-1]) < math.fabs(value - cum_array[idx])):
        return mass_array[idx-1]
    else:
        return mass_array[idx]

"""
x = np.exp(np.linspace(-2.66,5,10000))
print(x)
# x = np.linspace(0.07,200)
y = mass_dist(x)
y = y/np.sum(y)
z = cumulative_dist(y)

print(find_nearest(z,x,0.98))

# plt.plot(np.log(x),np.log(y))
# plt.plot(np.log(x),z)
plt.plot(x,y)
plt.show()
"""

def mass_generator(length):
    random_array = np.random.rand(length)
    mass_list = []
    for i in range(length):
        if random_array[i] < 0.41:
            mass_list.append(np.random.uniform(0.07,0.25))
        elif 0.41 <= random_array[i] < 0.69:
            mass_list.append(np.random.uniform(0.25,0.5))
        elif 0.69 <= random_array[i] < 0.88:
            mass_list.append(np.random.uniform(0.5,1))
        elif 0.88 <= random_array[i] < 0.96:
            mass_list.append(np.random.uniform(1,2))
        elif 0.96 <= random_array[i] < 0.99:
            mass_list.append(np.random.uniform(2,4))
        else:
            mass_list.append(np.random.uniform(4,69))
    return mass_list

def luminosity(mass_list):
    lum_list = []
    for m in mass_list:
        if m<0.43:
            lum_list.append(0.23*m**(2.3))
        elif 0.43<=m<2:
            lum_list.append(m**4)
        elif 2<=m<55:
            lum_list.append(1.4*m**3.5)
        else:
            lum_list.append(32000*m)
    return lum_list

def color(lum_list):
    color_list = []
    for lum in lum_list:
        if lum<0.08:
            # color_list.append("M")
            color_list.append("red")
        elif 0.08<=lum<0.6:
            # color_list.append("K")
            color_list.append("orange")
        elif 0.6<=lum<1.5:
            # color_list.append("G")
            color_list.append("yellow")
        elif 1.5<=lum<5:
            # color_list.append("F")
            color_list.append("yellow")
        elif 5<=lum<25:
            # color_list.append("A")
            color_list.append("white")
        elif 25<=lum<30000:
            # color_list.append("B")
            color_list.append("blue")
        else:
            # color_list.append("O")
            color_list.append("blue")
    return color_list

def radius(mass_list):
    radii_list = []
    for m in mass_list:
        radii_list.append(np.sqrt(m))
        # if m<0.25:
            # radii_list.append(np.random.uniform(0.08,0.3))
        # elif 0.25<=m<0.5:
            # radii_list.append(np.random.uniform(0.3,0.5))
        # elif 0.5<=m<1:
            # radii_list.append(np.random.uniform(0.5,1))
        # elif 1<=m<2:
            # radii_list.append(np.random.uniform(1,1.8))
        # elif 2<=m<4:
            # radii_list.append(np.random.uniform(1.8,3))
        # else:
            # radii_list.append(np.sqrt(m))
    return radii_list

def temperature(lum,rad):
    rad = rad*6.9e8
    lum = lum*3.8e26
    return (lum/(4*5.7e-8*np.pi*rad**2))**(1/4)

masses = mass_generator(10000)
luminosities = luminosity(masses)
lamp_adjust_lums = lamp_adjust_lum_list(np.array([masses]), 1e10, 5)/3.843e33
print(type(lamp_adjust_lums))
print(np.shape(lamp_adjust_lums[0]))
print(luminosities+lamp_adjust_lums[0])
radii = radius(masses)
new_lums = luminosities + lamp_adjust_lums[0]


colors = color(luminosities)
adjust_colors = color(new_lums)
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.set_facecolor('xkcd:black')
ax2.set_facecolor('xkcd:black')
ax1.scatter(masses, np.log(luminosities), s=masses*100, color=colors)
ax2.scatter(masses, np.log(new_lums), s=masses*100, color=adjust_colors)
# temperatures = temperature(np.array([luminosities]),np.array([radii]))
# print(temperatures)
# plt.plot(temperatures[::-1], np.log(np.array([luminosities])),'ok')
plt.show()
