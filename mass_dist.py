import numpy as np
import matplotlib.pyplot as plt
import math

'''
The purpose of this file is the function called mass_generator(). It takes as input an integer,
and will output that integer number of masses, generated as per the mass distribution of the milky way
galaxy.
'''

def mass_dist(mass_array):
    # const = 1**(-1.3)
    return_array = np.zeros(len(mass_array))
    i = 0
    for m in mass_array:
        if m>1:
            return_array[i] = m**(-0.55)
            i+=1
        else:
            return_array[i] = m**(-0.55)
            i+=1
    return return_array

def cumulative_dist(return_array):
    cumulative_dist = np.zeros(len(return_array))
    for i in range(len(return_array)):
        cumulative_dist[i] = np.sum(return_array[0:i])
    return cumulative_dist

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
            mass_list.append(np.random.uniform(4,100))

    return mass_list

masses = mass_generator(10000)
plt.hist(masses,100)
plt.show()