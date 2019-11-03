This project aims to look at how different CMB temperatures would effect the brightness of stars of different masses. The final goal is to create a galaxy simulator that can be observed through different filters as a function of time.

Packages required:
 - MatplotLib
 - Numpy
 - Plotly
 - Math

The breakdown of the python files is as follows:
 - main.py contains code for doing the entire simulation. Running it will execute the plotting of a randomly generated galaxy over time. 
 - header.py contains all hyper parameters used and a brief description of them.
 - galaxy.py contains all the functions responsible for creating the galaxy, rotating it with time, and changing the colorscheme of the plot to prioritize the density of regions or height or velocities or luminosities. 
 - starsim.py, mass.py are computations of the flux and mass distributions based on the external heat lamp
 
 Optimally, the code is run by excecuting "python main.py" in the repository directory.
