# -*- coding: utf-8 -*-
# =============================================================================
# 2mmn40 week 3 report
# version 2017-12-03 afternoon 
# BA
# 
# 
# for BA: Make sure to run in directory
# C:\Users\20165263\Dropbox\tue\2mmn40\src
# 
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

# Objective: simulate a diatomic bond. So we're just integrating f=ma over t.
# To integrate f=ma, we need f, m, v0 and q0.
# f is obtained from potentials. f = -grad u
# m, v0, x0 are all given.

# Data structure required: molecule geometry. So, a list of lists of molecules.
# Each molecule needs to have a mass, an x0, a v0, and explicitly 

### part 1: diatomic molecule



#molecule parameters
bondList = [[1],[0]]
kbond = 1.0
rbond = 1.0
m = np.array([1.0, 1.0])

#simulation parameters: choice of integrator
# 0 - forward euler
# 1 - verlet
# 2 - velocity verlet

integrator = 0
maxsteps = 1000

# take a small enough timestep
dt = min(np.sqrt( kbond/m )) /100

#initial values
q0 = np.array([[0.0, 0.1, -0.1], 
               [1.01, 0.9, 0.95]])
v0 = np.array([[0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0]])

#initialize system state
q = q0.copy()
v = v0.copy()


#find distance: r and dr
dr = q - q[:, np.newaxis]
r = np.linalg.norm(dr, axis=2)

# find bond forces
# will throw a RuntimeWarning due to the dividing by zero along diagonal elements of r.
# However, nan_to_num converts those nan's to zero in the result, so ignore the warning.
# A particle cannot exert a force on itself, so that makes sense
fbond = np.nan_to_num( -kbond * dr * (rbond - r[:,:,np.newaxis]) / r[:,:,np.newaxis])
ftotal = np.sum(fbond,axis=1)

#integrate a single step:
if integrator == 0:
    q += dt*v + dt**2 /(2*m[:,np.newaxis]) *ftotal
    v += dt/m[:,np.newaxis] *ftotal
elif integrator == 1:
    #Verlet integration step
    q += 0
elif integrator == 2:
    #Velocity Verlect integration step
    q += 0
else:
    raise ('Unkown integrator selected')
    
