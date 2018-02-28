# -*- coding: utf-8 -*-
# =============================================================================
# 2mmn40 code
# Diatomic molecule simulation
# version  2018-01-15
# BA
# 
# 
# for BA: Make sure to run in directory
# C:\Users\20165263\Dropbox\tue\2mmn40\src
# 
# =============================================================================


# Objective: simulate a diatomic bond. So we're just integrating f=ma over t.
# To integrate f=ma, we need f, m, v0 and q0.
# f is obtained from potentials. f = -grad u
# m, v0, x0 are all given.

import numpy as np

#seed random number gen
np.random.seed(123456)

# Data structure required: molecule geometry. So, a list of lists of molecules.
# Each molecule needs to have a mass, an x0, a v0, and explicit geometry

### part 1: diatomic molecule

### Parameters
#molecule parameters
bondList = [[1],[0]]
kbond = 1.0
rbond = 1.0
m = np.array([1.0, 1.0])

#ensemble parameters
#we're only simulating one diatomic molecule here so let's have it fixed for now
n=2

#simulation parameters: choice of integrator
# 0 - forward euler
# 1 - verlet
# 2 - velocity verlet

integrator = 2
maxsteps = 2000

#output filename
output = 'output.xyz'

#initial values
q0 = np.array([[0.0, 0.1, 0.01], 
               [0.9, 0.05, -0.01]])
v0 = np.array([[0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0]])

#add small random velocity for system perturbation
v0 += np.random.normal(size=[n,3])

##END PARAMETERS



def findForces(q):
    #find distance: r and dr
    dr = q - q[:, np.newaxis]
    r = np.linalg.norm(dr, axis=2)
    
    # find bond forces
    # will throw a RuntimeWarning due to the dividing by zero along diagonal elements of r.
    # However, nan_to_num converts those nan's to zero in the result, so ignore the warning.
    # A particle cannot exert a force on itself, so that makes sense
    fbond = np.nan_to_num( -kbond * dr * (rbond - r[:,:,np.newaxis]) / r[:,:,np.newaxis])
    ftotal = np.sum(fbond,axis=1)
    return ftotal
    
def main():
    #initialize system state
    q = [q0]
    v = [v0]
    
    # take a small enough timestep
    dt = min(np.sqrt( kbond/m )) /100
    
    #integrator selection
    if integrator==0:
        #Forward Euler
        for i in range(maxsteps):
            f = findForces(q[i])
            #integration step
            q.append(q[i] + dt*v[i] + dt**2 /(2*m[:,np.newaxis]) *f)
            v.append(v[i] + dt/m[:,np.newaxis] *f)
            
    elif integrator==1:
        #Verlet
        # First step: make an Euler step on q
        # we do this because Verlet needs two q states in the past
        f=findForces(q[0])
        q.append(q[0] + dt*v[0] + dt**2 /(2*m[:,np.newaxis]) *f)
        
        for i in range(1,maxsteps):
            f = findForces(q[i])
            q.append(2*q[i] - q[i-1] + dt**2 / m[:, np.newaxis] *f)
            v.append((q[i+1]-q[i-1])/2)
            
    elif integrator==2:
        #Velocity Verlet
        f = findForces(q[0])
        for i in range(maxsteps):
            q.append(q[i] + dt*v[i] + dt**2 / (2*m[:, np.newaxis]) *f)
            # here the forces have to be known at time t and at time t+dt;
            # so we find the forces now, and use this in the next run
            fnext = findForces(q[i+1])
            v.append(v[i] + dt/(2*m[:, np.newaxis]) * (f + fnext))
            f = fnext
            
    else:
        raise Exception('Invalid integrator {}'.format(integrator))
    
    #export result to xyz
    with open(output,'w') as outfile:
        #enumerate iterates through a list, and lets use an index too
        for i, snapshot in enumerate(q):
            print('{}'.format(n), file=outfile)
            print('Single diatomic molecule. t = {:10.5f}'.format(i*dt), 
                  file=outfile)
            for molecule in snapshot:
                print('O {:10.5f} {:10.5f} {:10.5f}'
                      .format(molecule[0],molecule[1],molecule[2]), 
                      file=outfile)

#for running script as file:   
#run main
main()