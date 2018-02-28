# -*- coding: utf-8 -*-
# =============================================================================
# 2mmn40 code
# Water molecule simulation
# version  2018-01-16
# BA
# 
# 
# for BA: Make sure to run in directory
# C:\Users\20165263\Dropbox\tue\2mmn40\src
# 
# =============================================================================


# Objective: simulate a water molecule. So we're just integrating f=ma over t.
# To integrate f=ma, we need f, m, v0 and q0.
# f is obtained from potentials. f = -grad u
# m, v0, x0 are all given.

import numpy as np

# Data structure required: molecule geometry. So, a list of lists of molecules.
# Each molecule needs to have a mass, an x0, a v0, and explicit geometry

### part 1: diatomic molecule

### Parameters
#molecule parameters
bondList = [[1,2],[0],[0]]
angleList = [[1,0,2]]

#this is only for the OH bond
kbond = 1.0
rbond = 1.0

# list of masses of atoms
m = np.array([16.0, 1.0, 1.0])

#ensemble parameters
n=len(m)

#simulation parameters: choice of integrator
# 0 - forward euler
# 1 - verlet
# 2 - velocity verlet

integrator = 2
maxsteps = 2000

#output filename
output = 'outputH2O.xyz'

#initial values
q0 = np.array([[0.0, 0.0, 0.0], 
               [1., 0.5, 0.0],
               [-1., 0.5, 0.0]])

v0 = np.zeros(shape=[n,3])


##END PARAMETERS

# find the bond force on a particle with index ids[0]
# as exerted by particle with index ids[1]
def findBondForces(r,dr,ids):
    pass

def findForces(qin):
    #find distance: r and dr
    dr = qin - qin[:, np.newaxis]
    r = np.linalg.norm(dr, axis=2)
    
    ftot = np.zeros([n,3])
    
    # find bond forces
    # triangular loop since forces are symmetric
    for i, bonds in enumerate(bondList):
        for j in bonds:
            if j>i:
                fij = -kbond * dr[i,j,:] * (rbond - r[i,j]) / r[i,j]
                ftot[i] += fij
                ftot[j] += fij
    
    #implement angles
    for angle in angleList:
        pass
        #UGHHH
    
    return ftot
    
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
            print('Water molecule. t = {:10.5f}'.format(i*dt), 
                  file=outfile)
            #create list of triples: first dimension is number of molecules
            molecules = snapshot.reshape([n//3,3,3])
            for molecule in molecules:
                #slightly hacky starred expression in format: unpacking
                print('O {:10.5f} {:10.5f} {:10.5f}'
                      .format(*molecule[0],), 
                      file=outfile)
                print('H {:10.5f} {:10.5f} {:10.5f}'
                      .format(*molecule[1],), 
                      file=outfile)
                print('H {:10.5f} {:10.5f} {:10.5f}'
                      .format(*molecule[2],), 
                      file=outfile)

#for running script as file:   
#run main
main()
