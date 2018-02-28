# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 20:25:16 2018

@author: roman
"""

#%reset -f

###################################################################################
################################# IMPORTS #########################################
###################################################################################
import os
#os.chdir("C:/Users/roman/Documents/University/Eindhoven/Introduction to Molecular Modeling and Simulation/Python")
import numpy as np
#import matplotlib.pyplot as plt
from functionsExam import *
np.set_printoptions(suppress=True)



###################################################################################
################################### MAIN ##########################################
###################################################################################

# Set your integrator: 
#myIntegrator = "euler"
#myIntegrator = "verlet"
myIntegrator = "velocity verlet"

# Set time parameters:
dt = 2
totalTime = 200

# Set boxSize:
boxSize = 50 # in Angström

# Lennart Jones cut off length:
rLJCutOff = 8

# Get molecule structures
structure = getMoleculeStructure()

# Several parameters:
meanV = 0
stDevV = 0.05
rescale = 0 # in {0,1}: 1 means: do a rescale every time step. 0 means: no rescaling
targetTemperature = 300


#####

totalNumMolecules = 10**3#2**3 # Has to be a number to the power 3!!!!
percentageEthanol = 13.5#13.5 # In percentage in [0, 100]

XYZInitial, allBonds, allAngles, allMasses, allTypes, allMoleculeNumbers = initializeConfiguration(totalNumMolecules, percentageEthanol)
XYZ = XYZInitial
vInitial = initializeV(allMoleculeNumbers, meanV, stDevV)
v = vInitial

temperature = np.zeros(int(totalTime/dt+1))
temp, v = thermostat(v, allMasses, 1, targetTemperature)
temperature[0], v = thermostat(v, allMasses, rescale, targetTemperature)


totalNumAtoms = XYZ.shape[0]

# Start configuration xyz of system at time 0
currentTime = 0
forcesPeriodMin1 = []


if os.path.isfile("myTrajectory.xyz"):
    os.remove("myTrajectory.xyz")
file = open("myTrajectory.xyz","w")
file.write(str(totalNumAtoms)+ "\n")
file.write("Generated trajectory: SOL t=0\n")
file.write('\n'.join('         '.join(str(cell) for cell in row) for row in np.concatenate((allTypes[:, 0].reshape(allTypes.shape[0], 1), np.round(np.mod(XYZ, boxSize), 5).astype(str)), axis = 1)))
file.write("\n")



"""
for i in range(0, 100):
    file.write(str(totalNumAtoms)+ "\n")
    file.write("Generated trajectory: SOL t=" + str(i) + "\n")
    #file.write(str(np.round(XYZ, 5)))
    file.write('\n'.join('         '.join(str(cell) for cell in row) for row in np.concatenate((allTypes[:, 0].reshape(allTypes.shape[0], 1), np.round(XYZ+i, 5).astype(str)), axis = 1)))
    file.write("\n")
"""
potentialEnergy = np.zeros(int(totalTime/dt+1))
kineticEnergy = np.zeros(int(totalTime/dt+1))

forces, potentialEnergy[0] = calculateForces(XYZ, allBonds, allAngles, allTypes, allMoleculeNumbers, boxSize, rLJCutOff)
kineticEnergy[0] = sum(0.5*allMasses[:,0]*((np.linalg.norm(v, axis = 1))**2))

for currentTime in np.arange(dt, totalTime+dt, dt):
    print("At time " + str(currentTime) + " which is " + str(currentTime*100/totalTime) + " %")
    
    
    if currentTime != dt:
        XYZPeriodMin2 = XYZPeriodMin1
        vPeriodMin2 = vPeriodMin1
        #forcesPeriodMin2 = forcesPeriodMin1
        
    XYZPeriodMin1 = XYZ
    vPeriodMin1 = v
    forcesPeriodMin1 = forces
       
    
    #if myIntegrator != "velocity verlet" || currentTime == 0:
        
    
    # Integrators:
    
    # On units:
    
    # [U] = kJ / mol 
    # [F] = kJ / mol / Angström = 1000 N*m / mol / Angström = 1000 kg * m^2 / s^2 / mol / Angström
    # = 1000000 g * m^2 / (10^30 fs^2) / mol / Angström = 10^-24  g * m^2 / fs^2 / mol / Angström
    # = 10^-24  g * m^2 / fs^2 / mol / (10^-10 m) = 10^-14  g * m / fs^2 / mol 
    # = 10^-4  g * Angström / fs^2 / mol 
    # [t] = [dt]= fs 
    # Euler: q(t + dt) = q(t) + dt*v(t) + dt^2 * F /(2*m) 
    # [q] = Angström
    # [v] = Angström / fs
    # so F must be multiplied by 10^-4 and m must have unit: [m] = g/mol = amu
        
    # Euler:
    if myIntegrator == "euler":
        print("Euler")
        XYZ = XYZPeriodMin1 + dt*v + (dt**2)*(10**-4)*forcesPeriodMin1/(2*allMasses)
        v = vPeriodMin1 + dt*(10**-4)*forcesPeriodMin1/allMasses
        temperature[int(currentTime/dt)], v = thermostat(v, allMasses, rescale, targetTemperature)
        kineticEnergy[int(currentTime/dt)] = sum(0.5*allMasses[:,0]*((np.linalg.norm(v, axis = 1))**2))
        forces, potentialEnergy[int(currentTime/dt)] = calculateForces(XYZ, allBonds, allAngles, allTypes, allMoleculeNumbers, boxSize, rLJCutOff)
    
    # Verlet:
    if myIntegrator == "verlet":
        if currentTime == dt:
            # Forward Euler
            print("Euler")
            XYZ = XYZPeriodMin1 + dt*v + (dt**2)*(10**-4)*forcesPeriodMin1/(2*allMasses)
            #v = vPeriodMin1 + dt*(10**-4)*forcesPeriodMin1/allMasses
        else:
            print("Verlet")
            XYZ = 2*XYZPeriodMin1 - XYZPeriodMin2 + (dt**2)*(10**-4)*forcesPeriodMin1/(allMasses)
            vPeriodMin1 = (XYZ - XYZPeriodMin2)/(2*dt)
            temperature[int(currentTime/dt)], vPeriodMin1 = thermostat(vPeriodMin1, allMasses, rescale, targetTemperature)
            kineticEnergy[int(currentTime/dt)-1] = sum(0.5*allMasses[:,0]*((np.linalg.norm(vPeriodMin1, axis = 1))**2))
        forces, potentialEnergy[int(currentTime/dt)] = calculateForces(XYZ, allBonds, allAngles, allTypes, allMoleculeNumbers, boxSize, rLJCutOff)
            
    # Verlet velocity:
    if myIntegrator == "velocity verlet":
        print("Velocity Verlet")
        XYZ = XYZPeriodMin1 + dt*vPeriodMin1 + (dt**2)*(10**-4)*forcesPeriodMin1/(2*allMasses)
        forces, potentialEnergy[int(currentTime/dt)] = calculateForces(XYZ, allBonds, allAngles, allTypes, allMoleculeNumbers, boxSize, rLJCutOff)
        v = vPeriodMin1 + dt*(10**-4)*(forcesPeriodMin1 + forces)/(2*allMasses)
        temperature[int(currentTime/dt)], v = thermostat(v, allMasses, rescale, targetTemperature)
        kineticEnergy[int(currentTime/dt)] = sum(0.5*allMasses[:,0]*((np.linalg.norm(v, axis = 1))**2))
        #vPeriodMin1 = vPeriodMin2 + dt*(forcesPeriodMin2 + forcesPeriodMin1)/(2*allMasses)
            
    
    
    file.write(str(totalNumAtoms)+ "\n")
    file.write("Generated trajectory: SOL t=" + str(currentTime) + "\n")
    file.write('\n'.join('         '.join(str(cell) for cell in row) for row in np.concatenate((allTypes[:, 0].reshape(allTypes.shape[0], 1), np.round(np.mod(XYZ, boxSize), 5).astype(str)), axis = 1)))
    file.write("\n")



kineticEnergy = (10**7)*kineticEnergy # in J/mol
potentialEnergy = (10**3)*potentialEnergy # in J/mol 
totalEnergy = potentialEnergy + kineticEnergy # in J/mol

import matplotlib.pyplot as plt
plt.figure(figsize=(20, 10))
plt.plot(np.arange(0, len(totalEnergy)), totalEnergy, 'b-', np.arange(0, len(totalEnergy)), potentialEnergy, 'r-', np.arange(0, len(totalEnergy)), kineticEnergy, 'g-')
plt.axis([0, len(totalEnergy), 0, max(totalEnergy)])
plt.legend(["Total energy", "Potential energy", "Kinetic energy"])


#plt.plot(temperature[1:])

file.close()



