# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 20:25:16 2018

@author: roman
"""

%reset -f

###################################################################################
################################# IMPORTS #########################################
###################################################################################
import os
os.chdir("C:/Users/roman/Documents/University/Eindhoven/Introduction to Molecular Modeling and Simulation/Python")
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

# Set timestep:
dt = 2

# Get molecule structures
structure = getMoleculeStructure()

# Several parameters:
numTimeSteps = 10


# Water parameters:
numMoleculesWater = 10**3
numAtomsPerMoleculeWater = countIn("atoms", "Water")
numBondsPerMoleculeWater = countIn("bonds", "Water")
numAnglesPerMoleculeWater = countIn("angles", "Water")
# Ethanol parameters:
numMoleculesEthanol = 0
numAtomsPerMoleculeEthanol = countIn("atoms", "Ethanol")
numBondsPerMoleculeEthanol = countIn("bonds", "Ethanol")
numAnglesPerMoleculeEthanol = countIn("angles", "Ethanol")

# Calculate useful information:
totalNumWaterBonds = numMoleculesWater*numBondsPerMoleculeWater
totalNumWaterAngles = numMoleculesWater*numAnglesPerMoleculeWater
totalNumWaterAtoms = numMoleculesWater*numAtomsPerMoleculeWater
totalNumEthanolBonds = numMoleculesEthanol*numBondsPerMoleculeEthanol
totalNumEthanolAngles = numMoleculesEthanol*numAnglesPerMoleculeEthanol
totalNumEthanolAtoms = numMoleculesEthanol*numAtomsPerMoleculeEthanol
totalNumBonds = totalNumWaterBonds + totalNumEthanolBonds
totalNumAngles = totalNumWaterAngles + totalNumEthanolAngles 
totalNumAtoms = totalNumWaterAtoms + totalNumEthanolAtoms

#

# Create arrays with all the bonds and angles
allBonds = np.zeros([totalNumBonds, 4]) # Indices of atoms in first 2 columns, kb in 3rd column, r0 in 4th column
allAngles = np.zeros([totalNumAngles, 5]) # Indices of atoms in first 3 columns, ktheta in 4th column, theta0 in 5th column
allMasses = np.zeros([totalNumAtoms, 1])#max(numAtomsPerMoleculeWater, numAtomsPerMoleculeEthanol)])
allTypes = np.chararray([totalNumAtoms, 1])

# Fill the arrays
for moleculeWater in range(0, numMoleculesWater):
    allBonds[(numBondsPerMoleculeWater*moleculeWater):(numBondsPerMoleculeWater*(moleculeWater + 1)),:] = np.concatenate((structure["Water"]["bonds"] + numAtomsPerMoleculeWater*moleculeWater, structure["Water"]["kb"], structure["Water"]["r0"]), axis = 1) # Indices of atoms in first 2 columns, kb in 3rd column, r0 in 4th column
    allAngles[(numAnglesPerMoleculeWater*moleculeWater):(numAnglesPerMoleculeWater*(moleculeWater + 1)),:] = np.concatenate((structure["Water"]["angles"] + numAtomsPerMoleculeWater*moleculeWater, structure["Water"]["ktheta"], structure["Water"]["theta0"]), axis = 1) # Indices of atoms in first 3 columns, ktheta in 4th column, theta0 in 5th column
    allMasses[(numAtomsPerMoleculeWater*moleculeWater):(numAtomsPerMoleculeWater*(moleculeWater + 1)),:] = structure["Water"]["masses"].reshape(numAtomsPerMoleculeWater, 1)
    allTypes[(numAtomsPerMoleculeWater*moleculeWater):(numAtomsPerMoleculeWater*(moleculeWater + 1)),:] = structure["Water"]["atoms"]#.reshape(numAtomsPerMoleculeWater, 1)

for moleculeEthanol in range(0, numMoleculesEthanol):
    #print(moleculeEthanol)
    allBonds[(numBondsPerMoleculeEthanol*moleculeEthanol + totalNumWaterBonds):(numBondsPerMoleculeEthanol*(moleculeEthanol + 1) + totalNumWaterBonds),:] = np.concatenate((structure["Ethanol"]["bonds"] + numAtomsPerMoleculeEthanol*moleculeEthanol + totalNumWaterAtoms, structure["Ethanol"]["kb"], structure["Ethanol"]["r0"]), axis = 1) # Indices of atoms in first 2 columns, kb in 3rd column, r0 in 4th column
    allAngles[(numAnglesPerMoleculeEthanol*moleculeEthanol + totalNumWaterAngles):(numAnglesPerMoleculeEthanol*(moleculeEthanol + 1) + totalNumWaterAngles),:] = np.concatenate((structure["Ethanol"]["angles"] + numAtomsPerMoleculeEthanol*moleculeEthanol + totalNumWaterAtoms, structure["Ethanol"]["ktheta"], structure["Ethanol"]["theta0"]), axis = 1) # Indices of atoms in first 3 columns, ktheta in 4th column, theta0 in 5th column
    allMasses[(numAtomsPerMoleculeEthanol*moleculeEthanol + totalNumWaterAtoms):(numAtomsPerMoleculeEthanol*(moleculeEthanol + 1) + totalNumWaterAtoms),:numAtomsPerMoleculeEthanol] = structure["Ethanol"]["masses"].reshape(numAtomsPerMoleculeEthanol, 1)
    allTypes[(numAtomsPerMoleculeEthanol*moleculeEthanol + totalNumWaterAtoms):(numAtomsPerMoleculeEthanol*(moleculeEthanol + 1) + totalNumWaterAtoms),:numAtomsPerMoleculeEthanol] = structure["Ethanol"]["atoms"]#.reshape(numAtomsPerMoleculeEthanol, 1)



XYZ = initializeXYZ(numMoleculesWater, numMoleculesEthanol)
v = initializeV(numMoleculesWater, numMoleculesEthanol)

# Start configuration xyz of system at time 0
currentTime = 0
forcesPeriodMin1 = []


if os.path.isfile("myTrajectory.xyz"):
    os.remove("myTrajectory.xyz")
file = open("myTrajectory.xyz","w")
file.write(str(totalNumAtoms)+ "\n")
file.write("Generated trajectory: SOL t=0\n")
file.write('\n'.join('         '.join(str(cell) for cell in row) for row in np.concatenate((allTypes, np.round(XYZ, 5).astype(str)), axis = 1)))
file.write("\n")



"""
for i in range(0, 100):
    file.write(str(totalNumAtoms)+ "\n")
    file.write("Generated trajectory: SOL t=" + str(i) + "\n")
    #file.write(str(np.round(XYZ, 5)))
    file.write('\n'.join('         '.join(str(cell) for cell in row) for row in np.concatenate((allTypes, np.round(XYZ+i, 5).astype(str)), axis = 1)))
    file.write("\n")
"""

forces = calculateForces(XYZ, allBonds, allAngles)

for currentTime in range(0, 100):
    print(currentTime+1)
    
    
    
   
    if currentTime != 0:
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
        forces = calculateForces(XYZ, allBonds, allAngles)
    
    # Verlet:
    if myIntegrator == "verlet":
        if currentTime == 0:
            # Forward Euler
            print("Euler")
            XYZ = XYZPeriodMin1 + dt*v + (dt**2)*(10**-4)*forcesPeriodMin1/(2*allMasses)
            #v = vPeriodMin1 + dt*(10**-4)*forcesPeriodMin1/allMasses
        else:
            print("Verlet")
            XYZ = 2*XYZPeriodMin1 - XYZPeriodMin2 + (dt**2)*(10**-4)*forcesPeriodMin1/(allMasses)
            vPeriodMin1 = (XYZ - XYZPeriodMin2)/(2*dt)
        forces = calculateForces(XYZ, allBonds, allAngles)
            
    # Verlet velocity:
    if myIntegrator == "velocity verlet":
        print("Velocity Verlet")
        XYZ = XYZPeriodMin1 + dt*vPeriodMin1 + (dt**2)*(10**-4)*forcesPeriodMin1/(2*allMasses)
        forces = calculateForces(XYZ, allBonds, allAngles)
        v = vPeriodMin1 + dt*(10**-4)*(forcesPeriodMin1 + forces)/(2*allMasses)
        #vPeriodMin1 = vPeriodMin2 + dt*(forcesPeriodMin2 + forcesPeriodMin1)/(2*allMasses)
            
    
    
    file.write(str(totalNumAtoms)+ "\n")
    file.write("Generated trajectory: SOL t=" + str((currentTime + 1)*dt) + "\n")
    file.write('\n'.join('         '.join(str(cell) for cell in row) for row in np.concatenate((allTypes, np.round(XYZ, 5).astype(str)), axis = 1)))
    file.write("\n")




file.close()


'''

# Start configuration of velocities at time 0:
v = {currentTime: np.zeros(XYZ[currentTime].shape)}


# Select the XYZ and v at this moment
XYZNow = XYZ[currentTime]
vNow = v[currentTime]


result = calculateForces(XYZNow, allBonds, allAngles)







###################################################################
###################################################################


# Initialize the force on each atom:
force = {0: np.zeros(XYZNow.shape)}
forceNow = force[currentTime]

# Now integrate:

        
# 50 000 time steps: 0.1 ns = 100 000 fs. Timestep = 2 fs


if myIntegrator == "Euler":
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
    XYZNext = XYZNow + dt*vNow + (dt**2)*(10**-4)*forceNow/(2*allMasses)
    
    if currentTime == 0:
        vNext = (XYZNext - XYZNow)/dt
else:
    print("No integrator found")






# Calculate all the force within the molecules:
for molecule in range(0, numMoleculesWater):
    for atom in range(0, numAtomsPerMoleculeWater):
        
        # Do all the bond forces:
        for bondedTo in bonds[atom]:
            r = np.linalg.norm(XYZNow[3*molecule + atom] - XYZNow[3*molecule + bondedTo])
            forceNow[molecule, atom] = forceNow[molecule, atom] + (kb*(r - r0)/r)*(XYZNow[3*molecule + atom] - XYZNow[3*molecule + bondedTo])
        #forceNow[molecule, atom] = forceNow[molecule, atom] + 
        #forceNow[molecule, atom] = forceNow[molecule, atom] + 



forceNow


'''


