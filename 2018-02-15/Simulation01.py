#%reset -f

###################################################################################
################################# IMPORTS #########################################
###################################################################################

# Python imports and settings
import os
#os.chdir("C:/Users/roman/Documents/University/Eindhoven/Introduction to Molecular Modeling and Simulation/Python")
import numpy as np
import matplotlib.pyplot as plt
import time

s = time.time()

np.set_printoptions(suppress=True)

# Import own functions:
from functionsSimulation import *



###################################################################################
################################### MAIN ##########################################
###################################################################################


####################################
##### Parameters for the user: #####
####################################

# Set your integrator: choose from "euler", "verlet", "velocity verlet"
myIntegrator = "velocity verlet"

# Set time parameters:
dt = 2 # Timestep for numerical integrator in fs
totalTime = 100000 # Total simulation time in fs

# Set properties for the system:
boxSize = 31 # in Angström: box will have a volume of boxSize^3
totalNumMolecules = 10**3 # Has to be a number to the power 3! 
percentageEthanol = 0 # In percentage in [0, 100]

# Thermostat: do we want to rescale the velocities: rescaling means that we use a thermostat
rescale = 1 # in {0,1}: 1 means: do a rescale (i.e. use a thermostat) every time step. 0 means: no rescaling

# Set the initial temperature (if we do not use a thermostat) or set the temperature to be used by the thermostat as target temperature:
targetTemperature = 300 # in Kelvin

# Set the name of the .xyz file to write to:
myFileName = "XYZ" # Do not add .xyz


#####################################
##### More specific parameters: #####
#####################################

# Lennart Jones cut off length:
rLJCutOff = 8 # in Angström

# Parameters for the initialization of the velocities (in Angström/fs):
meanV = 0 # in Angström / fs
stDevV = 0.05



###################################
##### Simulation starts here: #####
###################################

# Give a warning on the adjustment that was done in the assignment:
print("Warning: angle 314 in assignment is changed to 312 since 413 already exists! In the structure substract 1: 203 changed to 201")

# Initialize configuration of the system:
XYZInitial, allBonds, allAngles, allDihedrals, allMasses, allTypes, allMoleculeNumbers = initializeConfiguration(totalNumMolecules, percentageEthanol, boxSize)
XYZ = XYZInitial #+  [5, 5, 5]
vInitial = initializeV(allMoleculeNumbers, meanV, stDevV)
v = vInitial

# Find total number of atoms in system:
totalNumAtoms = XYZ.shape[0]

myFileName = myFileName + " with numMolecules " + str(totalNumMolecules) + " time " + str(totalTime) + " fs dt " + str(dt) + " fs box " + str(boxSize/10) + " nm percEthanol " + str(percentageEthanol) + " rescale " + str(rescale) + " targetTemp " + str(targetTemperature) + " K rLJcut " + str(rLJCutOff) + " nm.xyz"
# Delete the file to write to if it exists already:
if os.path.isfile(myFileName):
    os.remove(myFileName)

# Open the file to write to:
file = open(myFileName,"w")

# Write the first timestep (t = 0) to the .xyz file. Note that the molecules moving freely through space are to be put back into the box:
file.write(str(totalNumAtoms)+ "\n")
file.write("Generated trajectory: SOL t=0\n")
file.write('\n'.join('         '.join(str(cell) for cell in row) for row in np.concatenate((allTypes[:, 0].reshape(allTypes.shape[0], 1), np.round(np.mod(XYZ, boxSize), 5).astype(str)), axis = 1)))
file.write("\n")


# Initialize temperature vector:
temperature = np.zeros(int(totalTime/dt+1))

# Use the thermostat to set the starting temperature right:
temp, v = thermostat(v, allMasses, 1, targetTemperature)

# Use the thermostate to calculate current temperature and to rescale this:
temperature[0], v = thermostat(v, allMasses, rescale, targetTemperature)

# Start configuration xyz of system at time 0
currentTime = 0

# Initialize a few variables:
forcesPeriodMin1 = []
potentialEnergy = np.zeros(int(totalTime/dt+1))
kineticEnergy = np.zeros(int(totalTime/dt+1))
index = 0

# Start timer
myTime = time.time()

# Calculate the forces and energies at t = 0
forces, potentialEnergy[0] = calculateForcesEnergy(XYZ, allBonds, allAngles, allDihedrals, allTypes, allMoleculeNumbers, boxSize, rLJCutOff)
kineticEnergy[0] = sum(0.5*allMasses[:,0]*((np.linalg.norm(v, axis = 1))**2))

# Time the first step and message how long it took
elapsedTime = time.time() - myTime
print("The first step took " + str(elapsedTime) + " seconds.")

# Start simulation: iterate over t in {dt, 2*dt, 3*dt, ..., totalTime-dt, totalTime}
for currentTime in np.arange(dt, totalTime+dt, dt):
    
    # Start timer
    myTime = time.time()
    
    # Print at which time we are currently:
    print("At time " + str(currentTime) + " which is " + str(currentTime*100/totalTime) + " %")
    
    # Increase our index by 1
    index += 1
    
    # Remember values from last round:
    if currentTime != dt:
        XYZPeriodMin2 = XYZPeriodMin1 
    XYZPeriodMin1 = XYZ
    vPeriodMin1 = v
    forcesPeriodMin1 = forces
       
    
    ###### Get units right: #######
    
    # [U] = kJ / mol 
    # [F] = kJ / mol / Angström = 1000 N*m / mol / Angström = 10^3 kg * m^2 / s^2 / mol / Angström
    # = 10^6 g * m^2 / (10^30 fs^2) / mol / Angström = 10^-24  g * m^2 / fs^2 / mol / Angström
    # = 10^-24  g * m^2 / fs^2 / mol / (10^-10 m) = 10^-14  g * m / fs^2 / mol 
    # = 10^-4  g * Angström / fs^2 / mol 
    # [t] = [dt]= fs 
    # [q] = Angström
    # [v] = Angström / fs
    # so F must be multiplied by 10^-4 and m must have unit: [m] = g/mol = amu
        
    # Euler numerical integration:
    if myIntegrator == "euler":
        XYZ = XYZPeriodMin1 + dt*v + (dt**2)*(10**-4)*forcesPeriodMin1/(2*allMasses)
        v = vPeriodMin1 + dt*(10**-4)*forcesPeriodMin1/allMasses
        temperature[index], v = thermostat(v, allMasses, rescale, targetTemperature)
        kineticEnergy[index] = sum(0.5*allMasses[:,0]*((np.linalg.norm(v, axis = 1))**2))
        forces, potentialEnergy[index] = calculateForcesEnergy(XYZ, allBonds, allAngles, allDihedrals, allTypes, allMoleculeNumbers, boxSize, rLJCutOff)
    
    # Verlet numerical integration:
    if myIntegrator == "verlet":
        if currentTime == dt:
            # Forward Euler for the first time step
            XYZ = XYZPeriodMin1 + dt*v + (dt**2)*(10**-4)*forcesPeriodMin1/(2*allMasses)
        else:
            # Verlet:
            XYZ = 2*XYZPeriodMin1 - XYZPeriodMin2 + (dt**2)*(10**-4)*forcesPeriodMin1/(allMasses)
            vPeriodMin1 = (XYZ - XYZPeriodMin2)/(2*dt)
            temperature[index], vPeriodMin1 = thermostat(vPeriodMin1, allMasses, rescale, targetTemperature)
            kineticEnergy[index-1] = sum(0.5*allMasses[:,0]*((np.linalg.norm(vPeriodMin1, axis = 1))**2))
        forces, potentialEnergy[index] = calculateForcesEnergy(XYZ, allBonds, allAngles, allDihedrals, allTypes, allMoleculeNumbers, boxSize, rLJCutOff)
            
    # Verlet velocity numerical integration:
    if myIntegrator == "velocity verlet":
        XYZ = XYZPeriodMin1 + dt*vPeriodMin1 + (dt**2)*(10**-4)*forcesPeriodMin1/(2*allMasses)
        codeTimer = time.time()
        forces, potentialEnergy[index] = calculateForcesEnergy(XYZ, allBonds, allAngles, allDihedrals, allTypes, allMoleculeNumbers, boxSize, rLJCutOff)
        print("Calculating forces took " + str(time.time() - codeTimer) + " seconds.")
        v = vPeriodMin1 + dt*(10**-4)*(forcesPeriodMin1 + forces)/(2*allMasses)
        temperature[index], v = thermostat(v, allMasses, rescale, targetTemperature)
        kineticEnergy[index] = sum(0.5*allMasses[:,0]*((np.linalg.norm(v, axis = 1))**2))
            
    # No integrator found message:
    if myIntegrator not in ["euler", "verlet", "velocity verlet"]:
        print("No integrator with the name " + str(myIntegrator) + " found!")
    
    
    # Write the current round to the .xyz file. Note that the molecules moving freely through space are to be put back into the box:
    file.write(str(totalNumAtoms)+ "\n")
    file.write("Generated trajectory: SOL t=" + str(currentTime) + "\n")
    file.write('\n'.join('         '.join(str(cell) for cell in row) for row in np.concatenate((allTypes[:, 0].reshape(allTypes.shape[0], 1), np.round(np.mod(XYZ, boxSize), 5).astype(str)), axis = 1)))
    file.write("\n")
    
    # Time the round and message how long it took
    elapsedTime = time.time() - myTime
    print("This step took " + str(elapsedTime) + " seconds.")



# Give information on the just completed simulation:
print("This simulation was done using " + str(myIntegrator) + " as numerical integrator")
print("This simulation contained " + str(totalNumMolecules) + " molecules of a mixture water-ethanol with an ethanol percentage of " + str(percentageEthanol) + " percent in a " + str(boxSize) + " by " + str(boxSize) + " by " + str(boxSize) + " angström box.")
print("This simulation had a timespan of " + str(totalTime) + " fs and a timestep of " + str(dt) + " fs.")
if rescale == 1:
    print("A thermostat was used. The simulation was done at " + str(targetTemperature) + " Kelvin.")
else:
    print("No thermostat was used. The simulation was started at " + str(targetTemperature) + " Kelvin.")

# Give again a warning on the adjustment that was done in the assignment:
print("Warning: angle 314 in assignment is changed to 312 since 413 already exists! In the structure substract 1: 203 changed to 201")

# Scale the energies to kJ/mol:
kineticEnergy = (10**4)*kineticEnergy # in kJ/mol
totalEnergy = potentialEnergy + kineticEnergy # in kJ/mol

# Make a plot for the energies:
plt.figure(figsize=(20, 10))
plt.plot(np.arange(0, totalTime+dt, dt), totalEnergy, 'b-', np.arange(0, totalTime+dt, dt), potentialEnergy, 'r-', np.arange(0, totalTime+dt, dt), kineticEnergy, 'g-')
plt.axis([0, totalTime, min(min(totalEnergy), min(kineticEnergy), min(potentialEnergy))*1.2, max(max(totalEnergy), max(kineticEnergy), max(potentialEnergy))*1.2])
plt.xlabel("Time in fs")
plt.ylabel("Energy in kJ/mol")
plt.title("Energy levels over time")
plt.legend(["Total energy", "Potential energy", "Kinetic energy"])

# Close the file we wrote all data to:
file.close()

# Write temperatures, potential, kinetic, total energies to file:
myFileNameTempEnergy = "T and E with numMolecules " + str(totalNumMolecules) + " time " + str(totalTime) + " fs dt " + str(dt) + " fs box " + str(boxSize/10) + " nm percEthanol " + str(percentageEthanol) + " rescale " + str(rescale) + " targetTemp " + str(targetTemperature) + " K rLJcut " + str(rLJCutOff) + " nm.csv"
#myFileNameTempEnergy = "T and E.csv"
if os.path.isfile(myFileNameTempEnergy):
    os.remove(myFileNameTempEnergy)
filetempenergy = open(myFileNameTempEnergy,"w")
filetempenergy.write("Temperature;Kinetic energy;Potential energy;Total energy\n")
filetempenergy.write('\n'.join(';'.join(str(cell) for cell in row) for row in np.concatenate((temperature.reshape(index+1, 1),kineticEnergy.reshape(index+1, 1), potentialEnergy.reshape(index+1, 1), totalEnergy.reshape(index+1, 1)), axis=1).astype(str)))
filetempenergy.write("\n")
filetempenergy.close()





totalElapsedTime = time.time() - s
print("This simulation took " + str(totalElapsedTime) + " seconds.")

