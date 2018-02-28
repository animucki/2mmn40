# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 21:38:06 2018

@author: roman
"""

###################################################################################
################################# FUNCTIONS #######################################
###################################################################################
import numpy as np


# Define the structure of different molecules:
def getMoleculeStructure():
    #import numpy as np
    struc =  {"Water": {"atoms": [["O"], ["H"], ["H"]], "masses": np.array([15.9994, 1.0080, 1.0080]), "bonds": np.array([[0, 1], [0, 2]]), "angles": np.array([[1, 0, 2]]), "dihedrals": np.array([]), "kb": np.array([[5024.16], [5024.16]]), "r0": np.array([[0.9572], [0.9572]]), "ktheta": np.array([[6.2802]]), "theta0": np.array([[104.52*np.pi/180]])}, # Oxygen atom is 0, the hydrogens are 1 and 2
             "Hydrogen": {"atoms": [["H"], ["H"]], "masses": np.array([1.0080, 1.0080]), "bonds": np.array([[0, 1]]), "angles": np.array([]), "dihedrals": np.array([])}, # The hydrogens are 0 and 1
             "Methane": {"atoms": [["C"], ["H"], ["H"], ["H"], ["H"]], "masses": np.array([12.0110, 1.0080, 1.0080, 1.0080, 1.0080]), "bonds": np.array([[0, 1], [0, 2], [0, 3], [0, 4]]), "angles": np.array([[1, 0, 2], [1, 0, 3], [1, 0, 4], [2, 0, 3], [2, 0, 4], [3, 0, 4]]), "dihedrals": np.array([])}, # The carbon is 0, the hydrogens 1, 2, 3, 4
             "Ethanol": {"atoms": [["C"], ["H"], ["H"], ["H"], ["C"], ["H"], ["H"], ["O"], ["H"]], "masses": np.array([12.0110, 1.0080, 1.0080, 1.0080, 12.0110, 1.0080, 1.0080, 15.9994, 1.0080]), "bonds": np.array([[0, 1], [0, 2], [0, 3], [0, 4], [4, 5], [4, 6], [4, 7], [7, 8]]), "angles": np.array([[1, 0, 4], [2, 0, 4], [3, 0, 4], [3, 0, 2], [3, 0, 1], [2, 0, 3], [5, 4, 6], [0, 4, 6], [0, 4, 5], [0, 4, 7], [4, 7, 8], [5, 4, 7], [6, 4, 7]]), "dihedrals": np.array([[1, 0, 4, 5], [2, 0, 4, 5], [3, 0, 4, 5], [1, 0, 4, 6], [2, 0, 4, 6], [3, 0, 4, 6], [1, 0, 4, 7], [2, 0, 4, 7], [3, 0, 4, 7], [0, 4, 7, 8], [5, 4, 7, 8], [6, 4, 7, 8]]), "kb": np.array([[2845.12], [2845.12] , [2845.12], [2242.624], [2845.12], [2845.12], [2677.76], [4627.50]]), "r0": np.array([[1.090], [1.090] , [1.090], [1.529], [1.090], [1.090], [1.410], [0.945]]), "ktheta": np.array([[2.9288], [2.9288] , [2.9288], [2.76144], [2.76144], [2.76144], [2.76144], [3.138], [3.138], [4.144] , [4.6024], [2.9288], [2.9288]]), "theta0": np.array([[108.5*np.pi/180], [108.5*np.pi/180] , [108.5*np.pi/180], [107.8*np.pi/180], [107.8*np.pi/180], [107.8*np.pi/180], [107.8*np.pi/180], [110.7*np.pi/180], [110.7*np.pi/180], [109.5*np.pi/180] , [108.5*np.pi/180], [109.5*np.pi/180], [109.5*np.pi/180]])} # Obvious from definition
             }
    
    return struc

def getLennartJonesParameters():
    struc = {"Water": {"epsilon": [0.66386, 0, 0], "sigma": [3.15061, 0, 0]},
             "Ethanol": {"epsilon": [0.276144, 0.125520, 0.125520, 0.125520, 0.276144, 0.125520, 0.125520, 0.711280, 0], "sigma": [3.5, 2.5, 2.5, 2.5, 3.5, 2.5, 2.5, 3.12, 0]}
             }
    return struc


# Count atoms/bonds/angles/dihedrals in molecule B
def countIn(A, B):
    struc = getMoleculeStructure()
    return len(struc[B][A])



# Initialize the positions
'''def initializeXYZ(numWater, numEthanol):
    # Put starting XYZ in Angström!
    import numpy as np
    # 2 water:
    #XYZ = {currentTime: np.array([[1.3, 16.2, 16.8], [1.9, 16.6, 17.4], [1.8, 15.7, 16.2], [12.7, 0.5, 6.2], [13.4, 0.1, 6.8], [13.3, 1.2, 5.7]])}
    # 2 water and 1 ethanol:
        
    #startXYZ[0.02] = [[1.4, 16.2, 16.8], [1.8, 16.6, 17.4], [1.8, 15.7, 16.2]]
    #del startXYZ[0.01]
    numMoleculesApprox = 10**3
    numMoleculeOneDirec = int(np.round(numMoleculesApprox**(1/3)))
    grid = np.meshgrid([x for x in range(0, numMoleculeOneDirec)], [x for x in range(0, numMoleculeOneDirec)], [x for x in range(0, numMoleculeOneDirec)])
    gridVectors = np.concatenate((grid[0].reshape(numMoleculesApprox, 1), grid[1].reshape(numMoleculesApprox, 1), grid[2].reshape(numMoleculesApprox, 1)), axis = 1)
    
    basicWater = np.array([[0, 0.5, 0.6], [0.6, 0.9, 1.2], [0.5, 0, 0]])
    basicEthanol = np.array([[-1.683, -0.523, 1.084], [-1.689, -1.638, 1.101], [-1.171, -0.174, 2.011], [-2.738, -0.167, 1.117], [-0.968, -0.008, -0.167], [0.094, -0.344, -0.200], [-1.490, -0.319, -1.102], [-0.953, 1.395, -0.142], [-1.842, 1.688, -0.250]]) + np.array([2.738, 1.638, 1.102])
    
    XYZinitial = basicWater + gridVectors[0]
    
    for m in range(0, numMoleculesApprox-1):
        #print(m)
        XYZinitial = np.concatenate((XYZinitial, basicWater + (1/numMoleculeOneDirec)*50*gridVectors[m + 1]))
    
    
    
    XYZinitial = np.concatenate((np.array([[1.3, 16.2, 16.8], [1.9, 16.6, 17.4], [1.8, 15.7, 16.2], [12.7, 0.5, 6.2], [13.4, 0.1, 6.8], [13.3, 1.2, 5.7]]), basicEthanol), axis = 0)
    
    XYZinitial = np.array([[1.3, 16.2, 16.8], [1.9, 16.6, 17.4], [1.8, 15.7, 16.2]])
    XYZinitial = basicEthanol
    
    return XYZinitial'''

def initializeConfiguration(totalNumMolecules, percentageEthanol):
    import numpy as np
    
    
    # Ethanol parameters:
    numMoleculesEthanol = int(np.round(totalNumMolecules*percentageEthanol/100))
    numAtomsPerMoleculeEthanol = countIn("atoms", "Ethanol")
    numBondsPerMoleculeEthanol = countIn("bonds", "Ethanol")
    numAnglesPerMoleculeEthanol = countIn("angles", "Ethanol")
    # Water parameters:
    numMoleculesWater = totalNumMolecules - numMoleculesEthanol
    numAtomsPerMoleculeWater = countIn("atoms", "Water")
    numBondsPerMoleculeWater = countIn("bonds", "Water")
    numAnglesPerMoleculeWater = countIn("angles", "Water")

    # Calculate useful information:
    totalNumWaterBonds = numMoleculesWater*numBondsPerMoleculeWater
    totalNumWaterAngles = numMoleculesWater*numAnglesPerMoleculeWater
    totalNumWaterAtoms = numMoleculesWater*numAtomsPerMoleculeWater
    totalNumEthanolBonds = numMoleculesEthanol*numBondsPerMoleculeEthanol
    totalNumEthanolAngles = numMoleculesEthanol*numAnglesPerMoleculeEthanol
    totalNumEthanolAtoms = numMoleculesEthanol*numAtomsPerMoleculeEthanol
    totalNumBonds = totalNumWaterBonds + totalNumEthanolBonds
    totalNumAngles = totalNumWaterAngles + totalNumEthanolAngles
    totalNumAtoms = numMoleculesWater*countIn("atoms", "Water") + numMoleculesEthanol*countIn("atoms", "Ethanol")
        
    # Create arrays with all the bonds and angles
    allBonds = np.zeros([totalNumBonds, 4]) # Indices of atoms in first 2 columns, kb in 3rd column, r0 in 4th column
    allAngles = np.zeros([totalNumAngles, 5]) # Indices of atoms in first 3 columns, ktheta in 4th column, theta0 in 5th column
    allMasses = np.zeros([totalNumAtoms, 1])#max(numAtomsPerMoleculeWater, numAtomsPerMoleculeEthanol)])
    allTypes = np.empty([totalNumAtoms, 2], dtype="<U7")
    allMoleculeNumbers = np.zeros([totalNumAtoms, 2], dtype = int)
        
    np.random.seed(0)
    ethanolPositions = np.random.choice(totalNumMolecules, numMoleculesEthanol, replace = False)
    
    moleculeVector = np.empty([totalNumMolecules, 1], dtype = "<U7")
    moleculeVector[:, 0] = "Water"
    moleculeVector[ethanolPositions, 0] = "Ethanol"
    repeatVector = np.zeros([totalNumMolecules, 1], dtype = int)
    repeatVector[:, 0] = countIn("atoms", "Water")
    repeatVector[ethanolPositions, 0] = countIn("atoms", "Ethanol")
    allTypes[:, 1] = np.repeat(moleculeVector[:,0], repeatVector[:,0])#.reshape([totalNumAtoms, 1])
    #np.diff(np.append(np.append([-1], np.where((np.diff(a) != 0))), [len(a)-1]))
    # Fill the arrays
    
   
    totalNumAngles = totalNumWaterAngles + totalNumEthanolAngles
    
    currentBondIndex = 0
    currentAngleIndex = 0
    currentAtomIndex = 0
    
    structure = getMoleculeStructure()

    for molecule in range(0, totalNumMolecules):
        moleculeType = moleculeVector[molecule, 0]
        bondsInType = countIn("bonds", moleculeType)
        anglesInType = countIn("angles", moleculeType)
        atomsInType = countIn("atoms", moleculeType)
        allBonds[currentBondIndex:(currentBondIndex + bondsInType),:] = np.concatenate((structure[moleculeType]["bonds"] + currentAtomIndex, structure[moleculeType]["kb"], structure[moleculeType]["r0"]), axis = 1) # Indices of atoms in first 2 columns, kb in 3rd column, r0 in 4th column
        allAngles[currentAngleIndex:(currentAngleIndex + anglesInType),:] = np.concatenate((structure[moleculeType]["angles"] + currentAtomIndex, structure[moleculeType]["ktheta"], structure[moleculeType]["theta0"]), axis = 1) # Indices of atoms in first 3 columns, ktheta in 4th column, theta0 in 5th column
        allMasses[currentAtomIndex:(currentAtomIndex + atomsInType),:] = structure[moleculeType]["masses"].reshape(countIn("atoms", moleculeType), 1)
        allTypes[currentAtomIndex:(currentAtomIndex + atomsInType),0] = np.concatenate((structure[moleculeType]["atoms"], structure[moleculeType]["atoms"]), axis = 1)[:,0]#np.concatenate((structure["Water"]["atoms"], np.repeat("Water", numAtomsPerMoleculeWater).reshape(numAtomsPerMoleculeWater, 1)), axis=1)#.reshape(numAtomsPerMoleculeWater, 1)
        allMoleculeNumbers[currentAtomIndex:(currentAtomIndex + atomsInType),:] = np.transpose(np.array([atomsInType*[molecule], [x for x in range(0, atomsInType)]]))#np.array([[molecule, 0], [molecule, 1], [molecule, 2]])
        
        currentBondIndex += bondsInType
        currentAngleIndex += anglesInType
        currentAtomIndex += atomsInType
     
    '''
    for moleculeWater in range(0, numMoleculesWater):
        allBonds[(numBondsPerMoleculeWater*moleculeWater):(numBondsPerMoleculeWater*(moleculeWater + 1)),:] = np.concatenate((structure["Water"]["bonds"] + numAtomsPerMoleculeWater*moleculeWater, structure["Water"]["kb"], structure["Water"]["r0"]), axis = 1) # Indices of atoms in first 2 columns, kb in 3rd column, r0 in 4th column
        allAngles[(numAnglesPerMoleculeWater*moleculeWater):(numAnglesPerMoleculeWater*(moleculeWater + 1)),:] = np.concatenate((structure["Water"]["angles"] + numAtomsPerMoleculeWater*moleculeWater, structure["Water"]["ktheta"], structure["Water"]["theta0"]), axis = 1) # Indices of atoms in first 3 columns, ktheta in 4th column, theta0 in 5th column
        allMasses[(numAtomsPerMoleculeWater*moleculeWater):(numAtomsPerMoleculeWater*(moleculeWater + 1)),:] = structure["Water"]["masses"].reshape(numAtomsPerMoleculeWater, 1)
        allTypes[(numAtomsPerMoleculeWater*moleculeWater):(numAtomsPerMoleculeWater*(moleculeWater + 1)),:] = np.concatenate((structure["Water"]["atoms"], np.repeat("Water", numAtomsPerMoleculeWater).reshape(numAtomsPerMoleculeWater, 1)), axis=1)#.reshape(numAtomsPerMoleculeWater, 1)
        allMoleculeNumbers[(numAtomsPerMoleculeWater*moleculeWater):(numAtomsPerMoleculeWater*(moleculeWater + 1)),:] = np.array([[moleculeWater, 0], [moleculeWater, 1], [moleculeWater, 2]])
     
    for moleculeEthanol in range(0, numMoleculesEthanol):
        #print(moleculeEthanol)
        allBonds[(numBondsPerMoleculeEthanol*moleculeEthanol + totalNumWaterBonds):(numBondsPerMoleculeEthanol*(moleculeEthanol + 1) + totalNumWaterBonds),:] = np.concatenate((structure["Ethanol"]["bonds"] + numAtomsPerMoleculeEthanol*moleculeEthanol + totalNumWaterAtoms, structure["Ethanol"]["kb"], structure["Ethanol"]["r0"]), axis = 1) # Indices of atoms in first 2 columns, kb in 3rd column, r0 in 4th column
        allAngles[(numAnglesPerMoleculeEthanol*moleculeEthanol + totalNumWaterAngles):(numAnglesPerMoleculeEthanol*(moleculeEthanol + 1) + totalNumWaterAngles),:] = np.concatenate((structure["Ethanol"]["angles"] + numAtomsPerMoleculeEthanol*moleculeEthanol + totalNumWaterAtoms, structure["Ethanol"]["ktheta"], structure["Ethanol"]["theta0"]), axis = 1) # Indices of atoms in first 3 columns, ktheta in 4th column, theta0 in 5th column
        allMasses[(numAtomsPerMoleculeEthanol*moleculeEthanol + totalNumWaterAtoms):(numAtomsPerMoleculeEthanol*(moleculeEthanol + 1) + totalNumWaterAtoms),:numAtomsPerMoleculeEthanol] = structure["Ethanol"]["masses"].reshape(numAtomsPerMoleculeEthanol, 1)
        allTypes[(numAtomsPerMoleculeEthanol*moleculeEthanol + totalNumWaterAtoms):(numAtomsPerMoleculeEthanol*(moleculeEthanol + 1) + totalNumWaterAtoms),:numAtomsPerMoleculeEthanol] = np.concatenate((structure["Ethanol"]["atoms"], np.repeat("Ethanol", numAtomsPerMoleculeEthanol).reshape(numAtomsPerMoleculeEthanol, 1)), axis=1)#.reshape(numAtomsPerMoleculeWater, 1)
        allMoleculeNumbers[(numAtomsPerMoleculeEthanol*moleculeEthanol + totalNumWaterAtoms):(numAtomsPerMoleculeEthanol*(moleculeEthanol + 1) + totalNumWaterAtoms),:numAtomsPerMoleculeEthanol] = np.array([[moleculeEthanol + numMoleculesWater, 0], [moleculeEthanol + numMoleculesWater, 1], [moleculeEthanol + numMoleculesWater, 2], [moleculeEthanol + numMoleculesWater, 3], [moleculeEthanol + numMoleculesWater, 4], [moleculeEthanol + numMoleculesWater, 5], [moleculeEthanol + numMoleculesWater, 6], [moleculeEthanol + numMoleculesWater, 7], [moleculeEthanol + numMoleculesWater, 8]])
    '''
    numMoleculesApprox = totalNumMolecules
    numMoleculeOneDirec = int(np.round(numMoleculesApprox**(1/3)))
    grid = np.meshgrid([x for x in range(0, numMoleculeOneDirec)], [x for x in range(0, numMoleculeOneDirec)], [x for x in range(0, numMoleculeOneDirec)])
    gridVectors = np.concatenate((grid[0].reshape(numMoleculesApprox, 1), grid[1].reshape(numMoleculesApprox, 1), grid[2].reshape(numMoleculesApprox, 1)), axis = 1)
    
    basicWater = np.array([[0, 0.5, 0.6], [0.6, 0.9, 1.2], [0.5, 0, 0]])
    basicEthanol = np.array([[-1.683, -0.523, 1.084], [-1.689, -1.638, 1.101], [-1.171, -0.174, 2.011], [-2.738, -0.167, 1.117], [-0.968, -0.008, -0.167], [0.094, -0.344, -0.200], [-1.490, -0.319, -1.102], [-0.953, 1.395, -0.142], [-1.842, 1.688, -0.250]]) + np.array([2.738, 1.638, 1.102])
    
    if moleculeVector[0] == "Water":
        XYZinitial = basicWater + gridVectors[0]
    else:
        XYZinitial = basicEthanol + gridVectors[0]
    
    for m in range(0, numMoleculesApprox-1):
        #print(m)
        if moleculeVector[m+1] == "Water":
            XYZinitial = np.concatenate((XYZinitial, basicWater + (1/numMoleculeOneDirec)*50*gridVectors[m + 1]))
        else:
            XYZinitial = np.concatenate((XYZinitial, basicEthanol + (1/numMoleculeOneDirec)*50*gridVectors[m + 1]))
        
    
    return XYZinitial, allBonds, allAngles, allMasses, allTypes, allMoleculeNumbers

# Initialize the velocities
def initializeV(allMoleculeNumbers, meanV, stDevV):
    #numMoleculesWater = numWater*countIn("atoms", "Water")
    #numMoleculesEthanol = numEthanol*countIn("atoms", "Ethanol")
    replications = np.diff(np.append(np.append([-1], np.where((np.diff(allMoleculeNumbers[:,0]) != 0))), [len(allMoleculeNumbers[:,0])-1]))
    totalNumMolecules = max(allMoleculeNumbers[:, 0])+1
    
    #v = np.concatenate((np.repeat(np.random.normal(meanV, stDevV, [numWater, 3]), countIn("atoms", "Water"), axis = 0), np.repeat(np.random.normal(meanV, stDevV, [numEthanol, 3]), countIn("atoms", "Ethanol"), axis = 0)), axis=0)
    np.random.seed(1)
    v = np.repeat(np.random.normal(meanV, stDevV, [totalNumMolecules, 3]), replications, axis=0)
    return v


# Create a calculateForces function:
def calculateForces(atomListXYZNow, bondList, angleList, typeList, moleculeNumberList, boxSize, rLJCutOff): # add dihedralList
    
    forces = np.zeros(atomListXYZNow.shape)
    potentialEnergy = 0
    
    
    ########### Bonds #############
    
    # Calculate all the bond forces within the molecules, iterate for all bonds in a for loop:
    for b in range(0, bondList.shape[0]):
        
        # Calculate distance between two atoms
        r = np.linalg.norm(atomListXYZNow[int(bondList[b, 0])] - atomListXYZNow[int(bondList[b, 1])])
         
        # Add potential energy:
        potentialEnergy += 0.5*bondList[b, 2]*((r-bondList[b, 3])**2)
        
        # Find the magnitude of the force:
        # r = sqrt((qx_1 - qx_0)^2 + (qy_1 - qy_0)^2 + (qz_1 - qz_0)^2) 
        # V = 1/2 * kb * (r - r0)^2 
        # Force wrt atom 0: F_0 = -grad(V) = kb * (r-r0)/r * (q_0-q_1)
        # Force wrt atom 1: F_1 = - kb * (r-r0)/r * (q_0-q_1)
        # Magnitude ||F|| = |kb|*|r-r0| * ( ||q_0 - q_1||/r ) = |kb|*|r-r0|
        magnitudeForce = bondList[b, 2]*abs(r-bondList[b, 3])
                
        # Find force direction with respect to atom 0 and r> r0: q_1 - q_0 
        # Normalize: q_1 - q_0 / r
        # Calculate Normalized force direction * magnitude with respect to atom 0 and the case that r > r0
        forceVectorWrt1Case1 = (atomListXYZNow[int(bondList[b, 1])] - atomListXYZNow[int(bondList[b, 0])])*magnitudeForce/r
                
        # Case 1:
        if r > bondList[b, 3]: 
            forces[int(bondList[b, 0])] = forces[int(bondList[b, 0])] + forceVectorWrt1Case1
            forces[int(bondList[b, 1])] = forces[int(bondList[b, 1])] - forceVectorWrt1Case1
            
        # Case 2:
        else:
            forces[int(bondList[b, 0])] = forces[int(bondList[b, 0])] - forceVectorWrt1Case1
            forces[int(bondList[b, 1])] = forces[int(bondList[b, 1])] + forceVectorWrt1Case1
            
    
    
    ########### Angles #############
    
    # Calculate all the angle forces within the molecules:
    for a in range(0, angleList.shape[0]):
    
        # Strucure:         atom2
        #                  /     \
        #                 /       \
        #                /       atom3
        #             atom1 
        #
        # with angle theta: angle 123 (< 180 degr)
        
        # Vector v21 is the vector from atom 2 to atom 1
        v21 = atomListXYZNow[int(angleList[a, 0]),] - atomListXYZNow[int(angleList[a, 1]),]
        # Vector v21 is the vector from atom 2 to atom 3 
        v23 = atomListXYZNow[int(angleList[a, 2]),] - atomListXYZNow[int(angleList[a, 1]),]
        
        # Find the angle theta
        #print(np.dot(v21, v23))
        #print(len(np.linalg.norm(v21) * np.linalg.norm(v23)))
        '''if currentTime == 14:
            print(np.dot(v21, v23))
            print(np.linalg.norm(v21) * np.linalg.norm(v23))'''
        theta = np.arccos((np.dot(v21, v23))/(np.linalg.norm(v21) * np.linalg.norm(v23)))
        
        # Add potential energy:
        potentialEnergy += 0.5*angleList[a, 3]*((theta - angleList[a, 4])**2)
        
        # Find the magnitude of the force acting on atom 1 
        # ||F_atom1|| = || grad_atom1(V) || = || dV/dtheta || * || dtheta / dq_atom1 || 
        # = | ktheta | * | theta - theta0 | * 1/| q_atom2 - q_atom1 |
        magnitudeForceOnAtom1 = angleList[a, 3]*abs(theta - angleList[a, 4])/np.linalg.norm(v21)
        
        # Find the magnitude of the force acting on atom 3 
        # ||F_atom3|| = || grad_atom3(V) || = || dV/dtheta || * || dtheta / dq_atom3 || 
        # = | ktheta | * | theta - theta0 | * 1/|| q_atom2 - q_atom3 ||
        magnitudeForceOnAtom3 = angleList[a, 3]*abs(theta - angleList[a, 4])/np.linalg.norm(v23)
        
        # Find the direction of the force acting on atom 1 and normalize 
        directionForceOnAtom1WrtCase1 =  np.cross(v21, np.cross(v23, v21))
        directionForceOnAtom1NormalizedWrtCase1 = directionForceOnAtom1WrtCase1/np.linalg.norm(directionForceOnAtom1WrtCase1)
        '''if currentTime == 14:
            print(v21)
            print(v23)
            print(np.linalg.norm(directionForceOnAtom1WrtCase1))'''
        
        # Find the direction of the force acting on atom 3 and normalize
        directionForceOnAtom3WrtCase1 =  np.cross(v23, np.cross(v21, v23))
        directionForceOnAtom3NormalizedWrtCase1 = directionForceOnAtom3WrtCase1/np.linalg.norm(directionForceOnAtom3WrtCase1)
        
        # Case 1:
        if theta > angleList[a, 4]:
            forces[int(angleList[a, 0])] = forces[int(angleList[a, 0])] + directionForceOnAtom1NormalizedWrtCase1*magnitudeForceOnAtom1
            forces[int(angleList[a, 2])] = forces[int(angleList[a, 2])] + directionForceOnAtom3NormalizedWrtCase1*magnitudeForceOnAtom3
            forces[int(angleList[a, 1])] = forces[int(angleList[a, 1])] - (directionForceOnAtom1NormalizedWrtCase1*magnitudeForceOnAtom1 + directionForceOnAtom3NormalizedWrtCase1*magnitudeForceOnAtom3)
                    
        # Case 2:
        else:
            forces[int(angleList[a, 0])] = forces[int(angleList[a, 0])] - directionForceOnAtom1NormalizedWrtCase1*magnitudeForceOnAtom1
            forces[int(angleList[a, 2])] = forces[int(angleList[a, 2])] - directionForceOnAtom3NormalizedWrtCase1*magnitudeForceOnAtom3
            forces[int(angleList[a, 1])] = forces[int(angleList[a, 1])] + (directionForceOnAtom1NormalizedWrtCase1*magnitudeForceOnAtom1 + directionForceOnAtom3NormalizedWrtCase1*magnitudeForceOnAtom3)
    
    ########### Dihedrals #############
    
    
    ########################################
    ########################################
    ############# TO DO ####################
    ########################################
    ########################################
    
    
    
    ########### Lennard Jones #############
    
    
    # Import Euclidean distance package
    from sklearn.metrics.pairwise import euclidean_distances
    
    #numAtoms = 5
    numAtoms = atomListXYZNow.shape[0]
    LJParameters = getLennartJonesParameters()
    # Distance XYZ
    for atom1 in range(0, numAtoms):
        rVector = floorDistanceVector(atomListXYZNow[atom1, :], atomListXYZNow, boxSize)
        rVector[:atom1] = 0
        closeAtoms = np.where((rVector <= rLJCutOff) & (rVector != 0) & (moleculeNumberList[:, 0] != moleculeNumberList[atom1, 0]))
        closeAtoms = closeAtoms[0]
        for i in range(0, len(closeAtoms)):
            atom2 = closeAtoms[i]
            r = rVector[atom2]
            epsilon = np.sqrt(LJParameters[typeList[atom1, 1]]["epsilon"][moleculeNumberList[atom1, 1]] * LJParameters[typeList[atom2, 1]]["epsilon"][moleculeNumberList[atom2, 1]])
            sigma = 0.5*(LJParameters[typeList[atom1, 1]]["sigma"][moleculeNumberList[atom1, 1]] + LJParameters[typeList[atom2, 1]]["sigma"][moleculeNumberList[atom2, 1]])
            sigma6 = sigma**6
            rMin7 = r**(-7)
            magnitudeLJForce = 24*epsilon*abs(sigma6*rMin7*(2*sigma6*r*rMin7 - 1))
            
            # Add potential energy:
            potentialEnergy += 4*epsilon*(((sigma6*r*rMin7)**2)-(sigma6*r*rMin7))
        
            atom1XYZBox = np.mod(atomListXYZNow[atom1, :], boxSize)
            atom2XYZBox = np.mod(atomListXYZNow[atom2, :], boxSize)
            directionForceWrtAtom1LargeR = atom2XYZBox - atom1XYZBox
            for direction in range(0, 3):
                if atom2XYZBox[direction] - atom1XYZBox[direction] > .5*boxSize:
                    directionForceWrtAtom1LargeR[direction] = -directionForceWrtAtom1LargeR[direction]
            if r < (2**(1/6))*sigma:
                directionForceWrtAtom1LargeR = -directionForceWrtAtom1LargeR
            normalizedDirectionForceWrtAtom1LargeR = directionForceWrtAtom1LargeR/np.linalg.norm(directionForceWrtAtom1LargeR)
            forces[atom1, :] = forces[atom1, :] + magnitudeLJForce*normalizedDirectionForceWrtAtom1LargeR
            forces[atom2, :] = forces[atom2, :] - magnitudeLJForce*normalizedDirectionForceWrtAtom1LargeR
            
         
    return forces, potentialEnergy


def floorDistanceVector(a, b, size):
    rx = abs(a[0] - b[:, 0])
    ry = abs(a[1] - b[:, 1])
    rz = abs(a[2] - b[:, 2])
    rxN = rx - size*np.floor(rx/size + 0.5)
    ryN = ry - size*np.floor(ry/size + 0.5)
    rzN = rz - size*np.floor(rz/size + 0.5)
    dist = np.linalg.norm(np.concatenate((rxN.reshape([len(rxN), 1]), ryN.reshape([len(ryN), 1]), rzN.reshape([len(rzN), 1])), axis = 1), axis = 1)
    
    return dist


def thermostat(v, allMasses, rescale, targetTemperature):
    import numpy as np
    boltzmannConstant = 1.38064852*6.02214086*(10**(-7)) # in angström^2 * amu * fs^-2 * K^-1
    totalNumAtoms = allMasses.shape[0]
    currentTemperature = (2/(3*totalNumAtoms*boltzmannConstant))*sum(.5*allMasses[:,0]*((np.linalg.norm(v, axis=1))**2)) # in K
    
    if rescale == 1:
        rescaledV = np.sqrt(targetTemperature/currentTemperature)*v
    else:
        rescaledV = v
    
    
    return currentTemperature, rescaledV
    
    
    
    
    
    
    
    

