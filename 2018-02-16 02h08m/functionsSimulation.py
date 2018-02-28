###################################################################################
################################# FUNCTIONS #######################################
###################################################################################
import numpy as np
import time
from sklearn.metrics.pairwise import euclidean_distances

# Define the structure of different molecules:
def getMoleculeStructure():
    
    # Define the properties of different molecules: masses, bonds, angles, dihedrals, equillibrium length and angles, potential constants, Lennart-Jones parameters, etc. 
    struc =  {"Water": {"atoms": [["O"], ["H"], ["H"]], "masses": np.array([15.9994, 1.0080, 1.0080]), "bonds": np.array([[0, 1], [0, 2]]), "angles": np.array([[1, 0, 2]]), "dihedrals": np.empty([0, 4]), "kb": np.array([[5024.16], [5024.16]]), "r0": np.array([[0.9572], [0.9572]]), "ktheta": np.array([[100*6.2802]]), "theta0": np.array([[104.52*np.pi/180]]), "C1": np.empty([0, 1]), "C2": np.empty([0, 1]), "C3": np.empty([0, 1]), "C4": np.empty([0, 1]), "epsilon": [0.66386, 0, 0], "sigma": [3.15061, 0, 0]}, 
             "Hydrogen": {"atoms": [["H"], ["H"]], "masses": np.array([1.0080, 1.0080]), "bonds": np.array([[0, 1]]), "angles": np.empty([0, 3]), "dihedrals": np.empty([0, 4])}, 
             "Methane": {"atoms": [["C"], ["H"], ["H"], ["H"], ["H"]], "masses": np.array([12.0110, 1.0080, 1.0080, 1.0080, 1.0080]), "bonds": np.array([[0, 1], [0, 2], [0, 3], [0, 4]]), "angles": np.array([[1, 0, 2], [1, 0, 3], [1, 0, 4], [2, 0, 3], [2, 0, 4], [3, 0, 4]]), "dihedrals": np.empty([0, 4])},
             "Ethanol": {"atoms": [["C"], ["H"], ["H"], ["H"], ["C"], ["H"], ["H"], ["O"], ["H"]], "masses": np.array([12.0110, 1.0080, 1.0080, 1.0080, 12.0110, 1.0080, 1.0080, 15.9994, 1.0080]), "bonds": np.array([[0, 1], [0, 2], [0, 3], [0, 4], [4, 5], [4, 6], [4, 7], [7, 8]]), "angles": np.array([[1, 0, 4], [2, 0, 4], [3, 0, 4], [3, 0, 2], [3, 0, 1], [2, 0, 1], [5, 4, 6], [0, 4, 6], [0, 4, 5], [0, 4, 7], [4, 7, 8], [5, 4, 7], [6, 4, 7]]), "dihedrals": np.array([[1, 0, 4, 5], [2, 0, 4, 5], [3, 0, 4, 5], [1, 0, 4, 6], [2, 0, 4, 6], [3, 0, 4, 6], [1, 0, 4, 7], [2, 0, 4, 7], [3, 0, 4, 7], [0, 4, 7, 8], [5, 4, 7, 8], [6, 4, 7, 8]]), "kb": np.array([[2845.12], [2845.12] , [2845.12], [2242.624], [2845.12], [2845.12], [2677.76], [4627.50]]), "r0": np.array([[1.090], [1.090] , [1.090], [1.529], [1.090], [1.090], [1.410], [0.945]]), "ktheta": np.array([[100*2.9288], [100*2.9288] , [100*2.9288], [100*2.76144], [100*2.76144], [100*2.76144], [100*2.76144], [100*3.138], [100*3.138], [100*4.144] , [100*4.6024], [100*2.9288], [100*2.9288]]), "theta0": np.array([[108.5*np.pi/180], [108.5*np.pi/180] , [108.5*np.pi/180], [107.8*np.pi/180], [107.8*np.pi/180], [107.8*np.pi/180], [107.8*np.pi/180], [110.7*np.pi/180], [110.7*np.pi/180], [109.5*np.pi/180] , [108.5*np.pi/180], [109.5*np.pi/180], [109.5*np.pi/180]]), "C1": np.array([[0.62760], [0.62760], [0.62760], [0.62760], [0.62760], [0.62760], [0.97905], [0.97905], [0.97905], [-0.44310], [0.94140], [0.94140]]), "C2": np.array([[1.88280], [1.88280], [1.88280], [1.88280], [1.88280], [1.88280], [2.93716], [2.93716], [2.93716], [3.83255], [2.82420], [2.82420]]), "C3": np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0.72801], [0], [0]]), "C4": np.array([[-3.91622], [-3.91622], [-3.91622], [-3.91622], [-3.91622], [-3.91622], [-3.91622], [-3.91622], [-3.91622], [-4.11705], [-3.76560], [-3.76560]]), "epsilon": [0.276144, 0.125520, 0.125520, 0.125520, 0.276144, 0.125520, 0.125520, 0.711280, 0], "sigma": [3.5, 2.5, 2.5, 2.5, 3.5, 2.5, 2.5, 3.12, 0]}
             }
    
    return struc


def getLennartJonesCrossSigma(allTypes, allMoleculeNumbers):
    
    LJsigma = np.zeros([allTypes.shape[0], allTypes.shape[0]])
    struc = getMoleculeStructure()
    
    for i in range(0, allTypes.shape[0]):
        for j in range(0, allTypes.shape[0]):
            LJsigma[i, j] = 0.5*(struc[allTypes[i, 1]]["sigma"][allMoleculeNumbers[i, 1]] + struc[allTypes[j, 1]]["sigma"][allMoleculeNumbers[j, 1]])
    
    return LJsigma

def getLennartJonesCrossEpsilon(allTypes, allMoleculeNumbers):
    
    LJepsilon = np.zeros([allTypes.shape[0], allTypes.shape[0]])
    struc = getMoleculeStructure()
    
    for i in range(0, allTypes.shape[0]):
        for j in range(0, allTypes.shape[0]):
            LJepsilon[i, j] = (1*(allMoleculeNumbers[i,0] != allMoleculeNumbers[j,0]))*np.sqrt(struc[allTypes[i, 1]]["epsilon"][allMoleculeNumbers[i, 1]] * struc[allTypes[j, 1]]["epsilon"][allMoleculeNumbers[j, 1]])
    
    return LJepsilon


# Count atoms/bonds/angles/dihedrals (A) in molecule B
def countIn(A, B):
    struc = getMoleculeStructure()
    return len(struc[B][A])


# Create an initial configuration of the mixture in the prescribed box:
def initializeConfiguration(totalNumMolecules, percentageEthanol, boxSize):
    
    # Ethanol parameters:
    numMoleculesEthanol = int(np.round(totalNumMolecules*percentageEthanol/100))
    numAtomsPerMoleculeEthanol = countIn("atoms", "Ethanol")
    numBondsPerMoleculeEthanol = countIn("bonds", "Ethanol")
    numAnglesPerMoleculeEthanol = countIn("angles", "Ethanol")
    numDihedralsPerMoleculeEthanol = countIn("dihedrals", "Ethanol")
    # Water parameters:
    numMoleculesWater = totalNumMolecules - numMoleculesEthanol
    numAtomsPerMoleculeWater = countIn("atoms", "Water")
    numBondsPerMoleculeWater = countIn("bonds", "Water")
    numAnglesPerMoleculeWater = countIn("angles", "Water")
    numDihedralsPerMoleculeWater = countIn("dihedrals", "Water")

    # Calculate useful information:
    totalNumWaterBonds = numMoleculesWater*numBondsPerMoleculeWater
    totalNumWaterAngles = numMoleculesWater*numAnglesPerMoleculeWater
    totalNumWaterDihedrals = numMoleculesWater*numDihedralsPerMoleculeWater
    totalNumWaterAtoms = numMoleculesWater*numAtomsPerMoleculeWater
    totalNumEthanolBonds = numMoleculesEthanol*numBondsPerMoleculeEthanol
    totalNumEthanolAngles = numMoleculesEthanol*numAnglesPerMoleculeEthanol
    totalNumEthanolDihedrals = numMoleculesEthanol*numDihedralsPerMoleculeEthanol
    totalNumEthanolAtoms = numMoleculesEthanol*numAtomsPerMoleculeEthanol
    totalNumBonds = totalNumWaterBonds + totalNumEthanolBonds
    totalNumAngles = totalNumWaterAngles + totalNumEthanolAngles
    totalNumDihedrals = totalNumWaterDihedrals + totalNumEthanolDihedrals
    totalNumAtoms = numMoleculesWater*numAtomsPerMoleculeWater + numMoleculesEthanol*numAtomsPerMoleculeEthanol
        
    # Create empty arrays with all the bonds, angles, dihedrals, types, masses, numbers. Also useful constants are included
    allBonds = np.zeros([totalNumBonds, 4]) # Indices of 2 atoms in the bond in first 2 columns, kb in 3rd column, r0 in 4th column
    allAngles = np.zeros([totalNumAngles, 5]) # Indices of 3 atoms in the angle in first 3 columns, ktheta in 4th column, theta0 in 5th column
    allDihedrals = np.zeros([totalNumDihedrals, 8]) # Indices of 4 atoms in the dihedral in first 4 columns, C1 in 5th column, C2 in 6th column, C3 in 7th column and C4 in 8th column
    allMasses = np.zeros([totalNumAtoms, 1]) # Mass in amu in first column
    allTypes = np.empty([totalNumAtoms, 2], dtype="<U7") # Atom letter in first column, molecule name in second column
    allMoleculeNumbers = np.zeros([totalNumAtoms, 2], dtype = int) # Molecule number in first column, atom number within molecule in second column
    
    # Set a seed in order to obtain same results:
    np.random.seed(0)
    
    # Get a vector of molecule numbers which are going to be ethanol:
    ethanolPositions = np.random.choice(totalNumMolecules, numMoleculesEthanol, replace = False)
    
    # Create a vector of all molecules and there molecule name which is assigned
    moleculeVector = np.empty([totalNumMolecules, 1], dtype = "<U7")
    moleculeVector[:, 0] = "Water"
    moleculeVector[ethanolPositions, 0] = "Ethanol"
    repeatVector = np.zeros([totalNumMolecules, 1], dtype = int)
    repeatVector[:, 0] = countIn("atoms", "Water")
    repeatVector[ethanolPositions, 0] = countIn("atoms", "Ethanol")
    
    # Fill the second column of allTypes with the molecule names (per atom: "Water" or "Ethanol")
    allTypes[:, 1] = np.repeat(moleculeVector[:,0], repeatVector[:,0])
    
    # Initialize indices to use in the for loop:        
    currentBondIndex = 0
    currentAngleIndex = 0
    currentDihedralIndex = 0
    currentAtomIndex = 0
    
    # Get the structure of molecules:
    structure = getMoleculeStructure()

    # Iteratre over all molecules:
    for molecule in range(0, totalNumMolecules):
        
        # Which molecule do we have? Water or ethanol:
        moleculeType = moleculeVector[molecule, 0]
        
        # How many bonds, angles, dihedrals, atoms are in such a molecule?:
        bondsInType = countIn("bonds", moleculeType)
        anglesInType = countIn("angles", moleculeType)
        dihedralsInType = countIn("dihedrals", moleculeType)
        atomsInType = countIn("atoms", moleculeType)
        
        # Fill the list of bonds, angles, dihedrals, masses, types and molecule numbers with the right information:
        allBonds[currentBondIndex:(currentBondIndex + bondsInType),:] = np.concatenate((structure[moleculeType]["bonds"] + currentAtomIndex, structure[moleculeType]["kb"], structure[moleculeType]["r0"]), axis = 1) # Indices of atoms in first 2 columns, kb in 3rd column, r0 in 4th column
        allAngles[currentAngleIndex:(currentAngleIndex + anglesInType),:] = np.concatenate((structure[moleculeType]["angles"] + currentAtomIndex, structure[moleculeType]["ktheta"], structure[moleculeType]["theta0"]), axis = 1) # Indices of atoms in first 3 columns, ktheta in 4th column, theta0 in 5th column
        allDihedrals[currentDihedralIndex:(currentDihedralIndex + dihedralsInType), :] = np.concatenate((structure[moleculeType]["dihedrals"] + currentAtomIndex, structure[moleculeType]["C1"], structure[moleculeType]["C2"], structure[moleculeType]["C3"], structure[moleculeType]["C4"]), axis = 1)
        allMasses[currentAtomIndex:(currentAtomIndex + atomsInType),:] = structure[moleculeType]["masses"].reshape(countIn("atoms", moleculeType), 1)
        allTypes[currentAtomIndex:(currentAtomIndex + atomsInType),0] = np.concatenate((structure[moleculeType]["atoms"], structure[moleculeType]["atoms"]), axis = 1)[:,0]#np.concatenate((structure["Water"]["atoms"], np.repeat("Water", numAtomsPerMoleculeWater).reshape(numAtomsPerMoleculeWater, 1)), axis=1)#.reshape(numAtomsPerMoleculeWater, 1)
        allMoleculeNumbers[currentAtomIndex:(currentAtomIndex + atomsInType),:] = np.transpose(np.array([atomsInType*[molecule], [x for x in range(0, atomsInType)]]))#np.array([[molecule, 0], [molecule, 1], [molecule, 2]])
        
        # Increment the indices:
        currentBondIndex += bondsInType
        currentAngleIndex += anglesInType
        currentAtomIndex += atomsInType
        currentDihedralIndex += dihedralsInType
        
    
    # How many molecules fit in one of the three directions: (e.g. if totalNumMolecules = 15**3, we have 15 molecules in one direction (we are filling a 3D grid))
    numMoleculeOneDirec = int(np.round(totalNumMolecules**(1/3)))
    
    # Define the 3D grid:
    grid = np.meshgrid([x for x in range(0, numMoleculeOneDirec)], [x for x in range(0, numMoleculeOneDirec)], [x for x in range(0, numMoleculeOneDirec)])
    gridVectors = np.concatenate((grid[0].reshape(totalNumMolecules, 1), grid[1].reshape(totalNumMolecules, 1), grid[2].reshape(totalNumMolecules, 1)), axis = 1)
    
    # Initialize a water and a ethanol molecule
    basicWater = np.array([[0, 0.5, 0.6], [0.6, 0.9, 1.2], [0.5, 0, 0]])
    basicEthanol = np.array([[-1.683, -0.523, 1.084], [-1.689, -1.638, 1.101], [-1.171, -0.174, 2.011], [-2.738, -0.167, 1.117], [-0.968, -0.008, -0.167], [0.094, -0.344, -0.200], [-1.490, -0.319, -1.102], [-0.953, 1.395, -0.142], [-1.842, 1.688, -0.250]]) + np.array([2.738, 1.638, 1.102])
    
    # Put the first molecule in the first grid point:
    if moleculeVector[0] == "Water":
        XYZinitial = basicWater + gridVectors[0]
    else:
        XYZinitial = basicEthanol + gridVectors[0]
    
    # Put all the other molecules in the next grid points:
    for m in range(0, totalNumMolecules-1):
        if moleculeVector[m+1] == "Water":
            XYZinitial = np.concatenate((XYZinitial, basicWater + (1/numMoleculeOneDirec)*boxSize*gridVectors[m + 1]))
        else:
            XYZinitial = np.concatenate((XYZinitial, basicEthanol + (1/numMoleculeOneDirec)*boxSize*gridVectors[m + 1]))
        
    # Return all initializations:
    return XYZinitial, allBonds, allAngles, allDihedrals, allMasses, allTypes, allMoleculeNumbers

# Initialize the velocities
def initializeV(allMoleculeNumbers, meanV, stDevV):
    
    # How many molecules do we have?
    totalNumMolecules = max(allMoleculeNumbers[:, 0])+1
    
    # Find how many time the same velocity should be replicated (either 3 times for water or 9 times for ethanol): each atom within a molecule should have the same initial velocity:
    replications = np.diff(np.append(np.append([-1], np.where((np.diff(allMoleculeNumbers[:,0]) != 0))), [len(allMoleculeNumbers[:,0])-1]))
    
    # Set a seed to get the same answers everytime:
    np.random.seed(1)
    
    # Create random velocities per molecule, v is a 3-dimensional normal variable with mean meanV and standard deviation of stDevV and correlation of 0 between the three dimensions 
    v = np.repeat(np.random.normal(meanV, stDevV, [totalNumMolecules, 3]), replications, axis=0)
    
    # Return the initial velocity vector:
    return v


# Create a calculateForces function:
def calculateForcesEnergy(atomListXYZNow, bondList, angleList, dihedralList, typeList, moleculeNumberList, boxSize, rLJCutOff, LJsigma, LJepsilon, LJsigma6, LJsigma12): # add dihedralList
    '''
    atomListXYZNow, bondList, angleList, dihedralList, typeList, moleculeNumberList = XYZ, allBonds, allAngles, allDihedrals, allTypes, allMoleculeNumbers
    
    '''
    # Initialize the force vector and the potential energy number:    
    forces = np.zeros(atomListXYZNow.shape)
    potentialEnergy = 0
    
    
    ########### Forces for bond potentials #############
    codeTimer = time.time()

    # Calculate all the forces resulting from the bond potential within the molecules, iterate for all bonds in a for-loop:
    for b in range(0, bondList.shape[0]):
        
        # Calculate distance between the two atoms
        r = np.linalg.norm(atomListXYZNow[int(bondList[b, 0])] - atomListXYZNow[int(bondList[b, 1])])
         
        # Bond potential: V(r) = 0.5*k_b*(r-r0)^2 
        
        # Increase potential energy:
        potentialEnergy += 0.5*bondList[b, 2]*((r-bondList[b, 3])**2)
        
        # Structure: atom0 ------- atom1
        
        # Find the magnitude of the force:
        # r = sqrt((qx_1 - qx_0)^2 + (qy_1 - qy_0)^2 + (qz_1 - qz_0)^2) 
        # V = 1/2 * kb * (r - r0)^2 
        # Force wrt atom 0: F_0 = -grad(V) = kb * (r-r0)/r * (q_0-q_1)
        # Force wrt atom 1: F_1 = - kb * (r-r0)/r * (q_0-q_1)
        # Magnitude ||F|| = |kb|*|r-r0| * ( ||q_0 - q_1||/r ) = |kb|*|r-r0|
        magnitudeForce = bondList[b, 2]*abs(r-bondList[b, 3])
        
        # Case 1: r > r0
        # Case 2: r <= r0
        
        # Find force direction with respect to atom 0 and case 1 r > r0: q_1 - q_0 
        # Normalize: q_1 - q_0 / r
        # Calculate Normalized force direction * magnitude with respect to atom 0 and the case that r > r0
        forceVectorAtom0Case1 = (atomListXYZNow[int(bondList[b, 1])] - atomListXYZNow[int(bondList[b, 0])])*magnitudeForce/r
        
        # For atom 1 this vector is in the opposite direction: forceVectorAtom1Case1 = -forceVectorAtom0Case1
        # For case 2 we clearly have: forceVectorAtom0Case2 = -forceVectorAtom0Case1 and forceVectorAtom1Case2 = forceVectorAtom0Case1
        
        correctSign = np.sign(r - bondList[b, 3])
        forces[int(bondList[b, 0])] = forces[int(bondList[b, 0])] + correctSign*forceVectorAtom0Case1
        forces[int(bondList[b, 1])] = forces[int(bondList[b, 1])] - correctSign*forceVectorAtom0Case1
        '''
        # Case 1: r > r0
        if r > bondList[b, 3]: 
            # Add the right forces to the right atoms:
            forces[int(bondList[b, 0])] = forces[int(bondList[b, 0])] + forceVectorAtom0Case1
            forces[int(bondList[b, 1])] = forces[int(bondList[b, 1])] - forceVectorAtom0Case1
            
        # Case 2: r <= r0
        else:
            # Add the right forces to the right atoms:
            forces[int(bondList[b, 0])] = forces[int(bondList[b, 0])] - forceVectorAtom0Case1
            forces[int(bondList[b, 1])] = forces[int(bondList[b, 1])] + forceVectorAtom0Case1
        '''
    
    print("Calculating bond forces took " + str(time.time() - codeTimer) + " seconds.")
        
    ########### Forces for angle potentials #############

    codeTimer = time.time()
    # Calculate all the forces resulting from the angle potential within the molecules, iterate for all angles in a for-loop:
    for a in range(0, angleList.shape[0]):
    
        # Structure:         atom2
        #                  /     \
        #                 /       \
        #                /       atom3
        #             atom1 
        #
        # with angle theta: angle 123 (< 180 degr = pi rad)
        
        # Bond potential: V(theta) = 0.5*k_theta*(theta-theta0)^2 
        
        atom1 = int(angleList[a, 0])
        atom2 = int(angleList[a, 1])
        atom3 = int(angleList[a, 2])
        
        XYZatom1 = atomListXYZNow[atom1,]
        XYZatom2 = atomListXYZNow[atom2,]
        XYZatom3 = atomListXYZNow[atom3,]

        k_theta = angleList[a, 3]        
        theta0 = angleList[a, 4]
        
        # Vector v21 is the vector from atom 2 to atom 1
        v21 = XYZatom1 - XYZatom2
        rv21 = np.linalg.norm(v21)
        # Vector v23 is the vector from atom 2 to atom 3 
        v23 = XYZatom3 - XYZatom2
        rv23 = np.linalg.norm(v23)
        
        # Find the angle theta
        theta = np.arccos((np.dot(v21, v23))/(rv21 * rv23))
        
        # Increase potential energy:
        potentialEnergy += 0.5*k_theta*((theta - theta0)**2)
        
        # Find the magnitude of the force acting on atom 1 
        # ||F_atom1|| = || grad_atom1(V) || = || dV/dtheta || * || dtheta / dq_atom1 || 
        # = | ktheta | * | theta - theta0 | * 1/| q_atom2 - q_atom1 |
        magnitudeForceOnAtom1 = k_theta*abs(theta - theta0)/rv21
        
        # Find the magnitude of the force acting on atom 3 
        # ||F_atom3|| = || grad_atom3(V) || = || dV/dtheta || * || dtheta / dq_atom3 || 
        # = | ktheta | * | theta - theta0 | * 1/|| q_atom2 - q_atom3 ||
        magnitudeForceOnAtom3 = k_theta*abs(theta - theta0)/rv23
        
        # Case 1: theta > theta0
        # Case 2: theta <= theta0
        
        # Find the direction of the force acting on atom 1 and normalize, for case 1 theta > theta0: force is pointing inwards (to make the angle smaller)
        directionForceOnAtom1WrtCase1 =  np.cross(v21, np.cross(v23, v21))
        directionForceOnAtom1NormalizedWrtCase1 = directionForceOnAtom1WrtCase1/np.linalg.norm(directionForceOnAtom1WrtCase1)
        
        # Find the direction of the force acting on atom 3 and normalize, for case 1 theta > theta0: force is pointing inwards (to make the angle smaller)
        directionForceOnAtom3WrtCase1 =  np.cross(v23, np.cross(v21, v23))
        directionForceOnAtom3NormalizedWrtCase1 = directionForceOnAtom3WrtCase1/np.linalg.norm(directionForceOnAtom3WrtCase1)
        
        # With respect to case 2 the forces are in opposite directions
        
        # Force on atom 2 is minus the force on atom 1 minus the force on atom 3
        correctSign = np.sign(theta - angleList[a, 4])
        forceAtom1 = correctSign*directionForceOnAtom1NormalizedWrtCase1*magnitudeForceOnAtom1
        forceAtom3 = correctSign*directionForceOnAtom3NormalizedWrtCase1*magnitudeForceOnAtom3
        forces[atom1] = forces[atom1] + forceAtom1
        forces[atom3] = forces[atom3] + forceAtom3
        forces[atom2] = forces[atom2] - (forceAtom1 + forceAtom3)
        '''
        # Case 1: theta > theta0
        if theta > angleList[a, 4]:
            # Add the right forces to the right atoms:
            forces[int(angleList[a, 0])] = forces[int(angleList[a, 0])] + directionForceOnAtom1NormalizedWrtCase1*magnitudeForceOnAtom1
            forces[int(angleList[a, 2])] = forces[int(angleList[a, 2])] + directionForceOnAtom3NormalizedWrtCase1*magnitudeForceOnAtom3
            forces[int(angleList[a, 1])] = forces[int(angleList[a, 1])] - (directionForceOnAtom1NormalizedWrtCase1*magnitudeForceOnAtom1 + directionForceOnAtom3NormalizedWrtCase1*magnitudeForceOnAtom3)
                    
        # Case 2: theta <= theta0
        else:
            # Add the right forces to the right atoms:
            forces[int(angleList[a, 0])] = forces[int(angleList[a, 0])] - directionForceOnAtom1NormalizedWrtCase1*magnitudeForceOnAtom1
            forces[int(angleList[a, 2])] = forces[int(angleList[a, 2])] - directionForceOnAtom3NormalizedWrtCase1*magnitudeForceOnAtom3
            forces[int(angleList[a, 1])] = forces[int(angleList[a, 1])] + (directionForceOnAtom1NormalizedWrtCase1*magnitudeForceOnAtom1 + directionForceOnAtom3NormalizedWrtCase1*magnitudeForceOnAtom3)
        '''
    
    print("Calculating angle forces took " + str(time.time() - codeTimer) + " seconds.")
    
    codeTimer = time.time()
    ########### Forces for dihedral potentials #############
    
    # Calculate all the forces resulting from the dihedral potential within the molecules, iterate for all dihedrals in a for-loop:
    for dih in range(0, dihedralList.shape[0]):
        
        # Define atoms:
        atom_a = int(dihedralList[dih, 0])
        atom_b = int(dihedralList[dih, 1])
        atom_c = int(dihedralList[dih, 2])
        atom_d = int(dihedralList[dih, 3])
        
        # Get parameters:
        C1 = dihedralList[dih, 4]
        C2 = dihedralList[dih, 5]
        C3 = dihedralList[dih, 6]
        C4 = dihedralList[dih, 7]
        
        # Dihedral potential: 
        # Let theta be the torsion angle (angle between plane described by atoms a, b, c and the plane described by atoms b, c, d)
        # Let psi = theta - pi
        # V(psi) = 0.5*(C1*(1+cos(psi)) + C2*(1-cos(2*psi)) + C3*(1+cos(3*psi)) + C4*(1-cos(4*psi)))
        
        # Get XYZ:
        XYZ_a = atomListXYZNow[atom_a, :]
        XYZ_b = atomListXYZNow[atom_b, :]
        XYZ_c = atomListXYZNow[atom_c, :]
        XYZ_d = atomListXYZNow[atom_d, :]
        
        # Point 0 is the midpoint of bond b ----- c
        XYZ_0 = 0.5*XYZ_b + 0.5*XYZ_c
        
        # Get vectors from b pointing to a and to c and from c to b and d and from 0 pointing to c:
        v_b_to_a = XYZ_a - XYZ_b
        v_b_to_c = XYZ_c - XYZ_b
        v_c_to_b = XYZ_b - XYZ_c
        v_c_to_d = XYZ_d - XYZ_c
        v_0_to_c = XYZ_c - XYZ_0
        
        # Find normalized normal on plane abc and on plane bcd:
        n_abc = np.cross(v_b_to_a, v_b_to_c)
        n_abc = n_abc/np.linalg.norm(n_abc)
        n_bcd = np.cross(v_c_to_d, v_c_to_b)
        n_bcd = n_bcd/np.linalg.norm(n_bcd)
        
        # Let the vector m be the opposite of the normal of plane abc and normalize it
        m = -n_abc
        m = m/np.linalg.norm(m) 
        
        # Let n be the normal of plane bcd
        n = n_bcd
        
        # Find the dihedral angle by using a more stable version of finding the angle using arctan2:
        theta = np.arctan2(((np.dot(np.cross(m, n), v_c_to_b))/(np.linalg.norm(v_c_to_b))), (np.dot(m, n)))
        
        # Find psi = theta - pi
        psi = theta - np.pi
        
        # Find the angle abc in a----b----c and the angle bcd in b----c----d:
        theta_abc = np.arccos(np.dot(v_b_to_a, v_b_to_c)/(np.linalg.norm(v_b_to_a)*np.linalg.norm(v_b_to_c)))
        theta_bcd = np.arccos(np.dot(v_c_to_b, v_c_to_d)/(np.linalg.norm(v_c_to_b)*np.linalg.norm(v_c_to_d)))
        
        # Find signed force magnitudes:
        part1_of_magnitude = -0.5*(C1*np.sin(psi)-2*C2*np.sin(2*psi)+3*C3*np.sin(3*psi)-4*C4*np.sin(4*psi))
        signed_magnitude_force_a = part1_of_magnitude/(np.sin(theta_abc)*(np.linalg.norm(v_b_to_a)))
        signed_magnitude_force_d = part1_of_magnitude/(np.sin(theta_bcd)*(np.linalg.norm(v_c_to_d)))
        
        # Calculate the forces such that sum of forces is zero and the torque is zero as well:
        force_a = signed_magnitude_force_a*n_abc
        force_d = signed_magnitude_force_d*n_bcd
        force_c = (1/((np.linalg.norm(v_0_to_c))**2))*np.cross(-(np.cross(v_0_to_c, force_d) + 0.5*np.cross(v_c_to_d, force_d) + 0.5*np.cross(v_b_to_a, force_a)), v_0_to_c)
        force_b = -force_a - force_d - force_c
        
        # Add the right forces to the right atoms:
        forces[atom_a, :] = forces[atom_a, :] + force_a
        forces[atom_b, :] = forces[atom_b, :] + force_b
        forces[atom_c, :] = forces[atom_c, :] + force_c
        forces[atom_d, :] = forces[atom_d, :] + force_d
        
        # Increase potential energy:        
        potentialEnergy += 0.5*(C1*(1+np.cos(psi))+C2*(1-np.cos(2*psi))+C3*(1+np.cos(3*psi))+C4*(1-np.cos(4*psi)))
        
    
    
    print("Calculating dihedral forces took " + str(time.time() - codeTimer) + " seconds.")
    
    codeTimer = time.time()
    ########### Forces for Lennart-Jones potentials #############
    
    
    # Import Euclidean distance package
    #from sklearn.metrics.pairwise import euclidean_distances
    
    # Get the molecule structure
    struc = getMoleculeStructure()
    
    # Which not to incorporate:
    #noLJ = np.where(((typeList[:,1] == "Water") & ((moleculeNumberList[:,1] == 1)+(moleculeNumberList[:,1] == 2))) + ((typeList[:,1] == "Ethanol") & (moleculeNumberList[:,1] == 8)))    
    
    pairwiseDistances = np.zeros([atomListXYZNow.shape[0], atomListXYZNow.shape[0]])#euclidean_distances(atomListXYZNow, atomListXYZNow)
    #pairwiseDistances = floorDistanceVector(atomListXYZNow, boxSize)
    for i in range(0, pairwiseDistances.shape[0]):
        pairwiseDistances[i,(i+1):] = floorDistanceVectorOld(atomListXYZNow[i, :], atomListXYZNow[(i+1):, :], boxSize)
        #if sum(abs(pairwiseDistances[i,:] - test[i,:]) < 10**(-6)) != pairwiseDistances.shape[0]:
        #    print("WWWWRRROOONNNGGGG!!!!!!")
        #    print(str(sum(abs(pairwiseDistances[i,:] - test[i,:]) < 10**(-6))))
        #    print(str(pairwiseDistances.shape[0]))
        #pairwiseDistances[i, (moleculeNumberList[:,0] == moleculeNumberList[i,0])] = 0
        #pairwiseDistances[i, :i] = 0
    #pairwiseDistances[:, noLJ] = 0
    #pairwiseDistances[noLJ, :] = 0
    
    pairwiseDistances = pairwiseDistances*(1*(LJepsilon > 0))
    
    print("Calculating Lennart-Jones forces part 1 took " + str(time.time() - codeTimer) + " seconds.")
    
    LJList = np.where((pairwiseDistances < rLJCutOff) & (pairwiseDistances != 0))
    LJList = np.concatenate((LJList[0].reshape([LJList[0].shape[0], 1]), LJList[1].reshape([LJList[1].shape[0], 1])), axis=1)
    
    print(LJList.shape[0])
    
    codeTimer = time.time()
    '''
    r = pairwiseDistances[LJList[:,0], LJList[:,1]]
    epsilon = LJepsilon[LJList[:,0], LJList[:,1]]
    sigma = LJsigma[LJList[:,0], LJList[:,1]]
    sigma6 = LJsigma6[LJList[:,0], LJList[:,1]]
    sigma12 = LJsigma12[LJList[:,0], LJList[:,1]]
    rMin7 = r**(-7)
    rMin6 = r*rMin7
    
    # Find magnitude of the force:(note that ||grad(V)|| = ||dV/dr||*||dr/dq|| = ||dV/dr||*1)
    magnitudeLJForce = 24*epsilon*abs(rMin7*(2*sigma12*rMin6 - sigma6))
    
    # Add potential energy:
    potentialEnergy += sum(4*epsilon*((sigma12*((rMin6)**2))-(sigma6*rMin6)))
    
    # Put the atoms back in the box (they are moving free through space):
    atom1XYZBox = np.mod(atomListXYZNow[LJList[:,0], :], boxSize)
    atom2XYZBox = np.mod(atomListXYZNow[LJList[:,1], :], boxSize)
    directionForceWrtAtom1LargeR = atom2XYZBox - atom1XYZBox
    
    # Find the direction of the force, take into account the boundary conditions
    directionForceWrtAtom1LargeR = directionForceWrtAtom1LargeR - boxSize*(2*(atom2XYZBox - atom1XYZBox > 0) - 1)*(abs(atom2XYZBox - atom1XYZBox) > .5*boxSize)
    
    # Force are opposite if atoms are too close:
    #if r < (2**(1/6))*sigma:
    #    directionForceWrtAtom1LargeR = -directionForceWrtAtom1LargeR
    
    # Normalize:
    normalizedDirectionForceWrtAtom1LargeR = directionForceWrtAtom1LargeR/np.linalg.norm(directionForceWrtAtom1LargeR, axis=1).reshape(-1, 1)
    
    # Add forces:
    correctSign = np.sign(r - (2**(1/6))*sigma)
    forceAtom1Vec = correctSign.reshape(-1, 1)*magnitudeLJForce.reshape(-1, 1)*normalizedDirectionForceWrtAtom1LargeR
    forces[LJList[:,0], :] = forces[LJList[:,0], :] + forceAtom1Vec
    forces[LJList[:,1], :] = forces[LJList[:,1], :] - forceAtom1Vec
    
    '''
    
    atom1 = LJList[:, 0]
    atom2 = LJList[:, 1]
    
    r = pairwiseDistances[atom1, atom2]
    
    epsilon = LJepsilon[atom1, atom2]
    sigma = LJsigma[atom1, atom2]
    sigma6 = LJsigma6[atom1, atom2]
    sigma12 = LJsigma12[atom1, atom2]
    rMin7 = r**(-7)
    rMin6 = r*rMin7
    
    magnitudeLJForce = 24*epsilon*abs(rMin7*(2*sigma12*rMin6 - sigma6))
    
    potentialEnergy += sum(4*epsilon*((sigma12*((rMin6)**2))-(sigma6*rMin6)))
    
    atom1XYZBox = np.mod(atomListXYZNow[atom1, :], boxSize)
    atom2XYZBox = np.mod(atomListXYZNow[atom2, :], boxSize)
    directionForceWrtAtom1LargeR = atom2XYZBox - atom1XYZBox
    
    directionForceWrtAtom1LargeR = directionForceWrtAtom1LargeR - boxSize*(np.sign(atom2XYZBox - atom1XYZBox))*(abs(atom2XYZBox - atom1XYZBox) > .5*boxSize)
    
    # Normalize:
    normalizedDirectionForceWrtAtom1LargeR = directionForceWrtAtom1LargeR/(np.linalg.norm(directionForceWrtAtom1LargeR, axis=1).reshape([directionForceWrtAtom1LargeR.shape[0], 1]))
    
    
    # Add forces:
    correctSign = np.sign(r - (2**(1/6))*sigma)
    forceAtom1Vec = correctSign.reshape([correctSign.shape[0], 1])*magnitudeLJForce.reshape([correctSign.shape[0], 1])*normalizedDirectionForceWrtAtom1LargeR
    #if sum(abs(forceAtom1 - forceAtom1Vec[lj, :]) < 10**(-10)) != 3:
    #    print("HO!!!!!!!!")
    for lj in range(0, LJList.shape[0]):
        forces[LJList[lj, 0], :] = forces[LJList[lj, 0], :] + forceAtom1Vec[lj, :]
        forces[LJList[lj, 1], :] = forces[LJList[lj, 1], :] - forceAtom1Vec[lj, :]
    #forces[atom1, :] = forces[atom1, :] + forceAtom1Vec # THIS DOES NOT WORK FOR SOME REASON !??!!!??!
    #forces[atom2, :] = forces[atom2, :] - forceAtom1Vec # THIS DOES NOT WORK FOR SOME REASON !??!!!??!
    
    '''
    for lj in range(0, LJList.shape[0]):
        atom1 = LJList[lj, 0]
        atom2 = LJList[lj, 1]
        # Distance between atom1 and atom2:
        r = pairwiseDistances[atom1, atom2]
        
        
        # Find parameters and powers:
        #epsilon = np.sqrt(struc[typeList[atom1, 1]]["epsilon"][moleculeNumberList[atom1, 1]] * struc[typeList[atom2, 1]]["epsilon"][moleculeNumberList[atom2, 1]])
        #sigma = 0.5*(struc[typeList[atom1, 1]]["sigma"][moleculeNumberList[atom1, 1]] + struc[typeList[atom2, 1]]["sigma"][moleculeNumberList[atom2, 1]])
        #sigma6 = sigma**6
        epsilon = LJepsilon[atom1, atom2]
        sigma = LJsigma[atom1, atom2]
        sigma6 = LJsigma6[atom1, atom2]
        sigma12 = LJsigma12[atom1, atom2]
        rMin7 = r**(-7)
        rMin6 = r*rMin7
        
        # Find magnitude of the force:(note that ||grad(V)|| = ||dV/dr||*||dr/dq|| = ||dV/dr||*1)
        magnitudeLJForce = 24*epsilon*abs(rMin7*(2*sigma12*rMin6 - sigma6))
        
        # Add potential energy:
        #potentialEnergy += 4*epsilon*((sigma12*((rMin6)**2))-(sigma6*rMin6))
        
        # Put the atoms back in the box (they are moving free through space):
        atom1XYZBox = np.mod(atomListXYZNow[atom1, :], boxSize)
        atom2XYZBox = np.mod(atomListXYZNow[atom2, :], boxSize)
        directionForceWrtAtom1LargeR = atom2XYZBox - atom1XYZBox
        
        # Find the direction of the force, take into account the boundary conditions
        directionForceWrtAtom1LargeR = directionForceWrtAtom1LargeR - boxSize*(2*(atom2XYZBox - atom1XYZBox > 0) - 1)*(abs(atom2XYZBox - atom1XYZBox) > .5*boxSize)
        
        # Force are opposite if atoms are too close:
        #if r < (2**(1/6))*sigma:
        #    directionForceWrtAtom1LargeR = -directionForceWrtAtom1LargeR
        
        # Normalize:
        normalizedDirectionForceWrtAtom1LargeR = directionForceWrtAtom1LargeR/np.linalg.norm(directionForceWrtAtom1LargeR)
        
        # Add forces:
        correctSign = np.sign(r - (2**(1/6))*sigma)
        forceAtom1 = correctSign*magnitudeLJForce*normalizedDirectionForceWrtAtom1LargeR
        if sum(abs(forceAtom1 - forceAtom1Vec[lj, :]) < 10**(-9)) != 3:
            print("HO!!!!!!!!")
        forces[atom1, :] = forces[atom1, :] + forceAtom1
        forces[atom2, :] = forces[atom2, :] - forceAtom1
        
        #print(str(sum(abs(forceAtom1 - forceAtom1Vec[lj,:]) < 10**(-9))))
        '''
        
        
    
    print("Calculating Lennart-Jones forces part 2 took " + str(time.time() - codeTimer) + " seconds.")
    
    
    
    
    
    '''
    # Calculate all the forces resulting from the Lennart-Jones potential for non-bonded interactions between different molecules, iterate for all atoms in a for-loop:
    for atom1 in range(0, atomListXYZNow.shape[0]):
        
        #codeTimer = time.time()
        
        # Find the distances (taking care of the box) to the other atoms:
        #rVector = floorDistanceVector(atomListXYZNow[atom1, :], atomListXYZNow, boxSize)
        rVector = floorDistanceVector(atomListXYZNow[atom1, :], atomListXYZNow[(atom1+1):,], boxSize)
        
        # All atoms before atom1 have already been incorporated, don't incorporate it twice!
        #rVector[:atom1] = 0
        
        # Find all atoms which interact (they should not be farther than rLJCutOff, should not be already considere and should not belong to the same molecule)
        #closeAtoms = np.where((rVector <= rLJCutOff) & (rVector != 0) & (moleculeNumberList[:, 0] != moleculeNumberList[atom1, 0]))
        #closeAtoms = closeAtoms[0]
        
        closeAtoms = np.where((rVector <= rLJCutOff) & (moleculeNumberList[(atom1+1):, 0] != moleculeNumberList[atom1, 0]))
        closeAtoms = closeAtoms[0] + atom1 + 1
        #closeAtoms = np.where((moleculeNumberList[(atom1+1):, 0] != moleculeNumberList[atom1, 0])) + atom1 + 1
        
        # For each of these close atoms to atom1: find the potentials and force ina for-loop:
        for i in range(0, len(closeAtoms)):
            
            # Select atom2:
            atom2 = closeAtoms[i]
            
            # Distance between atom1 and atom2:
            r = rVector[atom2 - atom1 - 1]
            
            # Find parameters and powers:
            epsilon = np.sqrt(struc[typeList[atom1, 1]]["epsilon"][moleculeNumberList[atom1, 1]] * struc[typeList[atom2, 1]]["epsilon"][moleculeNumberList[atom2, 1]])
            sigma = 0.5*(struc[typeList[atom1, 1]]["sigma"][moleculeNumberList[atom1, 1]] + struc[typeList[atom2, 1]]["sigma"][moleculeNumberList[atom2, 1]])
            sigma6 = sigma**6
            rMin7 = r**(-7)
            
            # Find magnitude of the force:(note that ||grad(V)|| = ||dV/dr||*||dr/dq|| = ||dV/dr||*1)
            magnitudeLJForce = 24*epsilon*abs(sigma6*rMin7*(2*sigma6*r*rMin7 - 1))
            
            # Add potential energy:
            potentialEnergy += 4*epsilon*(((sigma6*r*rMin7)**2)-(sigma6*r*rMin7))
                  
            # Put the atoms back in the box (they are moving free through space):
            atom1XYZBox = np.mod(atomListXYZNow[atom1, :], boxSize)
            atom2XYZBox = np.mod(atomListXYZNow[atom2, :], boxSize)
            directionForceWrtAtom1LargeR = atom2XYZBox - atom1XYZBox
            
            # Find the direction of the force, take into account the boundary conditions
            directionForceWrtAtom1LargeR = directionForceWrtAtom1LargeR - boxSize*(2*(atom2XYZBox - atom1XYZBox > 0) - 1)*(abs(atom2XYZBox - atom1XYZBox) > .5*boxSize)
                    
            # Force are opposite if atoms are too close:
            #if r < (2**(1/6))*sigma:
            #    directionForceWrtAtom1LargeR = -directionForceWrtAtom1LargeR
                
            # Normalize:
            normalizedDirectionForceWrtAtom1LargeR = directionForceWrtAtom1LargeR/np.linalg.norm(directionForceWrtAtom1LargeR)
            
            # Add forces:
            correctSign = np.sign(r - (2**(1/6))*sigma)
            forceAtom1 = correctSign*magnitudeLJForce*normalizedDirectionForceWrtAtom1LargeR
            forces[atom1, :] = forces[atom1, :] + forceAtom1
            forces[atom2, :] = forces[atom2, :] - forceAtom1
            
        #print("Calculating LJ forces for atom " + str(atom1) + " took " + str(time.time() - codeTimer) + " seconds.")
            
    '''
    #print("Calculating Lennart-Jones forces took " + str(time.time() - codeTimer) + " seconds.")
    
    # Return forces and potential energy:
    return forces, potentialEnergy


# Special function for distances when taking into account the boundary conditions:
def floorDistanceVector(b, size):#(a, b, size):
    
    # Put back in box:
    #a = np.mod(a, size)
    #b = np.mod(b, size)
    c = np.mod(b, size)
    
    # Get absolute differences of components:
    #rx = abs(a[0] - b[:, 0])
    #ry = abs(a[1] - b[:, 1])
    #rz = abs(a[2] - b[:, 2])
    
    rx = euclidean_distances(c[:,0].reshape(-1, 1))
    ry = euclidean_distances(c[:,1].reshape(-1, 1))
    rz = euclidean_distances(c[:,2].reshape(-1, 1))
    
    # Take into account the box:
    rxN = rx - size*np.floor(rx/size + 0.5)
    ryN = ry - size*np.floor(ry/size + 0.5)
    rzN = rz - size*np.floor(rz/size + 0.5)
    
    # Calculate the distances:
    #dist = np.linalg.norm(np.concatenate((rxN.reshape([len(rxN), 1]), ryN.reshape([len(ryN), 1]), rzN.reshape([len(rzN), 1])), axis = 1), axis = 1)
    dim = rxN.shape[0]
    d = np.concatenate((rxN.reshape([1, dim, dim]), ryN.reshape([1, dim, dim]), rzN.reshape([1, dim, dim])), axis = 0)
    dist = np.linalg.norm(d, axis = 0)
    
    # Return the distances
    return dist

def floorDistanceVectorOld(a, b, size):#(a, b, size):
    
    # Put back in box:
    a = np.mod(a, size)
    b = np.mod(b, size)
    
    # Get absolute differences of components:
    rx = abs(a[0] - b[:, 0])
    ry = abs(a[1] - b[:, 1])
    rz = abs(a[2] - b[:, 2])
    
    # Take into account the box:
    rxN = rx - size*np.floor(rx/size + 0.5)
    ryN = ry - size*np.floor(ry/size + 0.5)
    rzN = rz - size*np.floor(rz/size + 0.5)
    
    # Calculate the distances:
    dist = np.linalg.norm(np.concatenate((rxN.reshape([len(rxN), 1]), ryN.reshape([len(ryN), 1]), rzN.reshape([len(rzN), 1])), axis = 1), axis = 1)
    
    # Return the distances
    return dist


# Function for the thermostat: find temperature and rescale velocities
def thermostat(v, allMasses, rescale, targetTemperature):
    
    # Define Boltzmann constant:
    boltzmannConstant = 1.38064852*6.02214086*(10**(-7)) # in angström^2 * amu * fs^-2 * K^-1
    
    # Get how many atoms are in the system:
    totalNumAtoms = allMasses.shape[0]
    
    # Find the current temperature:
    currentTemperature = (2/(3*totalNumAtoms*boltzmannConstant))*sum(.5*allMasses[:,0]*((np.linalg.norm(v, axis=1))**2)) # in K
    
    # Rescale if one wants to use the thermostat. Don't rescale if one does not want to:
    if (rescale == 1) & (int(currentTemperature) != 0):
        rescaledV = np.sqrt(targetTemperature/currentTemperature)*v
    else:
        rescaledV = v
    
    # Return the temperature before rescaling and the rescaled velocities
    return currentTemperature, rescaledV
    
    
    
    
    
    
    
    

