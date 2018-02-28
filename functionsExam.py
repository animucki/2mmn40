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
             "Hydrogen": {"atoms": ["H", "H"], "masses": np.array([1.0080, 1.0080]), "bonds": np.array([[0, 1]]), "angles": np.array([]), "dihedrals": np.array([])}, # The hydrogens are 0 and 1
             "Methane": {"atoms": ["C", "H", "H", "H", "H"], "masses": np.array([12.0110, 1.0080, 1.0080, 1.0080, 1.0080]), "bonds": np.array([[0, 1], [0, 2], [0, 3], [0, 4]]), "angles": np.array([[1, 0, 2], [1, 0, 3], [1, 0, 4], [2, 0, 3], [2, 0, 4], [3, 0, 4]]), "dihedrals": np.array([])}, # The carbon is 0, the hydrogens 1, 2, 3, 4
             "Ethanol": {"atoms": [["C"], ["H"], ["H"], ["H"], ["C"], ["H"], ["H"], ["O"], ["H"]], "masses": np.array([12.0110, 1.0080, 1.0080, 1.0080, 12.0110, 1.0080, 1.0080, 15.9994, 1.0080]), "bonds": np.array([[0, 1], [0, 2], [0, 3], [0, 4], [4, 5], [4, 6], [4, 7], [7, 8]]), "angles": np.array([[1, 0, 4], [2, 0, 4], [3, 0, 4], [3, 0, 2], [3, 0, 1], [2, 0, 3], [5, 4, 6], [0, 4, 6], [0, 4, 5], [0, 4, 7], [4, 7, 8], [5, 4, 7], [6, 4, 7]]), "dihedrals": np.array([[1, 0, 4, 5], [2, 0, 4, 5], [3, 0, 4, 5], [1, 0, 4, 6], [2, 0, 4, 6], [3, 0, 4, 6], [1, 0, 4, 7], [2, 0, 4, 7], [3, 0, 4, 7], [0, 4, 7, 8], [5, 4, 7, 8], [6, 4, 7, 8]]), "kb": np.array([[2845.12], [2845.12] , [2845.12], [2242.624], [2845.12], [2845.12], [2677.76], [4627.50]]), "r0": np.array([[1.090], [1.090] , [1.090], [1.529], [1.090], [1.090], [1.410], [0.945]]), "ktheta": np.array([[2.9288], [2.9288] , [2.9288], [2.76144], [2.76144], [2.76144], [2.76144], [3.138], [3.138], [4.144] , [4.6024], [2.9288], [2.9288]]), "theta0": np.array([[108.5*np.pi/180], [108.5*np.pi/180] , [108.5*np.pi/180], [107.8*np.pi/180], [107.8*np.pi/180], [107.8*np.pi/180], [107.8*np.pi/180], [110.7*np.pi/180], [110.7*np.pi/180], [109.5*np.pi/180] , [108.5*np.pi/180], [109.5*np.pi/180], [109.5*np.pi/180]])} # Obvious from definition
             }
    
    return struc



# Count atoms/bonds/angles/dihedrals in molecule B
def countIn(A, B):
    struc = getMoleculeStructure()
    return len(struc[B][A])



# Initialize the positions
def initializeXYZ(numWater, numEthanol):
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
        print(m)
        XYZinitial = np.concatenate((XYZinitial, basicWater + (1/numMoleculeOneDirec)*50*gridVectors[m + 1]))
    
    
    '''
    
    XYZinitial = np.concatenate((np.array([[1.3, 16.2, 16.8], [1.9, 16.6, 17.4], [1.8, 15.7, 16.2], [12.7, 0.5, 6.2], [13.4, 0.1, 6.8], [13.3, 1.2, 5.7]]), basicEthanol), axis = 0)
    
    XYZinitial = np.array([[1.3, 16.2, 16.8], [1.9, 16.6, 17.4], [1.8, 15.7, 16.2]])
    XYZinitial = basicEthanol
    '''
    return XYZinitial



# Initialize the velocities
def initializeV(numWater, numEthanol):
    v = np.zeros([numWater*countIn("atoms", "Water") + numEthanol*countIn("atoms", "Ethanol"), 3])
    return v



# Create a calculateForces function:
def calculateForces(atomListXYZNow, bondList, angleList): # add dihedralList
    
    forces = np.zeros(atomListXYZNow.shape)
    
    # Calculate all the bond forces within the molecules, iterate for all bonds in a for loop:
    for b in range(0, bondList.shape[0]):
        
        # Calculate distance between two atomes
        r = np.linalg.norm(atomListXYZNow[int(bondList[b, 0])] - atomListXYZNow[int(bondList[b, 1])])
                
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
        theta = np.arccos((np.dot(v21, v23))/(np.linalg.norm(v21) * np.linalg.norm(v23)))
        
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
    
    
    return forces



