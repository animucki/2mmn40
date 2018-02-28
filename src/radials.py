#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 13:23:53 2018

@author: bartosz
"""

import numpy as np
import matplotlib.pyplot as plt

#Radial distribution functions.
# This script assumes correctly generated xyz files

infilename = 'mixture7norm with numMolecules 343 time 100000 fs dt 2 fs box 3.1 nm percEthanol 13.5 rescale 1 targetTemp 300 K rLJcut 8 nm.xyz'
refElement = 'H'
radialElement = 'H'

# non-bonded interaction approximation parameters
nonBondedOnly = True
clearClose =  8 #int

#Read last snapshot

parameters = infilename.split()

f=open(infilename,"r")
n = int(f.readline())
f.seek(0)

tsteps = int(int(parameters[5])/int(parameters[8]))
lines = (n+2)*(tsteps+1)
targetline = lines-n

size =float(parameters[11])*10 #box size in angstroms

#store data
types = ['']*n
q = np.zeros((n,3))

for l, line in enumerate(f):
    if l%(5000*(n+2))==1:
        print(l, line)
    if l>=targetline:
        sline = line.split()
        #save atom type
        types[l-targetline]=sline[0]
        #save xyz
        for i,elem in enumerate(sline[1:4]):
            q[l-targetline,i] = float(elem)
                
f.close()

# calculate pairwise distances
indicesToDelete = [i for i,x in enumerate(types) if x != radialElement]
qtarget = np.delete(q,indicesToDelete,0)

indicesToDelete = [i for i,x in enumerate(types) if x != refElement]
qref = np.delete(q,indicesToDelete,0)

dr = qtarget - qref[:,np.newaxis]
r = np.linalg.norm(dr,axis=2).flatten()

hist1, bins = np.histogram(r,bins=50,range=(0,12))
hist1[0] =0 #get rid of the self-reference
if nonBondedOnly: #get rid of bonded atoms
    hist1[1:clearClose]=[0]*(clearClose-1)
hist1 = hist1.astype(float)
hist1 /= 4*np.pi*np.linspace(0,12)**2 #normalize wrt to sphere
hist1 = np.nan_to_num(hist1)
hist1 /= np.linalg.norm(hist1) #normalize for density
plt.plot(bins[:-1], hist1)

#save bins and hist1
with open('histogram {}% {}-{}.csv'.format(parameters[14],refElement,radialElement), 'w') as outfile:
    for x,y in zip(bins,hist1):
        outfile.write('{},{}\n'.format(x,y))
        
