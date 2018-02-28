# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def ReadTrajectory(trajFile):
    """This will read the WHOLE trajectory into memory. This will not be possible
later very large trajectores. I'll let you worry about how to deal with that yourself..."""
    trajectory=[]
    with open(trajFile, "r") as tF:
        line = tF.readline()
        while line is not "":
            #first line is number of atoms
            N = int(line.strip())
            tF.readline().strip() # second line is a comment that we throw away

            q = []
            for i in range(N):
                line = tF.readline().strip().split(" ")
                for c in line[1:]:
                    if c is not "":
                        q.append(float(c))
            trajectory.append(np.array(q))

            line = tF.readline()

    return trajectory, N



t, n = ReadTrajectory("Hydrogen.xyz")

print("trajectory contains {} atoms and {} steps".format(n, len(t)))

d_hists=[]
for s,q in enumerate(t):
    print("Processing step {}".format(s))
    # First I will reshape q into a an (n,3) matrix instead of
    # to begin with q is a vector of length 3*n
    q = q.reshape(n, 3) # compare this to your trajectory
                        # make sure you have the same values
                        # as in the trajectory file

    # Now I will calculate all pair-wise distances:

    # First I will do this the easy way
    # r1 = np.zeros((n,n)) # I will store the pair-wise distances
    #                      # in an n,n matrix

#    print("Start slow loop:")
#    for i in range(n):
#        for j in range(n):
#            r1[i,j] = np.linalg.norm(q[i]-q[j]) # what is 'norm' here?
#    print("Done slow loop")

    # Generally speaking, looping through a np.array
    # as above is VERY slow (much the same as looping matlab matrices)
    # So I will show you a trick that will calculate the same as above
    # much faster...

    print("start faster calculation:")
    dr = q - q[:, np.newaxis] # dr is in R^(nxnx3)
                              # have a careful look at dr in
                              # spyder's variable explorer
                              # What do you think this tensor holds?
                              # what does the np.newaxis mean?

    r2 = np.linalg.norm(dr, axis=2) # what does 'axis' mean here?

    print("done faster calculation")
    # r1 and r2 should be very similar (actually identical)
#    m = np.abs(r1-r2) > 1e-3 # try these operations individually in spyder
                             # and understand what the output is

    print(m.sum()) # what does this mean? Why is this sum interesting?
                   # hint: could it be some kind of norm?

    # You should notice that the same calculation without looping is
    # MUCH faster.

    # You may comment out the slow loop for your calculations.
    # don't forget to also comment out the equality check

    # Now I will calculate the histograms of the distances
    # and store it
    d_hists.append(np.histogram(r2.reshape((n*n))))

# I'm going to plot one of the histograms, you will need more for
# your exercise
x = d_hists[10][1]
x = (x[1:] - x[:1])*0.5
y = d_hists[10][0]
plt.plot(x, y)
plt.xlabel('(nm)')
plt.show()
