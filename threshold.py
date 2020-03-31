"""Cut out particles (halos) in underdense regions using a Gaussian kernel density estimator"""

import argparse
import time
import numpy as np
import pandas as pd
import bigfile

def threshold(snapshot, a0=0, dim=3):
    """Calculates KDE for halo positions, retains those in regions with KDE
    greater than the mean and saves into a file. Also saves a file with the
    smoothing length used and the boxsize (for use with scms algorithm).

    Inputs are PIG snapshot path (from run directory),
    smoothing length factor (0 means use default smoothing of 2 Mpc),
    and number of dimensions (defaults to 3)"""

#--------------- Define function to handle the periodicity of the box
    # the input array should be separations between objects
    def perio(sep, boxsize):
        # if the distance between particles is greater than half the boxsize,
        # change to the periodic distance (i.e. "off the map")
        sep[np.where(sep > boxsize/2.)] = -boxsize + sep[np.where(sep > boxsize/2.)]
        # if the distance between particles is less than half the negative boxsize,
        # change to the periodic distance (i.e. "off the map")
        sep[np.where(sep < -boxsize/2.)] = boxsize + sep[np.where(sep < -boxsize/2.)]
        return(sep)

#--------------- Import the snapshot parameters and make halo position arrays
    pig = bigfile.BigFile(snapshot)
    x = pig["FOFGroups/MassCenterPosition"][:][:, 0] # in kpc
    y = pig["FOFGroups/MassCenterPosition"][:][:, 1]
    z = pig["FOFGroups/MassCenterPosition"][:][:, 2]
    n_halos = x.size  # number of halos
    boxsize = pig["Header"].attrs["BoxSize"][0] # boxsize in kpc
    print("Number of Halos:", n_halos)

#--------------- Calculate the (Gaussian) kernel density estimator
    start = time.time()
    if a0 != 0:
        sig = np.min((np.std(x), np.std(y), np.std(z)))
        # calculated smoothing length
        smoothing = a0/((dim+2)**(1./(dim+4)))*n_halos**(-1./(dim+4))*sig
    else:
        smoothing = 2000. # good generic value for 64 Mpc box
    kde = np.zeros(n_halos)
    for i in range(n_halos):
        # distance between ith halo and all halos (squared)
        dist = perio(x[i]-x, boxsize)**2 + perio(y[i]-y, boxsize)**2 + perio(z[i]-z, boxsize)**2
        kde[i] = np.sum(np.exp(-dist/(2.*smoothing**2)))
    end = time.time()
    print("Time for KDE calculation: ", np.round((end-start)/60, 3), "minutes")

#--------------- Cut out particles in the lowest density regions
    tau = np.mean(kde) # cutoff threshold for density
    cuts = np.where(kde > tau)[0] # indices for positions that meet criteria
    n_halos = cuts.size
    print('% Halos Remaining:', np.round(100.*n_halos/x.size, 1))

#--------------- Save retained halo positions and masses into a file
    halos = np.array([x[cuts], y[cuts], z[cuts]])
    masses = pig["FOFGroups/Mass"][:][cuts]

    df1 = pd.DataFrame({'x':halos[0], 'y':halos[1], 'z':halos[2]})
    df2 = pd.DataFrame({'mass':masses})
    dff = pd.concat([df1, df2], ignore_index=False, axis=1)
    dff.to_csv(snapshot+'threshold.out', sep=',', na_rep=-1, index=False)

#--------------- Save smoothing length and boxsize into file
    np.savetxt(snapshot+'params.out', np.array([smoothing, boxsize]))

#--------------- Run the code
PARSER = argparse.ArgumentParser()
PARSER.add_argument('snapshot', type=str)
PARSER.add_argument('--param', type=float, default=0, required=False)
PARSER.add_argument('--dimensions', type=int, default=3, required=False)
ARGS = PARSER.parse_args()

threshold(ARGS.snapshot, ARGS.param, ARGS.dimensions)
