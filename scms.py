"""Identify filamentary structure in the dark matter halos from simulations 
   (with periodic boxes) using the SCMS method"""

import argparse
import time
import numpy as np
import pandas as pd

def scms(path, tracers=np.empty(0), min_shift=1.0, dim=3):
    """Run the SCMS algorithm: for each tracer, calculates the Hessian at that
    position (w.r.t. the halos) and shifts based on the two Hessian eigenvectors
    associated with the smaller two eigenvalues (towards density ridges) and
    the mean shift vector (weights nearby density ridges more highly).
    Repeats until shift converges to set value, then repeats on remaining tracers.
    
    Inputs are path to threshold file (from run directory),
    array of tracers (defaults to the halo array)
    SCMS convergence value (default to 1.0 kpc),
    and number of dimensions (defaults to 3)"""

#--------------- Import the retained halo position arrays from threshold step
    threshold = pd.read_csv(path+'threshold.out', sep=',')
    halos = np.array([threshold["x"].values, threshold["y"].values, threshold["z"].values])

#--------------- Check if the tracers differ from the halos and are correct shape
    if tracers.size == 0:
        tracers = np.copy(halos)
    elif tracers.shape[0] != dim:
        tracers = tracers.T

#--------------- Number of halos and number of tracers
    n_tracers = tracers.shape[1] # number of filament tracers
    n_halos = halos.shape[1] # number of halos

#--------------- Import the smoothing length and boxsize
    smoothing = np.loadtxt(path+'params.out')[0]
    boxsize = np.loadtxt(path+'params.out')[1]

#--------------- Run algorithm on n_tracers particles, until shift converges to min_shift
    start = time.time()
    for i in range(n_tracers):
        # create shift array with high value to start while loop
        shift = 100.*np.ones(dim)
        while np.sqrt(np.sum(shift**2)) > min_shift:
            # make a version of the halo positions that is periodic, adjusted to current tracer position
            hp = np.copy(halos) # halos, periodic
            hp[0, np.where(hp[0] - tracers[0, i] > boxsize/2.)] = hp[0, np.where(hp[0] - tracers[0, i] > boxsize/2.)] - boxsize
            hp[0, np.where(hp[0] - tracers[0, i] < -boxsize/2.)] = hp[0, np.where(hp[0] - tracers[0, i] < -boxsize/2.)] + boxsize
            hp[1, np.where(hp[1] - tracers[1, i] > boxsize/2.)] = hp[1, np.where(hp[1] - tracers[1, i] > boxsize/2.)] - boxsize
            hp[1, np.where(hp[1] - tracers[1, i] < -boxsize/2.)] = hp[1, np.where(hp[1] - tracers[1, i] < -boxsize/2.)] + boxsize
            hp[2, np.where(hp[2] - tracers[2, i] > boxsize/2.)] = hp[2, np.where(hp[2] - tracers[2, i] > boxsize/2.)] - boxsize
            hp[2, np.where(hp[2] - tracers[2, i] < -boxsize/2.)] = hp[2, np.where(hp[2] - tracers[2, i] < -boxsize/2.)] + boxsize
            
            # distance between ith tracer and all halos
            xdist = tracers[0, i] - hp[0]
            ydist = tracers[1, i] - hp[1]
            zdist = tracers[2, i] - hp[2]
            
            # calculate the kernel 
            kernel = 1./np.sqrt(2*np.pi)*np.exp(-(xdist**2 + ydist**2 + zdist**2)/(2*smoothing**2))

            # create the Hessian
            hessian = np.zeros([dim, dim])
            hessian[0, 0] = np.sum(kernel*((xdist/smoothing**2)**2 - 1./smoothing**2))
            hessian[1, 1] = np.sum(kernel*((ydist/smoothing**2)**2 - 1./smoothing**2))
            hessian[2, 2] = np.sum(kernel*((zdist/smoothing**2)**2 - 1./smoothing**2))
            hessian[1, 0] = hessian[0, 1] = np.sum(kernel*xdist*ydist/smoothing**4)
            hessian[2, 0] = hessian[0, 2] = np.sum(kernel*xdist*zdist/smoothing**4)
            hessian[2, 1] = hessian[1, 2] = np.sum(kernel*ydist*zdist/smoothing**4)

            # eigenvector matrix from the Hessian (lower 2 eigenvalues)
            evecs = np.linalg.eigh(1./n_halos*hessian)[1].T[0:dim-1]

            # mean shift vector
            ms_vec = np.sum(kernel*hp, axis=1)/np.sum(kernel) - tracers[:, i]

            # calculate the shift to the ith particle and update its position
            shift = np.dot(np.dot(evecs.T, evecs), ms_vec)
            tracers[:, i] = tracers[:, i] + shift
            
            # check and move tracers that are outside the box
            if tracers[0, i] < 0:
                tracers[0, i] = boxsize + tracers[0, i]
            elif tracers[0, i] > boxsize:
                tracers[0, i] = tracers[0, i] - boxsize
            if tracers[1, i] < 0:
                tracers[1, i] = boxsize + tracers[1, i]
            elif tracers[1, i] > boxsize:
                tracers[1, i] = tracers[1, i] - boxsize
            if tracers[2, i] < 0:
                tracers[2, i] = boxsize + tracers[2, i]
            elif tracers[2, i] > boxsize:
                tracers[2, i] = tracers[2, i] - boxsize

        # keep track of percentage of tracers the algorithm has completed and time taken
        if(np.mod(i+1, n_tracers//10) == 0):
            end = time.time()
            print(str(np.int(np.ceil(i/n_tracers*100.)))+"% Completed:", np.round((end-start)/60., 1), "minutes")

#------------- Save the halo & filament locations and masses into a file
    df = pd.DataFrame({'x_filament':tracers[0], 'y_filament':tracers[1], 'z_filament':tracers[2]})
    df.to_csv(path+'scms.out', sep=',', na_rep=-1, index=False)



PARSER = argparse.ArgumentParser()
PARSER.add_argument('path', type=str)
PARSER.add_argument('--tracers', type=float, default=np.empty(0), required=False)
PARSER.add_argument('--min_shift', type=float, default=1.0, required=False)
PARSER.add_argument('--dimensions', type=int, default=3, required=False)
ARGS = PARSER.parse_args()

scms(ARGS.path, ARGS.tracers, ARGS.min_shift, ARGS.dimensions)