"""Separate filaments, assign halos, and calculate length, number and mass"""

import argparse
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt


def periodic(array, boxsize):
    array[np.where(array > boxsize/2.)] = -boxsize + array[np.where(array > boxsize/2.)]
    array[np.where(array < -boxsize/2.)] = boxsize + array[np.where(array < -boxsize/2.)]
    return(array)


def separate(path, cutoff, filaments, smoothing, boxsize):
    """Take in the SCMS filaments and separate the set into individual
    filaments, by assigning each tracer an ID, and return these IDs.

    Inputs are path to snapshot (where it will save the results),
    the cutoff length to distinguish filaments (will depend on redshift),
    the filament tracers (from the SCMS algorithm),
    the smoothing length used in thresholding and SCMS (in kpc),
    and the size of the simulation box (in kpc)"""

#--------------- Separate filaments using a cutoff distance and assigning ids to each tracer
    cutoff = (boxsize/64000.)*cutoff
    x, y, z = filaments[0], filaments[1], filaments[2]
    n_tracers = filaments.shape[1]

    id_num = 1 # initialize the ID number, corresponding to what will be the first filament
    i = 0 # start the index stepper with the first tracer
    current_fil_indices = np.array(i) # the indices that belong to the current ID number filament
    ids = np.zeros(n_tracers, dtype=int) # an array with an id number entry for every tracer

    unassigned = np.arange(x.size) # initialize list of tracers to use in separation (those not yet assigned)

    count = 1 # keep track of how many tracers have been assigned to filaments
    while count != n_tracers:
        dists = np.zeros(n_tracers) # all distances between current tracer and those not yet assigned (n_tracer size to keep indices consistent)
        dists[unassigned] = np.sqrt(periodic(x[i] - x[unassigned], boxsize)**2 + periodic(y[i] - y[unassigned], boxsize)**2 + periodic(z[i] - z[unassigned], boxsize)**2)
        min_dist = dists[np.where(dists != 0)].min()

        if min_dist < cutoff: # if the closest unassigned tracer is close enough
            # add that tracer index to current_fil_indices
            current_fil_indices = np.append(current_fil_indices, np.where(dists == min_dist)[0])

        elif min_dist >= cutoff: # if the closest unassigned tracer is not close enough
            # assign id_num to the ones that were stored in current_fil_indices
            ids[current_fil_indices] = id_num
            # since we are no longer on the same filament, update id_num
            id_num = id_num + 1
            # re-initialize the current_fil_indices array
            current_fil_indices = np.array(np.where(dists == min_dist)[0])

        # remove ith tracer from the unassigned list
        unassigned = np.setdiff1d(unassigned, i)
        # redefine i as the index of the closest tracer
        i = np.where(dists == min_dist)[0]
        # add one to count, since we have assigned one
        count = count + 1
        if count == n_tracers: # in case we end on a filament so it doesn't trigger assignment
            # assign id_num to the ones that were stored in current_fil_indices
            ids[current_fil_indices] = id_num

    # assign the last remaining tracer
    dist = np.sqrt(periodic(x[i] - x, boxsize)**2 + periodic(y[i] - y, boxsize)**2 + periodic(z[i] - z, boxsize)**2)
    min_dist = np.min(dist[np.where(dist != 0)])
    if min_dist < cutoff:
        ids[i] = ids[np.where(dist == min_dist)]
    else:
        ids[i] = id_num

#--------------- Combine filament segments into full filaments
    for i in range(1, np.int(ids.max())+1):
        if np.where(ids == i)[0].size != 0: # only look at id numbers with tracer assigned (this will change as it goes through the loop)

            # find minimum distance between any point on the current filament and all other filaments
            x_current, y_current, z_current = x[np.where(ids == i)[0]], y[np.where(ids == i)[0]], z[np.where(ids == i)[0]]
            x_others, y_others, z_others = x[np.where(ids != i)[0]], y[np.where(ids != i)[0]], z[np.where(ids != i)[0]]

            j = 0 # iteration counter
            min_dist = cutoff + 1
            while min_dist > cutoff and j != x_current.size:
                dist = np.sqrt(periodic(x_current[j] - x_others, boxsize)**2 + periodic(y_current[j] - y_others, boxsize)**2 + periodic(z_current[j] - z_others, boxsize)**2)
                min_dist = np.min(dist) # minimum distance between each tracer on current filament and all other filaments
                min_dist_ind = np.int(np.argsort(dist)[0]) # index corresponding to minimum distance from the jth tracer on the current filament to all other filaments
                j = j + 1

            closest_fil_index = np.where(x == x_others[min_dist_ind])[0] # the index for the same, but corresponding to the entire set of tracers
            if min_dist < cutoff: # if the minimum distance is less than cutoff
                combine = np.max([ids[closest_fil_index], i]) # determine the larger of the two ids
                ids[np.where(ids == i)] = combine #  and assign both filaments this id, combining them
                ids[np.where(ids == ids[closest_fil_index])] = combine

    # 'rename' the id numbers so that they are consecutive
    for i in range(np.unique(ids).size):
        ids[np.where(ids == np.unique(ids)[i])] = i+1

    return(ids)




def get_dists(path, ids, threshold, filaments, masses, smoothing, boxsize):
    """Calculate the lengths of the separated filaments, cut out those that are
    shorter than the smoothing length, assign halos to filaments, save the
    length/mass/number for each filament, and return the halo-filament ids.

    Inputs are path to snapshot (where it will save the results),
    the tracer filament ids from separation,
    the halo positions (post thresholding),
    the filament tracers (from the SCMS algorithm),
    the masses of all the halos (in 10^10 solar masses),
    the smoothing length used in thresholding and SCMS (in kpc),
    and the size of the simulation box (in kpc)"""


#------------- Calculate the lengths of the filaments by stepping along each one, bomber-man style
    length = np.zeros(np.unique(ids).size)
    large_step_fil = np.empty(0, dtype=int)
    for fil_num in np.unique(ids):
        # load up the tracer positions belonging to this filament
        x = filaments[0, np.where(ids == fil_num)[0]]
        y = filaments[1, np.where(ids == fil_num)[0]]
        z = filaments[2, np.where(ids == fil_num)[0]]

        # determine a likely filament-end tracer to start length calculation
        n_nearby = np.zeros(x.size)
        for i in range(x.size):
            count = 0 # will be the number of directions in which each tracer has other tracers within the smoothing distance
            # (end points should have other tracers in fewer directions, generally)
            dx = periodic(x[i] - x, boxsize)
            dy = periodic(y[i] - y, boxsize)
            dz = periodic(z[i] - z, boxsize)
            dist = np.sqrt(dx**2 + dy**2 + dz**2)

            # if there are any tracers within a smoothing distance and in the positive x direction add to count
            if np.where(np.logical_and(dx < 0, dist < smoothing))[0].size != 0:
                count = count + 1
            # same as above, but in negative x direction
            if np.where(np.logical_and(dx > 0, dist < smoothing))[0].size != 0:
                count = count + 1
            # same as above, but in positive y direction
            if np.where(np.logical_and(dy < 0, dist < smoothing))[0].size != 0:
                count = count + 1
            # same as above, but in negative y direction
            if np.where(np.logical_and(dy > 0, dist < smoothing))[0].size != 0:
                count = count + 1
            # same as above, but in positive z direction
            if np.where(np.logical_and(dz < 0, dist < smoothing))[0].size != 0:
                count = count + 1
            # same as above, but in negative z direction
            if np.where(np.logical_and(dz > 0, dist < smoothing))[0].size != 0:
                count = count + 1

            n_nearby[i] = count # store the count for each tracer

        # ------------------------------------------------------------------------
        # initialize arrays and set parameters
        i = np.where(n_nearby == np.min(n_nearby))[0][0] # set the index stepper to the initial point (hopefully, one of the ends)
        points = np.array([i], dtype=int) # initialize the array for indices to use in length calculation
        retained = np.setdiff1d(np.arange(0, x.size, 1, dtype=int), points) # initialize array for retained particles (outside the radius of the i particles)
        remove = np.empty(0, dtype=int) # initialize array to remove indices (inside radius of i particles)
        radius = (boxsize/64000.)*500. # set the radius for removal/retention

        # ------------------------------------------------------------------------
        # while there are still data not removed or added to points (which are used to determine the length)
        while remove.size < x.size:
            # the distance between the current tracer (ith) and all the others in this filament
            dist_all = np.sqrt(periodic(x[i] - x, boxsize)**2 + periodic(y[i] - y, boxsize)**2 + periodic(z[i] - z, boxsize)**2)
            dist_all = np.round(dist_all, 5)

            temp = np.setdiff1d(np.where(dist_all < radius)[0], remove) # array containing the new indices
            # inside the radius (i.e. all indices inside the radius without indices in common with remove
            # from the prior step)
            remove = np.append(remove, temp) # add temp to the prior step remove to create new remove
            retained = np.setdiff1d(retained, remove) # prior step retained without indices from remove
            # (so all the retained indices from previous step, minus the newly removed ones)

            if x[retained].size == 0: # if it has run out of particles (all removed or in points)
                # calculate the distance from current particle to the set from previous step (i.e. temp)
                dist = np.sqrt(periodic(x[i] - x[temp], boxsize)**2 + periodic(y[i] - y[temp], boxsize)**2 + periodic(z[i] - z[temp], boxsize)**2)
                dist = np.round(dist, 5)
                # and add the index corresponding to the max distance -- should be the end point or near enough
                points = np.append(points, np.where(dist_all == dist.max())[0][0])
                # and break out of the if and while
                break

            # the distance between the current particle (ith) and all other retained particles
            distx = periodic(x[i] - x[retained], boxsize)**2
            disty = periodic(y[i] - y[retained], boxsize)**2
            distz = periodic(z[i] - z[retained], boxsize)**2
            dist = np.round(np.sqrt(distx + disty + distz), 5)

            # add the index of the particle closest to current particle, but outside the radius
            points = np.append(points, np.where(dist_all == dist.min())[0][0])
            # step to the particle index corresponding to above
            i = np.where(dist_all == dist.min())[0][0]

        # ------------------------------------------------------------------------
        # calculate the length of the filament by stepping along points (in kpc)
        # but first, replace too large differences with radius (where it branches or didn't start at an end point)
        # and plot these to make sure it is appropriate
        diffs = np.sqrt(periodic(np.diff(x[points]), boxsize)**2 + periodic(np.diff(y[points]), boxsize)**2 + periodic(np.diff(z[points]), boxsize)**2)

        if np.any(diffs > 4.*radius):
            large_step_fil = np.append(large_step_fil, fil_num)

            fig = plt.figure(figsize=(18,6))

            ax1 = fig.add_subplot(131)
            ax1.scatter(x[points], y[points], c='red', s=np.arange(points.size+1))
            ax1.plot(x[points], y[points], 'r-')
            ax1.set_xlim([0.9*x.min(), np.min([1.1*x.max(), boxsize])])
            ax1.set_ylim([0.9*y.min(), np.min([1.1*y.max(), boxsize])])

            ax2 = fig.add_subplot(132)
            ax2.scatter(y[points], z[points], c='red', s=np.arange(points.size+1))
            ax2.plot(y[points], z[points], 'r-')
            ax2.set_xlim([0.9*y.min(), np.min([1.1*y.max(), boxsize])])
            ax2.set_ylim([0.9*z.min(), np.min([1.1*z.max(), boxsize])])

            ax3 = fig.add_subplot(133)
            ax3.scatter(z[points], x[points], c='red', s=np.arange(points.size+1))
            ax3.plot(z[points], x[points], 'r-')
            ax3.set_xlim([0.9*z.min(), np.min([1.1*z.max(), boxsize])])
            ax3.set_ylim([0.9*x.min(), np.min([1.1*x.max(), boxsize])])

            plt.tight_layout()
            fig.savefig(path+'filament_id#'+str(fil_num)+'.png')
            plt.close()

        diffs[np.where(diffs > 4.*radius)] = radius
        length[fil_num-1] = np.sum(diffs)/1000.


#------------- Remove any filaments that are shorter than the smoothing length
    print(length.max(), np.unique(ids).size, length.size, 'last two should be equal')
    too_short = np.unique(ids)[np.where(length <= 2.)[0]] # limit filaments to those larger than 2 Mpc

    for i in range(too_short.size):
        ids[np.where(ids == too_short[i])] = 0

    for i in range(np.unique(ids).size):
        old_id = np.unique(ids)[i]
        if np.any(old_id == large_step_fil): # rename figures to correspond to new id number
            os.rename(path+'filament_id#'+str(old_id)+'.png', path+'filament_id#'+str(i)+'.png')
        ids[np.where(ids == np.unique(ids)[i])] = i

    length = length[np.where(length > 2.)] # lengths longer than 2 Mpc
    print(np.unique(ids).size, length.size, 'first number should be 1 more than second number')

#------------- Assign halos to filaments based on distance and smoothing length
    halo_cut = 2000. # start with 2 Mpc proximity, possibly add scaling with boxsize, i.e. (boxsize/64000.)*2000.
    h_ids = np.zeros(threshold.shape[1], dtype=int)
    for i in range(threshold.shape[1]):
        xd = periodic(threshold[0, i] - filaments[0], boxsize)
        yd = periodic(threshold[1, i] - filaments[1], boxsize)
        zd = periodic(threshold[2, i] - filaments[2], boxsize)

        dist = np.sqrt(xd**2 + yd**2 + zd**2)
        min_dist = np.min(dist[np.where(ids != 0)])

        if min_dist < halo_cut:
            close_fil = np.where(dist == min_dist)[0]
            h_ids[i] = ids[close_fil]

#------------- Calculate filament properties and remove any filaments with fewer than 3 halos
    print(np.unique(h_ids).size, 'should be less or equal to first number of previous line')
    number = np.zeros(length.size, dtype=int) # number of halos assigned to filament
    mass = np.zeros(length.size)  # total mass of halos assigned to filament
    for i in range(1, np.unique(ids).size):
        number[i-1] = int(np.where(h_ids == i)[0].size)
        mass[i-1] = np.sum(masses[np.where(h_ids == i)[0]])

    length = length[np.where(number >= 3)[0]]
    mass = mass[np.where(number >= 3)[0]]
    number = number[np.where(number >= 3)[0]]

#------------- Save the length, number, and mass distributions into a file
    df = pd.DataFrame({'Length (Mpc)':length, ' Mass (10^10 Msun)':mass, ' Number of halos':number})
    df.to_csv(path+'dists.out', sep=',', na_rep=-1, index=False)

#------------- Make and save a figure showing the filaments and halos (field and filament halos)
    psize = 0.5
    fig = plt.figure(figsize=(27, 9))

    ax = fig.add_subplot(131)
    ax.set_xlim([0, boxsize])
    ax.set_ylim([0, boxsize])
    ax.set_axis_off()
    ax.scatter(filaments[0], filaments[1], s=psize, c='black') # all filaments
    ax.scatter(threshold[0, np.where(h_ids != 0)[0]], threshold[1, np.where(h_ids != 0)[0]], s=10*psize, c='green', alpha=0.5) # halos belonging to filaments
    ax.scatter(threshold[0, np.where(h_ids == 0)[0]], threshold[1, np.where(h_ids == 0)[0]], s=10*psize, c='red', alpha=0.5) # field halos
    ax.scatter(filaments[0, np.where(ids == 0)[0]], filaments[1, np.where(ids == 0)[0]], s=25*psize, c='blue', alpha=0.5) # removed filaments

    ax1 = fig.add_subplot(132)
    ax1.set_xlim([0, boxsize])
    ax1.set_ylim([0, boxsize])
    ax1.set_axis_off()
    ax1.scatter(filaments[1], filaments[2], s=psize, c='black')
    ax1.scatter(threshold[1, np.where(h_ids != 0)[0]], threshold[2, np.where(h_ids != 0)[0]], s=10*psize, c='green', alpha=0.5)
    ax1.scatter(threshold[1, np.where(h_ids == 0)[0]], threshold[2, np.where(h_ids == 0)[0]], s=10*psize, c='red', alpha=0.5)
    ax1.scatter(filaments[1, np.where(ids == 0)[0]], filaments[2, np.where(ids == 0)[0]], s=25*psize, c='blue', alpha=0.5)

    ax2 = fig.add_subplot(133)
    ax2.set_xlim([0, boxsize])
    ax2.set_ylim([0, boxsize])
    ax2.set_axis_off()
    ax2.scatter(filaments[2], filaments[0], s=psize, c='black')
    ax2.scatter(threshold[2, np.where(h_ids != 0)[0]], threshold[0, np.where(h_ids != 0)[0]], s=10*psize, c='green', alpha=0.5)
    ax2.scatter(threshold[2, np.where(h_ids == 0)[0]], threshold[0, np.where(h_ids == 0)[0]], s=10*psize, c='red', alpha=0.5)
    ax2.scatter(filaments[2, np.where(ids == 0)[0]], filaments[0, np.where(ids == 0)[0]], s=25*psize, c='blue', alpha=0.5)

    fig.tight_layout()
    fig.savefig(path+'all_filament.png')
    plt.close()

    return(h_ids, ids)




PARSER = argparse.ArgumentParser()
PARSER.add_argument('path', type=str)
PARSER.add_argument('cutoff', type=float)
ARGS = PARSER.parse_args()

start = time.time()
#--------------- Import halo, filament, and mass arrays
# positions of the halos before the SCMS algorithm, but after thresholding
threshold = pd.read_csv(ARGS.path+'threshold.out').values[:, 0:3].T
# masses for halos retained from the thresholding step
masses = pd.read_csv(ARGS.path+'threshold.out').values[:, 3]
# positions of the tracers after SCMS, defining the filaments
filaments = pd.read_csv(ARGS.path+'scms.out').values.T
# uncertainty not currently included****************************************

#--------------- Import smoothing length and boxsize
smoothing = np.loadtxt(ARGS.path+'params.out')[0]
boxsize = np.loadtxt(ARGS.path+'params.out')[1]


ids = separate(ARGS.path, ARGS.cutoff, filaments, smoothing, boxsize)

h_ids, f_ids = get_dists(ARGS.path, ids, threshold, filaments, masses, smoothing, boxsize)


#------------- Save the separated filaments, filament ids, halos, halo ids, masses to a file
df1 = pd.DataFrame({'Filament ID':f_ids}, dtype=int)
df2 = pd.DataFrame({' x_fil (kpc)':filaments[0], ' y_fil (kpc)':filaments[1], ' z_fil (kpc)':filaments[2]})
df3 = pd.DataFrame({' Halo ID':h_ids}, dtype=int)
df4 = pd.DataFrame({' x_halo (kpc)':threshold[0], ' y_halo (kpc)':threshold[1], ' z_halo (kpc)':threshold[2], ' mass (10^10 Msun)':masses})
# uncertainty?**************************
df = pd.concat([df1, df2, df3, df4], ignore_index=False, axis=1)
df.to_csv(ARGS.path+'separated.out', sep=',', na_rep=-1, index=False)

end = time.time()
print('completed in', np.round((end-start)/60, 1), 'minutes')
