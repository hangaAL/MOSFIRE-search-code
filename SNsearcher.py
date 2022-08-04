from astropy.io import fits
import numpy as np
import scipy.ndimage as nd
import pandas as pd
from time import perf_counter
import reqfunctions as rf

#Searching for significant S/N

def SNsearch(pos_SN_thresh, neg_SN_thresh, pos_count_thresh, neg_count_thresh, dyslit, negSN, continuum, imagepath, outfilepath, sandwiches, galaxies, tarlist, serlist, cf_list, poslist, mainxlist, mainylist):

    '''
    This function automates the process of searching for significant S/N and sandwiches in 
    a single image, and saving the resulting regions and csv files to the correct location. 

    -----

    Parameters: 

    pos_SN_thresh (integer): the threshold above which positive S/N is considered significant
    (should be positive)

    neg_SN_thresh (integer): the threshold below which negative S/N is considered significant
    (should be negative)

    pos_count_thresh (integer): the amount of consecutive positive S/N hits that is considered
    significant

    neg_count_thresh (integer): the amount of consecutive negative S/N hits that is considered
    significant

    dyslit (integer): how many pixels above and below the center of the slit you want the slit 
    to extend (for example, for a 10 pixel wide slit, you should set dyslit = 5)

    negSN (Boolean): determines if the program looks for negative S/N (True = yes, False = no)

    imagepath (string): the filepath to the FITS image data you want to use

    outfilepath (string): the filepath to the directory you want the output files saved in

    sandwiches (integer list): where the amount of sandwiches found in each file will be recorded

    galaxies (integer list): where the amount of galaxies found in each file will be recorded

    tarlist (integer list): where the y-locations of target galaxies found in each file will be 
    recorded

    serlist (integer list): where the y-locations of serendipitous galaxies found in each file will 
    be recorded

    cf_list (Boolean list): where whether or not each file had a continuum will be recorded

    poslist (integer list): where the amount of positive S/N hits in a file will be recorded if
    negSN == False

    mainxlist (integer list): where the x locations of either galaxies or positive S/N hits will
    be recorded depending on the value of negSN

    mainylist (integer list): where the y locations of either galaxies or positive S/N hits will
    be recorded depending on the value of negSN

    -----

    Outputs:

    None
    ''' 

    rgntxt = open(outfilepath + "galaxy_locations.reg", "w+") #this is where we're going to put the regions for ds9
    rgntxt.write("# Region file format: DS9 version 4.1 \n") #setup for how these guys are going to look
    rgntxt.write("global color=blue dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 \n")
    rgntxt.write("IMAGE \n")


    xlist = [] #registers the x positions of hits
    ylist = [] #ditto but for y positions
    hit_type_list = [] #ditto but for the type of hit

    spec2D, slitcenter, wave, num, den, imgysize = rf.make2Dspec(imagepath, dyslit) #make a 2D spectrum out of the data

    for m in range(0, spec2D.shape[0]-5, dyslit): #range set so the code marches down the spectrum in larger increments but doesn't go far past the edge 

        slice1D, comparray = rf.make1Dslice(spec2D, slitcenter, dyslit) #make a 1D slice of the 2D spectrum
        poscounter = 0 #counts how many S/N hits > positive threshold we've had in a row
        negcounter = 0 #ditto but for the negative threshold

        for i in range(slice1D.size): #time to mark those hits

            if slice1D[i] > pos_SN_thresh: #if S/N > positive threshold:

                poscounter = poscounter+1 #the positive counter increases by 1

                if negSN: #if we're looking for negative S/N

                    if negcounter > neg_count_thresh: #check to see how the negative counter's doing

                        rf.add_detect(negcounter, comparray, i, -1) #if it's high enough, we go back and register the hits

                    negcounter = 0 #either way, we reset the negative counter since what we just hit was not negative

            elif slice1D[i] < neg_SN_thresh: #the above but vice versa

                if negSN:

                    negcounter = negcounter+1

                if poscounter > pos_count_thresh:

                    rf.add_detect(poscounter, comparray, i, 1)

                poscounter = 0

            else: #if what we find doesn't pass either threshold, we check both count and if needed, register hits

                if poscounter > pos_count_thresh:

                    rf.add_detect(poscounter, comparray, i, 1)

                elif negcounter > neg_count_thresh:

                    rf.add_detect(negcounter, comparray, i, -1)

                poscounter = 0
                negcounter = 0

        rf.make_rgn_lists(comparray, rgntxt, xlist, ylist, hit_type_list, negSN, slitcenter) #then we make the regions file and, if needed, the required array

        slitcenter = slitcenter-5 #and then we move 5 pixels down to the next slit

    cont_exists = False #to make EXTRA sure this value is reset from the last time find_continuum was run! and that the sandwich finding codes run if the user doesn't want to screen for continuum spectra

    if negSN: #if we're looking for negative S/N

        if continuum: #if you want to screen for continuum spectra

            cont_exists = rf.find_continuum(xlist, ylist, cf_list) #looks for a continuum

        if not cont_exists: #if we don't have a continuum, or we're just not looking for them

            df_pos, df_neg = rf.createdfs(xlist, ylist, hit_type_list) #create dataframes with the relevant information about all the hits
            rf.findclusters(df_pos) #find the clusters of positive hits
            medylist, medxlist = rf.findsandwiches(df_pos, df_neg, rgntxt, sandwiches) #find the sandwiches and their locations
            galmedylist = rf.count_galaxies(medylist, medxlist, imgysize, tarlist, serlist, mainxlist, mainylist)
            rf.cut_out_detects(galmedylist, wave, num, den, dyslit, outfilepath, galaxies) #cut out and save the raw data corresponding to each detection

        if cont_exists: #if we do have a continuum

            galaxies.append(0) #and record there being zero of all this stuff
            sandwiches.append(0)
            tarlist.append(None)
            serlist.append(None)
            mainxlist.append(None)
            mainylist.append(None) 
            
    elif not negSN: #if we're not looking for negative S/N anyway

        rf.make_pos_hitlist(xlist, ylist, poslist, mainxlist, mainylist)

    rgntxt.close() #close the ds9 regions file with all of the hits and sandwiches marked
