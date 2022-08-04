from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as nd
import pandas as pd

def make2Dspec(imagepath, dyslit):

    '''
    This function takes in the FITS image data and creates a smoothed 2D spectrum of it,
    and also returns the starting position for the center of the slit that we'll be
    cutting out of the spectrum.

    -----

    Parameters:

    imagepath (string): the filepath to the FITS image data you want to use

    dyslit (integer): how many pixels above and below the center of the slit you want
    the slit to extend (for example, for a 10 pixel wide slit, you should set dyslit 
    = 5)

    -----

    Outputs:

    SN (2D float array): the smoothed 2D spectrum of the image data

    slitcenter (integer): the starting y-value of the center of the slit that will be 
    cut out later on

    wave (float array): the wavelengths present in the spectrum in Angstroms

    num (float array): numerator for the S/N

    den (float array): denominator for the S/N

    targy (integer): the expected y-pixel of the target galaxy    
    '''

    spec2D = fits.open(imagepath) #opening the file with the spectrum data
    h2 = spec2D[0].header #opening the header
    wave = (np.arange(1, h2['NAXIS1']+1) - h2['CRPIX1'])*h2['CD1_1'] + h2['CRVAL1'] #these are the wavelengths present in the spectrum
    num = nd.gaussian_filter(spec2D[0].data*spec2D[1].data, 2) #signal
    den = nd.gaussian_filter(spec2D[1].data, 5)**2 / spec2D[1].data #weight (NOT error!)
    SN = num/np.sqrt(den) #make the S/N 2D spectrum
    imgysize = spec2D[0].data.shape[0]
    slitcenter = imgysize - (dyslit+1) #set the initial location of the center of the slit
    spec2D.close() #close the file
    targy = int(h2["TARGYPIX"]) #extract the expected y pixel of the target galaxy

    return SN, slitcenter, wave, num, den, targy



def add_detect(counter, array, h, value): 

    '''
    This function looks at the value of the counter that counts how many consecutive hits
    of significant S/N the code has found. It then sets the corresponding past values in 
    the array being used to record hits to record the type of hit. For example, if the 
    counter recorded 5 hits, it would set the past 5 hits before wherever in 
    the array we are to a specified value.

    -----

    Parameters:

    counter (integer): the counter variable being used to record the amount of consecutive
    hits of significant S/N

    array (1D integer array): the array being used to record the locations and types of the 
    S/N hits

    h (integer): the iteration variable being used in the loop this function is used in (for 
    example, if the loop were "for i in range(7)", you'd want to set h to i)

    value (integer): the value you want to use to record hits in the array

    -----

    Outputs:

    None
    '''


    for g in range(1, counter+1): #set past amount of values in array to a value

        array[h-g] = value #based on the length of counter



def make1Dslice(spec2D, slitcenter, dyslit):

    '''
    This function takes in the 2D spectrum, cuts out a slit of a specified width centered at
    a certain y-value, and then collapses that slice to a 1D spectrum. It also produces a 
    corresponding array of the same shape that will be used to record the hits in the 1D
    spectrum.

    -----

    Parameters:

    spec2D (2D float array): the smoothed 2D spectrum of the image data

    slitcenter (integer): the starting y-value of the center of the slit that will be 
    cut out 

    dyslit (integer): how many pixels above and below the center of the slit you want
    the slit to extend (for example, for a 10 pixel wide slit, you should set dyslit 
    = 5)

    -----

    Outputs:

    slice1D (1D float array): the collapsed 1D spectrum of the cut out 2D slice

    comparray (1D integer array): an array of zeroes the same shape as slice1D that will
    be used to record hits
    '''

    slice2D = spec2D[[range(slitcenter-dyslit, slitcenter+dyslit+1)], :] #cutting out just a slit
    slice1D = np.nansum(slice2D, axis=(0, 1)) #collapsing the spectrum to 1D and getting rid of the extra axis that shows up for some reason
    comparray = np.zeros(slice1D.shape, dtype=int) #this array will be used to mark where we have S/N hits

    return slice1D, comparray



def make_rgn_lists(comparray, rgntxt, xlist, ylist, hit_type_list, negSN, slitcenter): 

    '''
    This function marks significant S/N hits for a specific slit of the data in a SAOImage DS9 regions file. If you are
    looking for negative S/N, it also records the positions and types of the hits in
    lists to later turn into a pandas dataframe.

    -----

    Parameters:

    comparray (1D integer array): the array that the hits are recorded in. 

    rgntxt (open text file): the DS9 regions file that the locations of the hits will also be recorded in

    xlist (integer list): the list of the x coordinates of the hits

    ylist (integer list): the list of the y coordinates of the hits

    hit_type_list (integer list): the list of the types of the hits (1 is is positive, -1 is negative)

    negSN (Boolean): determines if the program looks for negative S/N (True = yes, False = no)

    slitcenter (integer): the y-value of the center of the slit the program is currently looking at

    -----

    Outputs:

    None
    '''

    for k in range(comparray.size): #go through array checking for hits

        if comparray[k] == 1: #marking positive hits in the regions file

            rgntxt.write("circle " + str(k) + " " + str(slitcenter) + " 5 # color=blue \n")

            xlist.append(k) #record the x pixel of the positive hit
            ylist.append(slitcenter) #record the y pixel of the positive hit

            if negSN: #put in the array only if negSN=True

                hit_type_list.append(1)

        if negSN: #do all of the above for negative hits only if negSN=True

            if comparray[k] == -1:

                rgntxt.write("circle " + str(k) + " " + str(slitcenter) + " 5 # color=red \n")
                xlist.append(k) 
                ylist.append(slitcenter)
                hit_type_list.append(-1) 



def createdfs(xlist, ylist, hit_type_list): 

    '''
    This function creates dataframes containing the x and y coordinates for each type of S/N hit in
    an image for later use, with a column for recording hits that are parts of the same cluster for
    positive hits.

    -----

    Parameters:

    xlist (integer list): the list of the x coordinates of the hits

    ylist (integer list): the list of the y coordinates of the hits

    hit_type_list (integer list): the list of the types of the hits (1 is is positive, -1 is negative)

    -----

    Outputs:

    df_pos (pandas dataframe): the dataframe containing the x and y coordinates of the positive hits, with
    a cloumn for cluster numbers

    df_neg (pandas dataframe): the dataframe containing the x and y coordinates of the negative hits
    '''

    d = {"x": xlist, "y": ylist, "hit_type": hit_type_list} #make a dictionary first
    df = pd.DataFrame(data=d) #turn the dictionary into a dataframe
    df_pos = df[df["hit_type"]==1] #just the positive hits
    df_neg = df[df["hit_type"]==-1] #just the negative hits
    df_pos["cluster"] = np.nan #add a column for marking which cluster positive hits are in

    return df_pos, df_neg    



def findclusters(df_pos): #find clusters of positive hits
    
    '''
    This function looks through the dataframe of positive hits and groups them into clusters based on their
    locations relative to each other. Hits that are part of the same cluster are marked with the same
    cluster number. 

    -----

    Parameters:

    df_pos (pandas dataframe): the dataframe containing the x and y coordinates of the positive hits, with
    a cloumn for cluster numbers

    ------

    Outputs:

    None
    '''

    clusternum = 1 #this number will be used to mark which hits are part of the same cluster
    
    for n in df_pos.index: #iterates through the length of the positive dataframe
    
        if pd.isna(df_pos["cluster"][n]): #checks if the hit clusterless; if so,

            for a in df_pos.index: #iterates through the dataframe again 

                if (df_pos["x"][a] < (df_pos["x"][n] + 15)) & (df_pos["x"][a] > (df_pos["x"][n] - 15)): # and within 20 pixels 

                     if (df_pos["y"][a] < (df_pos["y"][n] + 15)) & (df_pos["y"][a] > (df_pos["y"][n] - 15)): #both in x & y

                        if pd.isna(df_pos["cluster"][a]): #makes sure the hit isn't already part of a cluster

                            df_pos.loc[a,"cluster"] = clusternum #adds the hit to the cluster
                            df_pos.loc[n, "cluster"] = clusternum #marks the original hit as part of a cluster

        clusternum =  clusternum+1 #to make sure everything isn't marked as part of the same cluster



def findsandwiches(df_pos, df_neg, rgntxt, sandwiches): #finding sandwiched clusters

    '''
    This function looks for for negative S/N hits directly above and below each cluster to determine if
    it is a "sandwich," i.e. it looks for the shadows above and below a spot of high S/N to determine if
    it could potentially be part of a high-redshift galaxy spectrum. It then saves the center of each 
    sandwich to a list for later use, and inserts a yellow circle around each sandwich in the regions file.

    -----

    Parameters:

    df_pos (pandas dataframe): the dataframe containing the x and y coordinates and cluster numbers of 
    the positive hits

    df_neg (pandas dataframe): the dataframe containing the x and y coordinates of the negative hits

    rgntxt (open text file): the DS9 regions file that the locations of the hits and sandwiches are
    being recorded in

    sandwiches (integer list): where the amount of sandwiches found in each file will be recorded

    -----

    Outputs:

    medylist (integer list): the list containing the center y pixel of each sandwich

    medxlist (integer list): the list containing the center x pixel of each sandwich
    '''

    medylist = [] #this is where we'll store the y-locations of any sandwiches we find 
    medxlist = []
    done_clusters = [] #every time we check to see if a cluster is sandwiched, we put it in this list so we don't accidentally do it again 
    above = False #this variable records if there's negative hits above the cluster
    below = False #ditto but for below

    for q in df_pos.index: #goes through the positive hits
    
        if df_pos["cluster"][q] not in done_clusters: #basically if it's a cluster we haven't done yet

            cn = df_pos["cluster"][q] #we record the cluster number

            medx = df_pos[df_pos["cluster"]==cn]["x"].median() #and record the median x and y values of the hits in the cluster
            medy = df_pos[df_pos["cluster"]==cn]["y"].median() #median because the mean would probably return a float and that's extra annoying steps :/

            for t in df_neg.index: #now we go looking for the two pieces of bread in our galaxy sandwich

                if (df_neg["x"][t] < (medx + 3)) & (df_neg["x"][t] > (medx - 3)): #make sure the hits are in the right x range first

                    if (df_neg["y"][t] < (medy + 25)) & (df_neg["y"][t] > medy): #check above

                        above = True #set to true if there's a slice of bread

                    if (df_neg["y"][t] > (medy - 25)) & (df_neg["y"][t] < medy): #check below

                        below = True #ditto

            if above & below: #if we fully have a sandwich

                rgntxt.write("circle " + str(medx) + " " + str(medy) + " 20  # color=yellow \n") #we circle it in the regions file
                medylist.append(int(medy)) #we record the y-locations of the sandwiches so we can cut out their data from the spectrum later
                medxlist.append(int(medx))

            done_clusters.append(cn) #then we put the cluster number there so we don't do the same cluster a million times
        
        above = False #set these to false to make sure we don't accidentally have open-faced sandwiches or just meat or something
        below = False 

    sandwiches.append(len(medylist)) #record the amount of sandwiches found
    
    return medylist, medxlist     



def cut_out_detects(galmedylist, wave, num, den, dyslit, outfilepath, galaxies):

    '''
    This function cuts out a slit of the unprocessed, raw data centered at the y-coordinate
    of each sandwich detection, and saves it as a pandas dataframe to a csv file.

    -----

    Parameters:

    galmedylist (integer list): the list containing the y coordinate of each galaxy

    wave (float array): the wavelengths present in the spectrum in Angstroms
 
    num (float array): numerator for the S/N

    den (float array): denominator for the S/N  

    dyslit (integer): the extent of the slit above and below its center in pixels

    outfilepath (string): the filepath to the directory you want the output files saved in

    galaxies (integer list): where the amount of galaxies found in each file will be recorded

    -----

    Outputs:

    None
    '''

    z = 1 #the number will be used to label each detection

    for y in galmedylist: #iterates through the y-locations of all the sandwiches
        #these might not actually be flux and error, come back to later
        flux, _ = make1Dslice(num, y, dyslit)  #cutting out a slit in that location in the flux
        error, _ = make1Dslice(den, y, dyslit) #ditto for the weight
        dic = {"wavelength": wave, "flux": flux, "error": error} #putting it into a dictionary
        dff = pd.DataFrame(data=dic) #turning that dictionary into a dataframe
        dff.to_csv(outfilepath + "detection_" + str(z) + ".csv", sep=" ", index=False) #saving it as a csv
        z = z+1 #Increasing the number to avoid accidentally overwritten or confused files

    galaxies.append(z-1) #saves the amount of detected sandwiches 


def find_continuum(xlist, ylist, cf_list): 

    '''
    This function checks to see if an image contains a continuum. If it does, it lets the main
    function know to skip looking for sandwiches in the image to save time. 

    -----

    Parameters:

    xlist (integer list): the list of the x coordinates of the hits

    ylist (integer list): the list of the y coordinates of the hits

    cf_list (Boolean list): the list containing the values that signify whether each images
    contains a continuum

    -----

    Outputs:

    cont_exists (Boolean): True if the image contains a continuum, False if not
    '''

    cont_exists = False #to make sure we reset from the previous time the function was run

    if (len(xlist) > 8000) & (len(ylist) > 8000): #checks if there's a high enough number of hits in the image

        cont_exists = True #if so, sets this value accordingly
        print("continuum found!") #lets the user know

    cf_list.append(cont_exists) #records what the function found 

    return cont_exists



def count_galaxies(medylist, medxlist, targy, tarlist, serlist, mainxlist, mainylist):

    '''
    This function looks for emission lines along the same y axis to classify as a galaxy,
    and returns the y-coordinate of each galaxy it finds. It also compares said y-
    coordinate to the expected target y-coordinate listed in the image header to check if
    it's a target or serendipitous galaxy.

    -----

    Parameters:

    medylist (integer list): the list containing the center y pixel of each sandwich

    medxlist (integer list): the list containing the center x pixel of each sandwich

    targy (integer): the expected y-pixel of the target galaxy

    tarlist (integer list): where the y-locations of target galaxies found in each file will be recorded

    serlist (integer list): where the y-locations of serendipitous galaxies found in each file will 
    be recorded

    mainxlist (integer list): where the x locations of either galaxies or positive S/N hits will
    be recorded depending on the value of negSN

    mainylist (integer list): where the y locations of either galaxies or positive S/N hits will
    be recorded depending on the value of negSN

    -----
    
    Outputs:

    galmedylist (integer list): the list containing the y coordinate of each galaxy
    '''

    gals = {} #the dictionary where each galaxy and its corresponding emission lines will be stored
    in_gal = False #used to mark whether or not a emission line is in a specific galaxy
    not_in = True #used to mark whether or not an emission line is included in the dictionary already
    c = 1 #galaxy number
    galmedylist = [] #for storing the y positions of the galaxies

    for t in medylist: #iterate through the sandwiches

        in_gal = False #set these to avoid accidentally marking stuff wrong
        not_in = True
        keys = list(gals.keys()) #update the list of keys in the dictionary with each iteration

        for u in keys: #go through each of the keys

            if not_in: #if the galaxy isn't already in the dictionary
                
                median = int(np.median(gals[str(u)])) #figure out the median location of all the sandwiches already in a galaxy

                if (t < median+10) & (t > median-10): #if the sandwich is close enough to median location

                    in_gal = True #marked as part of this galaxy
                    
                if in_gal: #if the sandwich is part of this galaxy

                    gals[str(u)].append(t) #add it to the dictionary under the correct key
                    not_in = False #mark it as in the dictionary already
        
        if (len(keys) == 0) or (not_in & (len(keys) != 0)): #TEST IF THIS WORKS!

            gals[str(c)] = [t] #put the sandwich in its own galaxy
            c = c+1 #increase the galaxy number by one

    for b in gals: #iterates through the galaxies in the dictionary

        galmed = int(np.median(gals[str(b)])) #finds each galaxy's y-location
        galmedylist.append(galmed) #puts it in the list

    targets = [] #list of target galaxies
    seren = [] #list of serendipitous galaxies MAYBE PUT THESE IN OUTPUT?

    for r in galmedylist: #iterates through the galaxies

        if (r < targy+10) & (r > targy-10): #if it's close to the target galaxy's expected location

            targets.append(r) #mark it as a potential target galaxy

        else: #otherwise mark it as a serendipitous detection

            seren.append(r)

    if len(targets) == 0: #if there's no detected galaxies, put null value in locations

        tarlist.append(None)

    if len(targets) != 0: #if there are, record their median y values
        
        tarlist.append(targets)
    
    if len(seren) == 0: #same but for serendipitous galaxies

        serlist.append(None)
    
    if len(seren) != 0:

        serlist.append(seren)
    
    if len(medxlist) == 0: #if there's no sandwiches, put null values for their x and y locations

        mainxlist.append(None)
        mainylist.append(None)

    if len(medxlist) != 0: #otherwise record their x and y locations
    
        mainxlist.append(medxlist)
        mainylist.append(medylist)

    return galmedylist



def make_pos_hitlist(xlist, ylist, poslist, mainxlist, mainylist):

    '''
    This function makes lists of the x and y locations of all positive S/N hits
    in an image if negSN == False.

    -----

    Parameters:

    xlist (integer list): the list of the x coordinates of the hits

    ylist (integer list): the list of the y coordinates of the hits

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

    poslist.append(len(xlist)) #record the amount of positive S/N hits in the images
    
    if len(xlist) == 0: #if there's no hits, record null values for the x and y locations 
        
        mainxlist.append(None)
        mainylist.append(None)

    else: #otherwise record all the x and y locations

        mainxlist.append(xlist)
        mainylist.append(ylist)
