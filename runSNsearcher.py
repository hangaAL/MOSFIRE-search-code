import os
from time import perf_counter
import pandas as pd
import sys
sys.path.append("C:/Users/hanga/Desktop/DAWN-IRES/scriptystuff/")
import SNsearcher as sns

def run_SNsearcher(dirfilepath, pos_SN_thresh, neg_SN_thresh, pos_count_thresh, neg_count_thresh, dyslit, negSN, continuum):

    '''
    This function runs the SNsearch function on all of the .fits files in a specified directory,
    making sure that the outputs for each file are kept separate from each other.

    -----

    Parameters:

    dirfilepath (string): the directory containing the files you want the program to run on, which
    the program will also save all the output files to.

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

    continuum (Boolean): determines if the program looks for continuum emission (True = yes, 
    False = no). Only matters if negSN == True

    -----

    Outputs:

    None
    '''
    fnames = [] #where we record filenames
    galaxies = [] #where we record the amount of galaxies found
    cf_list = [] #where we record if the spectrum had a continuum or not
    sandwiches = [] #where we record the number of "sandwiches"
    tarlist = [] #where we record the amount of target galaxies found
    serlist = [] #where we record the amount of serendipitous galaxies found
    poslist = [] #where we record the amount of positive hits found if negSN is set to False
    mainxlist = [] #where the x locations of either galaxies or positive hits are recorded based on the value of negSN
    mainylist = [] #ditto but for y values

    for file in os.listdir(dirfilepath): #iterates through each file in the given directory
        
        start_file = perf_counter()
        if file.endswith(".fits" or ".fit" or ".fts" or ".FITS" or ".FIT"): #ONLY runs the code on .fits files!

            fnames.append(file) #record the filename
            imagepath = os.path.join(dirfilepath + "/" + file) #finds the path to the .fits image
            outfilepath = os.path.join(dirfilepath + "/" + os.path.splitext(file)[0] + "/") #uses the filename to name the directory all the corresponding output files will be stored in

            if not os.path.exists(outfilepath): #makes sure that directory doesn't already exist

                os.mkdir(outfilepath) #if so, creates the directory

            sns.SNsearch(pos_SN_thresh, neg_SN_thresh, pos_count_thresh, neg_count_thresh, dyslit, negSN, continuum, imagepath, outfilepath, sandwiches, galaxies, tarlist, serlist, cf_list, poslist, mainxlist, mainylist) #runs the S/N searching code on the image
            end_file = perf_counter()
            print(file, "is done! Time taken:", end_file-start_file) #alerts you that it's done with the file in question

    if negSN: #if you're looking for negative S/N

        if continuum: #and screening for continuum spectra, we save all the relevant info to a csv file

            dic = {"File": fnames, "Emission Line Detections": sandwiches, "Galaxy Detections": galaxies, "Target Galaxies": tarlist, "Serendipitous Galaxies": serlist, "Emission Line x-values": mainxlist, "Emission Line y-values": mainylist, "Continuum Detection": cf_list}
            dataf = pd.DataFrame(data=dic)
            dataf.to_csv(dirfilepath + "/detections.csv", sep=" ", index=False)

        else: #without screening for continuum spectra, we only skip saving the continuum info

            dic = {"File": fnames, "Emission Line Detections": sandwiches, "Galaxy Detections": galaxies, "Target Galaxies": tarlist, "Serendipitous Galaxies": serlist, "Emission Line x-values": mainxlist, "Emission Line y-values": mainylist}
            dataf = pd.DataFrame(data=dic)
            dataf.to_csv(dirfilepath + "/detections.csv", sep=" ", index=False)

    if not negSN: #if we aren't looking for negative S/N, we only save the amount of positive hits and their locations

        dic = {"File": fnames, "High S/N Detections": poslist, "x-values": mainxlist, "y-values": mainylist}
        dataf = pd.DataFrame(data=dic)
        dataf.to_csv(dirfilepath + "/detections.csv", sep=" ", index=False)

#have which galaxies are target vs serendipitous specified in output