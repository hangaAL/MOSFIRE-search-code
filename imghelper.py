from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import scipy.ndimage as nd
import os
from regions import Regions 
from time import perf_counter
import pandas as pd
from platform import system as psys

def show_2Dspec(dirfilepath): 
    
    '''
    This function takes the regions files generated by SNsearcher and uses them to save pngs in a separate 
    folder comparing mthe original images with the versions with the regions overlaid for easier viewing. 
    Note: This function requires v0.6 (the latest version at the time of writing) of the in-development 
    Astropy Regions module to run (https://astropy-regions.readthedocs.io/en/stable/).

    -----

    Parameters: 

    dirfilepath (string): the directory containing the files you want the program to run on, which
    the program will also save all the output files to. Should be the same as the the directory 
    you ran SNsearcher on.

    -----

    Outputs:

    None    
    '''

    os.mkdir(dirfilepath + "/compare/") #make the folder all the images will go in
    
    mainstart = perf_counter() #start the timer for the whole function

    plt.ioff() #make sure we don't display plots unnecessarily

    for file in os.listdir(dirfilepath): #iterates through each image file

        if file.endswith(".fits" or ".fit" or ".fts" or ".FITS" or ".FIT"): #make sure it's a fits file

            start = perf_counter() #start the timer for just this image

            image = os.path.join(dirfilepath + "/" + file) #extract the filepath to the image
    
            spec2D = fits.open(image) #open the image
            
            num = nd.gaussian_filter(spec2D[0].data*spec2D[1].data, 2) #make the smoothed 2D spectrum
            den = nd.gaussian_filter(spec2D[1].data, 5)**2 / spec2D[1].data
            spec2D.close() #close the file lol
            SN = num/np.sqrt(den) #get the S/N

            fig, (ax1, ax2) = plt.subplots(2,1, figsize=(100, 17.5), sharey=True) #make the figure and subplots
            ax1.imshow(SN, vmin=-1.5, vmax=3, cmap='cubehelix', origin='lower') #the upper subplot, which will just be the original image
            ax1.set_aspect('auto')
            ax2.imshow(SN, vmin=-1.5, vmax=3, cmap='cubehelix', origin='lower') #the lower subplot, which will have the regions overlaid
            ax2.set_aspect("auto")
            
            rgn = Regions.read(dirfilepath + "/" + os.path.splitext(file)[0] + "/galaxy_locations.reg", format='ds9') #open and parse the corresponding regions file

            for shape in rgn: #put all the individual shapes on the image
                artist = shape.as_artist()
                ax2.add_artist(artist)
            
            plt.savefig(dirfilepath + "/compare/" + os.path.splitext(file)[0] +".png") #save the image

            end = perf_counter() #end the timer for the image

            print(file, "saved. Time taken:", end-start) #display the amount of time it took to run

    mainend = perf_counter() #end the timer for the whole function

    print("Total time taken:", mainend-mainstart) #display the amount of time it took for the whole function to run



def open_image(dirfilepath, loc):

    '''
    This function allows you to automatically open the comparison image generated by show_2Dspec()
    for any file you want to look at in your system's native image viewer (except for Linux, there's
    too much variability there so for Linux it just uses matplotlib). 

    -----

    Parameters:

    dirfilepath (string): the directory containing the files you want the program to run on, which
    the program will also save all the output files to. Should be the same as the the directory 
    you ran SNsearcher on.

    loc (integer): the index number of the file whose image you want to look at as specified in the
    detections.csv file generated by SNsearcher.

    -----

    Outputs:

    URL (string): the path to the image
    '''
    
    table = pd.read_csv(dirfilepath + "/detections.csv", sep=" ") #read in the table
    filename = table["File"][loc] #extract the name of the file
    imgname = os.path.splitext(filename)[0] + ".png" #convert it into the name of the image
    URL = dirfilepath + "/compare/" + imgname #generate the URL
    osys = psys() #find out what OS this is being run on

    if osys == "Windows": #if it's run on Windows, use the Windows command line file opening command

        os.system('\"' + URL + '\"')
    
    if osys == "Darwin": #if it's run on MacOS, use the MacOS file opening command

        os.system("open " + URL)

    if osys == "Linux": #if it's Linux, uh, idk we're just gonna use matoplotlib sorry

        image = img.imread(URL) #read in the image
        plt.imshow(image) #display it

    return URL