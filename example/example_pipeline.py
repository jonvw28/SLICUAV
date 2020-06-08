'''
Example script to create superpixels and compute their features for an example
set of data from our study (0.25 ha). We also label some of these pixels based 
upon a polygon for one crown as would be used to build the data for fitting
models to predict across the landscape. To see the full data pipeline from our
work go to the link in this package's readme.

@author: Jonathan Williams
@email: jonvw28@gmail.com

04/06/2020
'''

# Load necessary libraries
import os as os
import random
import geopandas as gpd # for reading shapefiles
import pandas as pd # for saving outputs
import numpy as np

from shapely.geometry import mapping # for creating shape polygons

#os.chdir('E:/SLICUAV')
os.chdir('D:/jon/Documents/Cache/species_id_manuscript/SLICUAV')

# import the SLIC-UAV package
import slicuav

random.seed(42)

# Set up the TreeMap object used for superpixel generation
spixel_obj = slicuav.TreeMap()

# Load into it the RGB image to be used for superpixel generation
spixel_obj.setPath('example/data/Example_RGB.tif')
spixel_obj.loadImage()

# Set clustering approach to SLIC as used in SLICUAV (should be default)
spixel_obj.setSegOpt('SLIC')
spixel_obj.setSegParam((5000,10,1))

# Apply segmentaiton and save it (both as tiff and as a shapefile)
spixel_obj.applySeg()
spixel_obj.saveSeg(saveShape=True,filestem='example/output/SLIC_superpixels')

# Save a version where superpixels are labelled with a 1 if they sufficiently
# overlap the crown polygon we provide as an example (you need to specify the
# attribute and value that identifies the polygon used for labelling)
spixel_obj.saveLabelledSeg('example/data/Example_crown.shp','id',151)



# Now we compute features for each superpixel, looping over all
spixels = gpd.read_file('example/output/SLIC_superpixels_labelled/SLIC_superpixels_labelled.shp')

flag = True # so we only save some variables once
ticker = 0 # so we can see the progress
for spixel_i in range(spixels.shape[0]):
    
    # Pull out the polygon for this superpixel
    shape = []
    shape.append(mapping(spixels['geometry'][spixel_i]))
    
    # create the object which does all the work (with unique id and species label first,
    # then paths to the three forms of imagery and finally the polygon)
    # conditional handles if this superpixel was labelled from our reference data in previous steps
    if spixels['label'][spixel_i] == 1:
        feat_obj = slicuav.SuperpixelFeatures(spixel_i,'Crown - Pulai','example/data/Example_RGB.tif',shape,dsm_path='example/data/Example_DSM.tif',ms_path='example/data/Example_MS.tif')
    else:
        feat_obj = slicuav.SuperpixelFeatures(spixel_i,'None','example/data/Example_RGB.tif',shape,dsm_path='example/data/Example_DSM.tif',ms_path='example/data/Example_MS.tif')
    # Run the piepline to generate features, using brightest 50% of pixels in
    # brightness filtered features, offsests upto 3 for glcm and autocorrelation
    # and including HSV features, but not computing the mode in any category
    feat_obj.runFeaturePipeline(thresh=0.5,glcm_steps=3,acor_steps=3,mode=False,HSV=True)

    # extract the features
    features = feat_obj.featStack
    
    if flag:
        pd.DataFrame(feat_obj.featList).to_csv('example/output/variable_names.csv',header=None,index=None)
        pd.DataFrame(feat_obj.featClass).to_csv('example/output/variable_class.csv',header=None,index=None)
        pd.DataFrame(feat_obj.featHeightInvar).to_csv('example/output/variable_Hinv.csv',header=None,index=None)
        pd.DataFrame(feat_obj.featSizeInvar).to_csv('example/output/variable_sizeInv.csv',header=None,index=None)
        pd.DataFrame(feat_obj.featScale).to_csv('example/output/variable_scale.csv',header=None,index=None)
        X = features
        Y = [feat_obj.tag_id,feat_obj.tag] # save a list of superpixel IDs to associate with each 
        flag = False
    else:
        X = np.vstack((X,features))
        Y = np.vstack((Y,[feat_obj.tag_id,feat_obj.tag] ))
    
    ticker +=1
    print('run superpixel {} of {}'.format(ticker,spixels.shape[0]))

    # Save it all
pd.DataFrame(X).to_csv('example/output/superpixel_features.csv',header=None,index=None)
pd.DataFrame(Y).to_csv('example/output/superpixel_labels.csv',header=None,index=None)