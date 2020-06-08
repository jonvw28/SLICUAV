# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 12:28:01 2019

Trying out automated clustering pipeline

@author: Jon
"""

import geopandas as gpd
import os as os
import random
os.chdir('E:/SLICUAV_manuscript_code/1_Assessing_models/2019_09_12_6_generate_crown_superpixels')

# import treemap class
from trees.treemap import TreeMap

# minimum proportion of points that must lie in crown for each segment
min_prop = 0.5




# load the manually drawn files for palm and random data
pal_ran_shps = gpd.read_file('E:/SLICUAV_manuscript_data/2_Crowns/'+
                             '2019_04_10_palms_and_random_trees.shp')
# match up data names
pal_ran_shps = pal_ran_shps.rename(index = str, columns = {"spp_tag": "Spp_tag", "TAGSTRING": "tag_string"})

random.seed(42)

# iterate over all available clips
for shp_i in range(pal_ran_shps.shape[0]):    
    # Get clip for this tree and its unique tag
    ths_tag = pal_ran_shps['tag_string'][shp_i]
    
    # Set up holder and load image
    thsTree=TreeMap()
    thsTree.setPath('E:/SLICUAV_manuscript_data/3_Clipped_OMs/2019_08_02_final_clips/all_clips/'+
                            ths_tag + '_RGB.tif')
    thsTree.loadImage()
    
    # Cluster via SLIC and save as a shapefile
    thsTree.setSegOpt('SLIC')
    thsTree.setSegParam((5000,10,1)) # change to 5000 once shapes is working
    thsTree.applySeg()
    thsfstem = 'E:/SLICUAV_manuscript_data/4_Crown_superpixels/' + ths_tag + '_SLIC_5000'
    thsTree.saveSeg(saveShape=True,filestem=thsfstem)
    # Generate the labelling on the SLIC shapefile
    thsTree.saveLabelledSeg('E:/SLICUAV_manuscript_data/2_Crowns/'+
                             '2019_04_10_palms_and_random_trees.shp','TAGSTRING',ths_tag)
del shp_i, thsTree, ths_tag