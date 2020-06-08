# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 12:28:01 2019

Trying out automated clustering pipeline

@author: Jon
"""

import geopandas as gpd
import os as os
import random
os.chdir('E:/SLICUAV_manuscript_code/3_Landscape_mapping/2019_09_03_1_generate_grid_superpixels')

# import treemap class
from trees.treemap import TreeMap

# minimum proportion of points that must lie in crown for each segment
min_prop = 0.5

# load the manually drawn files for field trees
grid_shps = gpd.read_file('E:/SLICUAV_manuscript_data/3_Clipped_OMs/'+
                         '2019_08_30_basecamp_grid/'+
                         '2019_08_30_basecamp_50m_grid.shp')


random.seed(42)

# iterate over all available clips
for i in range(100):    
    shp_i = i + 500
    # Get clip for this tree and its unique tag
    ths_id = grid_shps['id'][shp_i]
    
    # Set up holder and load image
    thsTree=TreeMap()
    thsTree.setPath('E:/SLICUAV_manuscript_data/3_Clipped_OMs/2019_08_30_basecamp_grid/all_clips/'+
                            'id_' + str(ths_id) + '_RGB.tif')
    thsTree.loadImage()
    
    # Cluster via SLIC and save as a shapefile
    thsTree.setSegOpt('SLIC')
    thsTree.setSegParam((5000,10,1)) # change to 5000 once shapes is working
    thsTree.applySeg()
    thsfstem = 'E:/SLICUAV_manuscript_data/Landscape_superpixels/' + str(ths_id) + '_SLIC_5000'
    thsTree.saveSeg(saveShape=True,filestem=thsfstem)
del shp_i, thsTree, ths_id