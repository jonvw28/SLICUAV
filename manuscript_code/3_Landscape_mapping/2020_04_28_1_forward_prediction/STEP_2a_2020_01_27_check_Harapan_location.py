import numpy as np
import geopandas as gpd
import pandas as pd
import random

from shapely.geometry import  Polygon


import os as os
os.chdir('E:/SLICUAV_manuscript_code/3_Landscape_mapping/2020_04_28_1_forward_prediction')

grid_shps = gpd.read_file('E:/SLICUAV_manuscript_data/3_Clipped_OMs/'+
                         '2019_08_30_basecamp_grid/'+
                         '2019_08_30_basecamp_50m_grid.shp')
harapan_shp = gpd.read_file('E:/SLICUAV_manuscript_data/7_Harapan_shapefiles/'+
                           '2020_01_27_Harapan_boundary_lat_long.shp')

flag = True
for shp_i in range(grid_shps.shape[0]):
    shp_flag = True
    random.seed(42)
    # Get unique tag for this block
    ths_id = grid_shps['id'][shp_i]
    
    # read in the segmented shapes
    cur_segs = gpd.read_file('E:/SLICUAV_manuscript_data/5_Landscape_superpixels/'+\
                             str(ths_id) +'_SLIC_5000.shp')
    for seg_i in range(cur_segs.shape[0]):
        ths_shp = []
        # check if it's in the area for which we collected data
        if harapan_shp.intersects(Polygon(cur_segs['geometry'][seg_i]))[0]:
            status = 1
        else:
            status = 0
        if shp_flag:
            grid = np.array([ths_id])
            seg = np.array([cur_segs['ID'][seg_i]])
            harapan = np.array([status])
            shp_flag = False
        else:
            grid =  np.concatenate((grid,np.array([ths_id])))
            seg =  np.concatenate((seg,np.array([cur_segs['ID'][seg_i]])))
            harapan = np.concatenate((harapan,np.array([status])))
        #print('run segment {} of {} during round {} of {}'.format(seg_i,cur_segs.shape[0],shp_i+1,grid_shps.shape[0]))
    # Save it all
    pd.DataFrame(np.transpose(np.vstack((grid,seg,harapan)))).to_csv('E:/SLICUAV_manuscript_data/5_Landscape_superpixels/Harapan_status/2020_01_27_harapan_status_' + str(ths_id) + '.csv',header=None,index=None)
    print('now finished round {} of {}'.format(shp_i+1,grid_shps.shape[0]))    