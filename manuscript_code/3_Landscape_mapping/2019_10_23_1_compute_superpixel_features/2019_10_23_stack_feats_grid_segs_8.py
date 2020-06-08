import numpy as np
import rasterio as rio
import geopandas as gpd
import pandas as pd
import random
#from osgeo import gdal, ogr, osr

from rasterio.mask import mask
from shapely.geometry import mapping, Polygon
from skimage.util import img_as_float

import os as os
os.chdir('E:/SLICUAV_manuscript_code/3_Landscape_mapping/2019_10_23_1_compute_superpixel_features')

# import machinery for this
from trees0.clusterfeatures import ClusterFeatures


grid_shps = gpd.read_file('E:/SLICUAV_manuscript_data/3_Clipped_OMs/'+
                         '2019_08_30_basecamp_grid/'+
                         '2019_08_30_basecamp_50m_grid.shp')

ftprnt_shp = gpd.read_file('E:/SLICUAV_manuscript_data/7_Harapan_shapefiles/'+
                           '2019_09_19_basecamp_footprint_both_years_latlong.shp')

flag = True
#for shp_i in range(all_shps.shape[0]):
for i in range(50):
    shp_flag = True
    random.seed(42)
    shp_i = i + 350
    # Get unique tag for this block
    ths_id = grid_shps['id'][shp_i]
    
    # load images
    rgbtif = rio.open('E:/SLICUAV_manuscript_data/3_Clipped_OMs/2019_09_19_basecamp_grid_with_buffer/all_clips/'+
                            'id_' + str(ths_id) + '_RGB.tif')
    rgbimg = rgbtif.read()
    # Reorder correctly as first dimension is bands
    rgbimg = np.swapaxes(rgbimg,0,2)
    rgbimg = np.swapaxes(rgbimg,0,1)
    rgbtif.close()
                
    mstif = rio.open('E:/SLICUAV_manuscript_data/3_Clipped_OMs/2019_09_19_basecamp_grid_with_buffer/all_clips/'+
                            'id_' + str(ths_id) + '_MS.tif')
    msimg = mstif.read()
    # Reorder correctly as first dimension is bands
    msimg = np.swapaxes(msimg,0,2)
    msimg = np.swapaxes(msimg,0,1)
    mstif.close()
    
    dsmtif = rio.open('E:/SLICUAV_manuscript_data/3_Clipped_OMs/2019_09_19_basecamp_grid_with_buffer/all_clips/'+
                            'id_' + str(ths_id) + '_DSM.tif')
    dsmimg = dsmtif.read()
    # Reorder correctly as first dimension is bands
    dsmimg = np.swapaxes(dsmimg,0,2)
    dsmimg = np.swapaxes(dsmimg,0,1)
    dsmtif.close()
    # Remove redundant third axis
    dsmimg = np.squeeze(dsmimg)
    # Deal with any missing value set to arbitrary negative number
    dsmimg[dsmimg<-1000]=0
    
    ### scale both actual images to 0-1
    rgbimg = img_as_float(rgbimg)
    msimg = msimg/65535
    
    # read in the segmented shapes
    cur_segs = gpd.read_file('E:/SLICUAV_manuscript_data/5_Landscape_superpixels/'+\
                             str(ths_id) +'_SLIC_5000.shp')
    seg_flag = True
    ticker = 0
    for seg_i in range(cur_segs.shape[0]):
        ths_shp = []
        # check if it's in the area for which we collected data
        if not ftprnt_shp.intersects(Polygon(cur_segs['geometry'][seg_i]))[0]:
            ticker += 1
            continue
        tmp_gjson = mapping(cur_segs['geometry'][seg_i])
        ths_shp.append(tmp_gjson)
        del tmp_gjson
   
        # Get RGB mask
        with rio.open('E:/SLICUAV_manuscript_data/3_Clipped_OMs/2019_09_19_basecamp_grid_with_buffer/all_clips/'+
                            'id_' + str(ths_id) + '_RGB.tif') as gtif:
            rgb_clip, clip_affine = mask(gtif,ths_shp,crop=False,all_touched=True)
        rgb_clip = np.swapaxes(rgb_clip,0,2)
        rgb_clip = np.swapaxes(rgb_clip,0,1)
        rgb_mask = np.nonzero(rgb_clip.sum(axis=2))
        del rgb_clip, clip_affine
        
        # Get MS mask
        with rio.open('E:/SLICUAV_manuscript_data/3_Clipped_OMs/2019_09_19_basecamp_grid_with_buffer/all_clips/'+
                            'id_' + str(ths_id) + '_MS.tif') as gtif:
            ms_clip, clip_affine = mask(gtif,ths_shp,crop=False,all_touched=True)
        ms_clip = np.swapaxes(ms_clip,0,2)
        ms_clip = np.swapaxes(ms_clip,0,1)
        ms_clip[ms_clip>65535]=0
        ms_mask = np.nonzero(ms_clip.sum(axis=2))
        del ms_clip, clip_affine
        
        # Get DSM mask
        with rio.open('E:/SLICUAV_manuscript_data/3_Clipped_OMs/2019_09_19_basecamp_grid_with_buffer/all_clips/'+
                            'id_' + str(ths_id) + '_DSM.tif') as gtif:
            dsm_clip, clip_affine = mask(gtif,ths_shp,crop=False,all_touched=True)
        dsm_clip = np.swapaxes(dsm_clip,0,2)
        dsm_clip = np.swapaxes(dsm_clip,0,1)
        dsm_mask = np.nonzero(dsm_clip.sum(axis=2))
        del dsm_clip, clip_affine
        
        feat_struct = ClusterFeatures(shp_i,'NA',rgbimg,rgb_mask,msimg,ms_mask,dsmimg,dsm_mask)
        feat_struct.runFeaturePipeline(thresh=0.5,glcm_steps=3,acor_steps=3,mode=False,HSV=True)
        feat_vec = feat_struct.featStack
        del  rgb_mask, ms_mask, dsm_mask, ths_shp
        
        if flag:
            pd.DataFrame(feat_struct.featList).to_csv('E:/SLICUAV_manuscript_data/5_Landscape_superpixels/features/2019_10_14_variable_names.csv',header=None,index=None)
            pd.DataFrame(feat_struct.featClass).to_csv('E:/SLICUAV_manuscript_data/5_Landscape_superpixels/features/2019_10_14_variable_class.csv',header=None,index=None)
            pd.DataFrame(feat_struct.featHeightInvar).to_csv('E:/SLICUAV_manuscript_data/5_Landscape_superpixels/features/2019_10_14_variable_Hinv.csv',header=None,index=None)
            pd.DataFrame(feat_struct.featSizeInvar).to_csv('E:/SLICUAV_manuscript_data/5_Landscape_superpixels/features/2019_10_14_variable_sizeInv.csv',header=None,index=None)
            pd.DataFrame(feat_struct.featScale).to_csv('E:/SLICUAV_manuscript_data/5_Landscape_superpixels/features/2019_10_14_variable_scale.csv',header=None,index=None)
            flag = False
        if shp_flag:
            X = feat_vec
            Y = [ths_id, cur_segs['ID'][seg_i]]
            shp_flag = False
        else:
            X = np.vstack((X,feat_vec))
            Y = np.vstack((Y,[ths_id,cur_segs['ID'][seg_i]]))
        
        # tidy
        del feat_vec, feat_struct
        ticker +=1
        print('run segment {} of {} during round {} of {}'.format(ticker,cur_segs.shape[0],i+1,50))
    # Save it all
    pd.DataFrame(X).to_csv('E:/SLICUAV_manuscript_data/5_Landscape_superpixels/features/2019_10_14_segment_features_id_' + str(ths_id) + '.csv',header=None,index=None)
    pd.DataFrame(Y).to_csv('E:/SLICUAV_manuscript_data/5_Landscape_superpixels/features/2019_10_14_cluster_labels_id_' + str(ths_id) + '.csv',header=None,index=None)
    del X, Y
    print('now finished round {} of {}'.format(i+1,50))    
    del rgbimg, msimg, dsmimg, ths_id, seg_i, cur_segs, ticker