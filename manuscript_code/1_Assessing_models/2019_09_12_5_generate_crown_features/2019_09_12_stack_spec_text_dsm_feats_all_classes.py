import numpy as np
import rasterio as rio
import geopandas as gpd
import pandas as pd
#from osgeo import gdal, ogr, osr

from rasterio.mask import mask
from shapely.geometry import mapping
from skimage.util import img_as_float

import os as os
os.chdir('E:/SLICUAV_manuscript_code/1_Assessing_models/2019_12_5_generate_crown_features')

# import machinery for this
from trees.clusterfeatures import ClusterFeatures


# load the manually drawn files
tree_shps = gpd.read_file('E:/SLICUAV_manuscript_data/2_Crowns'+
                         '2019_01_22_tree_crowns_2018_01_14_OM'+
                         '_merged_chunks_with_extra_data.shp')
misc_shps = gpd.read_file('E:/SLICUAV_manuscript_data/2_Crowns/2019_04_18_misc_category.shp')
pal_ran_shps = gpd.read_file('E:/SLICUAV_manuscript_data/2_Crowns/'+
                             '2019_04_10_palms_and_random_trees.shp')

# match up data names
misc_shps = misc_shps.rename(index = str, columns = {"TAG": "Spp_tag", "TAGSTRING": "tag_string"})
pal_ran_shps = pal_ran_shps.rename(index = str, columns = {"spp_tag": "Spp_tag", "TAGSTRING": "tag_string"})

# Add type column to tree datsets which just says tree
trees_temp = np.repeat('Tree',tree_shps.shape[0]).astype(object)
tree_shps['type'] = trees_temp
pal_ran_temp = np.repeat('Tree',pal_ran_shps.shape[0]).astype(object)
pal_ran_shps['type'] = pal_ran_temp
del trees_temp, pal_ran_temp

# bump up ids as needed
pal_ran_shps['id'] += tree_shps['id'].max()
misc_shps['id'] += pal_ran_shps['id'].max()

# combine it all
all_shps = tree_shps[['id','Spp_tag','tag_string','geometry']]
all_shps = all_shps.append(pal_ran_shps[['id','Spp_tag','tag_string','geometry']],ignore_index=True)
all_shps = all_shps.append(misc_shps[['id','Spp_tag','tag_string','geometry']],ignore_index=True)
del misc_shps, pal_ran_shps, tree_shps

# iterate over all available clips
for shp_i in range(all_shps.shape[0]):
    # Get clip for this tree and its unique tag
    ths_shp = []
    tmp_gjson = mapping(all_shps['geometry'][shp_i])
    ths_shp.append(tmp_gjson)
    del tmp_gjson
    ths_tag = all_shps['tag_string'][shp_i]
    ths_lbl = all_shps['Spp_tag'][shp_i] 
    
    # load images
    rgbtif = rio.open('E:/SLICUAV_manuscript_data/3_Clipped_OMs/2019_08_02_final_clips/all_clips/'+
                       ths_tag + '_RGB.tif')

    rgbimg = rgbtif.read()
    # Reorder correctly as first dimension is bands
    rgbimg = np.swapaxes(rgbimg,0,2)
    rgbimg = np.swapaxes(rgbimg,0,1)
    rgbtif.close()
                
    mstif = rio.open('E:/SLICUAV_manuscript_data/3_Clipped_OMs/2019_08_02_final_clips/all_clips/'+
                       ths_tag + '_MS.tif')
    msimg = mstif.read()
    # Reorder correctly as first dimension is bands
    msimg = np.swapaxes(msimg,0,2)
    msimg = np.swapaxes(msimg,0,1)
    mstif.close()
    
    dsmtif = rio.open('E:/SLICUAV_manuscript_data/3_Clipped_OMs/2019_08_02_final_clips/all_clips/'+
                       ths_tag + '_DSM.tif')
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
    
    # Get RGB mask
    with rio.open('E:/SLICUAV_manuscript_data/3_Clipped_OMs/2019_08_02_final_clips/all_clips/'+
                   ths_tag + '_RGB.tif') as gtif:
        rgb_clip, clip_affine = mask(gtif,ths_shp,crop=False,all_touched=True)
    rgb_clip = np.swapaxes(rgb_clip,0,2)
    rgb_clip = np.swapaxes(rgb_clip,0,1)
    rgb_mask = np.nonzero(rgb_clip.sum(axis=2))
    del rgb_clip, clip_affine
    
    # Get MS mask
    with rio.open('E:/SLICUAV_manuscript_data/3_Clipped_OMs/2019_08_02_final_clips/all_clips/'+
                      ths_tag + '_MS.tif') as gtif:
        ms_clip, clip_affine = mask(gtif,ths_shp,crop=False,all_touched=True)
    ms_clip = np.swapaxes(ms_clip,0,2)
    ms_clip = np.swapaxes(ms_clip,0,1)
    ms_clip[ms_clip>65535]=0
    ms_mask = np.nonzero(ms_clip.sum(axis=2))
    del ms_clip, clip_affine
    
    # Get DSM mask
    with rio.open('E:/SLICUAV_manuscript_data/3_Clipped_OMs/2019_08_02_final_clips/all_clips/'+
                       ths_tag + '_DSM.tif') as gtif:
        dsm_clip, clip_affine = mask(gtif,ths_shp,crop=False,all_touched=True)
    dsm_clip = np.swapaxes(dsm_clip,0,2)
    dsm_clip = np.swapaxes(dsm_clip,0,1)
    dsm_mask = np.nonzero(dsm_clip.sum(axis=2))
    del dsm_clip, clip_affine
    
    #
    feat_struct = ClusterFeatures(shp_i,ths_tag,rgbimg,rgb_mask,msimg,ms_mask,dsmimg,dsm_mask)
    feat_struct.runFeaturePipeline(thresh=0.5,glcm_steps=4,acor_steps=4,mode=False,HSV=True)
    feat_vec = feat_struct.featStack
    del rgbimg, msimg,dsmimg, rgb_mask, ms_mask, dsm_mask, ths_shp
    
    if shp_i == 0:
        X = feat_vec
        Y = [ths_lbl, ths_tag]
        pd.DataFrame(feat_struct.featList).to_csv('2019_09_12_variable_names.csv',header=None,index=None)
        pd.DataFrame(feat_struct.featClass).to_csv('2019_09_12_variable_class.csv',header=None,index=None)
        pd.DataFrame(feat_struct.featHeightInvar).to_csv('2019_09_12_variable_Hinv.csv',header=None,index=None)
        pd.DataFrame(feat_struct.featSizeInvar).to_csv('2019_09_12_variable_sizeInv.csv',header=None,index=None)
        pd.DataFrame(feat_struct.featScale).to_csv('2019_09_12_variable_scale.csv',header=None,index=None)
    else:
        X = np.vstack((X,feat_vec))
        Y = np.vstack((Y,[ths_lbl, ths_tag]))
    
    
    del feat_vec, ths_tag, ths_lbl, feat_struct
    print('now up to round {}'.format(shp_i+1))
    
# Save it all
pd.DataFrame(X).to_csv('2019_09_12_spectral_texture_dsm_features.csv',header=None,index=None)
pd.DataFrame(Y).to_csv('2019_09_12_cluster_labels.csv',header=None,index=None)
del all_shps, shp_i, X, Y