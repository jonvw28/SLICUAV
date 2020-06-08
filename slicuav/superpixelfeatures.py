# -*- coding: utf-8 -*-
"""
Structure to compute features for a given tree crown cluster or object created 
by SLIC-UAV

@author: Jonathan Williams
jonvw28@gmail.com
21/05/2020
"""

# modules
import numpy as np
import scipy as sp
import cv2 as cv2
import mahotas.features as mht
import rasterio as rio
import geopandas as gpd

# methods
from rasterio.mask import mask
from skimage.util import img_as_float
from skimage.feature.texture import local_binary_pattern
from skimage.exposure import rescale_intensity
from scipy.ndimage import convolve

# Create class definition
class SuperpixelFeatures:
    """
    Structure to compute features for a given tree crown cluster or object created 
    by SLICUAV

    Call

    output = SuperpixelFeatures(tag_id,tag,rgb_path,shape,dsm_path,ms_path)

    Inputs:

    tag_id - a unique numeric ID to assocaite for this object labelling each cluster that is analysed
    tag - the species three letter tag for this species (eg MAC)
    rgb_path - the path to the RGB imagery file (should be a tiff)
    shape - polygon from a shapefile for the region of the imagery to be used for feature generation
    ms_path - the path to the MS imagery file (should be a tiff)
    dsm_path - the path to the DSM imagery file (should be a tiff)


    shape can be created in this manner (as a snippet of code)
        import geopandas as gpd
        from shapely.geometry import mapping
        shps = gpd.read_file('shapefile.shp')
        shape = []
        shape.append(mapping(shps['geometry'][i]))
    where i is the index within python of the polygon within the file 
    shapefile.shp. This is a zero-indexed sequential index (ie the third feature
    would need i=2, and a file with only 1 polygon would use 0)

    general pipeline:

    cluster = ClusterFeatures(tag_id,tag,rgb_path,ms_path,dsm_path,shape)
    cluster.runFeaturePipeline()

    then get the following attributes from the structure for use in analysis:

    cluster.featStack - the features as 1-D vector
    cluster.featList - shorthand names of all the features
    cluster.featClass - a list of feature class types (listed below)
    cluster.featSizeInvar - a list of booleans for whether each feature is size independent for the cluster
    cluster.featHeightInvar - a list of booleans for whether each feature is height independent for the cluster
    cluster.featScale - a list of the relevant scale used in computing a feature (or 0 if the feature has no scale)

    valid classes for features
    rgb_band - stats on the RGB bands
    rgb_top - stats on the top 50% of RGB band pixels (based on lightness in cielab space)
    rgb_ind - stats on RGB indices - computed from RGB imagery
    rgb_glcm - stats from GLCM analysis of RGB image
    rgb_acor - autocorrelation of RGB imagery
    rgb_lbp - LBP histograms for RGB imagery
    rgb_laws - stats on laws texture features on RGB imagery
    rgb_hsv - stats on HSV space from RGB imagery
    ms_band - stats on the MS bands
    ms_ind - stats on MS indices - computed from MS imagery
    ms_glcm - stats from GLCM analysis of MS image
    ms_acor - autocorrelation of MS imagery
    ms_lbp - LBP histograms for MS imagery
    ms_laws - stats on laws texture features on MS imagery
    dsm_raw - stats on raw DSM raster
    dsm_glcm - stats from GLCM on quantised DSM raster
    dsm_acor - autocorrelation of DSM raster
    dsm_lbp - LBP histograms for DSM raster
    dsm_laws - laws features stats for DSM raster

    NB

    Methods exist for computing subsets of the features if you wish to only have
    some of these. To do this don't call the method runFeaturePipeline() but instead
    call each method you need for the relevant features then call the method
    stackFeats() to collate all of these to allow access by the attributes as above.

    Methods for subsets of features:

    RGB
    .createRGBBandFeats(mode=False) - generate rgb_band features, with option to 
                                      include mode of each band
    .createRGBThreshFeats(thresh=0.5,mode=False) - generate rgb_top features, with
                                                   the top thresh proportion of
                                                   brightest pixels
    .createRGBGLCMfeats(distance=1) - generate rgb_glcm features, using an offset of
                                      distance pixels
    .createRGBautoCorFeats(distance=1) - generate rgb_acor features, using an offset
                                         of distance pixels
    .createRGBLBPFeats(distance=1) - generate rgb_lbp features, using an offset of
                                     distance pixels
    .createRGBLawsFeats() - generate rgb_laws features
    .createHSVFeats(mode=False) - create rgb_hsv features, with optinal mode

    MS
    .createMSBandFeats(mode=False) - generate ms_band features, with option to 
                                     include mode of each band
    .createMSGLCMfeats(distance=1) - generate ms_glcm features, using an offset of
                                     distance pixels
     .createMSautoCorFeats(distance=1) - generate ms_acor features, using an offset
                                         of distance pixels
    .createMSLBPFeats(distance=1) - generate ms_lbp features, using an offset of
                                     distance pixels
    .createMSLawsFeats() - generate ms_laws features

    INDICES
    .createSpecIndices() - compute several RGB and MS vegetation indices (as in the
                           asscoaited manuscript)
    .createRGBIndFeats(mode=False) - compute rgb_ind features, with optional mode
    .createMSIndFeats(mode=False) - compute ms_ind features, with optional mode

    DSM
    .createDSMRawFeats(mode=False) - compute dsm_raw features, with optional mode
    .createDSMGLCMfeats(distance=1) - generate dsm_glcm features, using an offset of
                                      distance pixels
     .createDSMautoCorFeats(distance=1) - generate dsm_acor features, using offset
                                          of distance pixels
    .createDSMLBPFeats(distance=1) - generate dsm_lbp features, using an offset of
                                     distance pixels
    .createDSMLawsFeats() - generate dsm_laws features

    See each method in the class to see the full set of features and how they are
    computed

    @author: Jonathan Williams
    jonvw28@gmail.com
    21/05/2020
    """
    def __init__(self,tag_id,tag,rgb_path,shape,ms_path=None,dsm_path=None):
        # store inputs
        self.tag_id = tag_id
        self.tag = tag
        self.__loadImg(rgb_path,'RGB')
        self.__generateMask(rgb_path,shape,'RGB')
        rgb_pixels = self.rgb_img[self.rgb_mask]
        self.rgb_pixels = rgb_pixels[:,:3:]
        # clip the raw image to a buffer of 20 pixels around the region of interest
        rgb_bounds = [np.maximum(self.rgb_mask[0].min()-20,0),np.minimum(self.rgb_mask[0].max()+20,self.rgb_img.shape[0]),np.maximum(self.rgb_mask[1].min()-20,0),np.minimum(self.rgb_mask[1].max()+20,self.rgb_img.shape[1])]
        self.rgb_img_clip = self.rgb_img[rgb_bounds[0]:rgb_bounds[1]+1:,rgb_bounds[2]:rgb_bounds[3]+1:,:]
        self.rgb_mask_clip = (self.rgb_mask[0]-rgb_bounds[0],self.rgb_mask[1]-rgb_bounds[2])
        
        if ms_path is not None:
            self.__loadImg(ms_path,'MS')
            self.__generateMask(ms_path,shape,'MS')
            ms_pixels = self.ms_img[self.ms_mask]
            self.ms_pixels = ms_pixels[:,:4:]
            ms_bounds = [np.maximum(self.ms_mask[0].min()-20,0),np.minimum(self.ms_mask[0].max()+20,self.ms_img.shape[0]),np.maximum(self.ms_mask[1].min()-20,0),np.minimum(self.ms_mask[1].max()+20,self.ms_img.shape[1])]
            self.ms_img_clip = self.ms_img[ms_bounds[0]:ms_bounds[1]+1:,ms_bounds[2]:ms_bounds[3]+1:,:]
            self.ms_mask_clip = (self.ms_mask[0]-ms_bounds[0],self.ms_mask[1]-ms_bounds[2])

        if dsm_path is not None:
            self.__loadImg(dsm_path,'DSM')
            self.__generateMask(dsm_path,shape,'DSM')
            dsm_pixels = self.dsm_img[self.dsm_mask]
            self.dsm_pixels = dsm_pixels
            dsm_bounds = [np.maximum(self.dsm_mask[0].min()-20,0),np.minimum(self.dsm_mask[0].max()+20,self.dsm_img.shape[0]),np.maximum(self.dsm_mask[1].min()-20,0),np.minimum(self.dsm_mask[1].max()+20,self.dsm_img.shape[1])]
            self.dsm_img_clip = self.dsm_img[dsm_bounds[0]:dsm_bounds[1]+1:,dsm_bounds[2]:dsm_bounds[3]+1:]
            self.dsm_mask_clip = (self.dsm_mask[0]-dsm_bounds[0],self.dsm_mask[1]-dsm_bounds[2])
    
    def __loadImg(self,imagePath,type): # load in images from a path
        gtif = rio.open(imagePath)
        gimg = gtif.read()
        # Reorder correctly as first dimension is bands
        gimg = np.swapaxes(gimg,0,2)
        gimg = np.swapaxes(gimg,0,1)
        gtif.close()
        # now add to struct
        if(type=='RGB'):
            # Scale to 0-1
            self.rgb_img = img_as_float(gimg)
        elif(type=='MS'):
            # Scale to 0-1
            self.ms_img = gimg/65535
        elif(type=='DSM'):
            # Remove redundant third axis
            gimg = np.squeeze(gimg)
            # Deal with any missing value set to arbitrary negative number
            gimg[gimg<-1000]=0
            self.dsm_img = gimg
        else:
            raise ValueError('It appears you are trying to use a submethod \
            constucted to load images that is not designed for use externally \
            and have set an illegal value. See the source code to see what this \
            function does.')
    
    def __generateMask(self,imagePath,shape,type): # get masks for the imagery within the polygon provided
        with rio.open(imagePath) as gtif:
            clip, clip_affine = mask(gtif,shape,crop=False,all_touched=True)
        # Reorder correctly as first dimension is bands
        clip = np.swapaxes(clip,0,2)
        clip = np.swapaxes(clip,0,1)
        msk = np.nonzero(clip.sum(axis=2))
        if(type=='RGB'):
            self.rgb_mask = msk
        elif(type=='MS'):
            self.ms_mask = msk
        elif(type=='DSM'):
            self.dsm_mask = msk
        else:
            raise ValueError('It appears you are trying to use a submethod \
            constucted to create masks that is not designed for use externally \
            and have set an illegal value. See the source code to see what this \
            function does.')
    
    def createRGBBandFeats(self,mode=False):
        '''
        Compute stats on each of the RGB bands
        only compute mode value if mode = True
        '''
        self.rgb_band_max = self.rgb_pixels.max(axis=0) # max value
        self.rgb_band_min = self.rgb_pixels.min(axis=0) # min value
        self.rgb_band_mean = self.rgb_pixels.mean(axis=0) # mean value
        self.rgb_band_std = self.rgb_pixels.std(axis=0) # standard deviation
        self.rgb_band_median = np.median(self.rgb_pixels,axis=0) # median value
        self.rgb_band_cov = np.divide(self.rgb_band_std,self.rgb_band_mean) # coefficient of variation
        self.rgb_band_skew = sp.stats.skew(self.rgb_pixels,axis=0) # skewness
        self.rgb_band_kurt = sp.stats.kurtosis(self.rgb_pixels,axis=0) # kurtosis
        self.rgb_band_sum = self.rgb_pixels.sum(axis=0) # sum of all values
        self.rgb_band_rng = self.rgb_band_max-self.rgb_band_min # range of values
        self.rgb_band_rngsig = np.divide(self.rgb_band_rng,self.rgb_band_std) # range in number of sds
        self.rgb_band_rngmean = np.divide(self.rgb_band_rng,self.rgb_band_mean) # range expressed in means
        if(mode):
            self.rgb_band_mode = sp.stats.mode(self.rgb_pixels,axis=0)[0][0] # modal value
        self.rgb_band_deciles = np.percentile(
                                    self.rgb_pixels,np.linspace(10,90,9),axis=0) # deciles of band
        self.rgb_band_quartiles = np.percentile(self.rgb_pixels,[25,75],axis=0) # quartiles of band
        self.rgb_band_iqr = self.rgb_band_quartiles[1,:]-self.rgb_band_quartiles[0,:] # iqr
        self.rgb_band_iqrsig = np.divide(self.rgb_band_iqr,self.rgb_band_std) # iqr expressed in sds
        self.rgb_band_iqrmean = np.divide(self.rgb_band_iqr,self.rgb_band_mean) # iqr expressed in means
        self.rgb_band_ratio = self.rgb_band_mean[:3:]/np.sum(self.rgb_band_mean[:3:])# ratio of band to sum of bands
    
    def createRGBThreshFeats(self,thresh=0.5,mode=False):
        '''
        compute bandwise stats only on top thresh proportion of pixels based
        on L in cielab space
        '''
        lab_img = cv2.cvtColor(self.rgb_img[:,:,:3:].astype('float32'), cv2.COLOR_RGB2Lab)
        self.lab_pixels = lab_img[self.rgb_mask]
        lab_thresh = np.percentile(self.lab_pixels,100*thresh,axis=0)
        top_pixels = self.rgb_pixels[self.lab_pixels[:,0]>=lab_thresh[0],:]
        self.top_rgb_max = top_pixels.max(axis=0) # max value
        self.top_rgb_min = top_pixels.min(axis=0) # min value
        self.top_rgb_mean = top_pixels.mean(axis=0) # mean value
        self.top_rgb_std = top_pixels.std(axis=0) # standard deviation
        self.top_rgb_median = np.median(top_pixels,axis=0) # median value
        self.top_rgb_cov = np.divide(self.top_rgb_std,self.top_rgb_mean) # coeffiient of variation
        self.top_rgb_skew = sp.stats.skew(top_pixels,axis=0) # skewness
        self.top_rgb_kurt = sp.stats.kurtosis(top_pixels,axis=0) # kurtosis
        self.top_rgb_sum = top_pixels.sum(axis=0) # sum of all values
        self.top_rgb_rng = self.top_rgb_max-self.top_rgb_min # range
        self.top_rgb_rngsig = np.divide(self.top_rgb_rng,self.top_rgb_std) # range in sds
        self.top_rgb_rngmean = np.divide(self.top_rgb_rng,self.top_rgb_mean) # range in means
        if(mode):
            self.top_rgb_mode = sp.stats.mode(stop_pixels,axis=0)[0][0] # modal value
        self.top_rgb_deciles = np.percentile(
                                    top_pixels,np.linspace(10,90,9),axis=0) # deciles
        self.top_rgb_quartiles = np.percentile(top_pixels,[25,75],axis=0) # quartile
        self.top_rgb_iqr = self.top_rgb_quartiles[1,:]-self.top_rgb_quartiles[0,:] #iqr
        self.top_rgb_iqrsig = np.divide(self.top_rgb_iqr,self.top_rgb_std) #iqr in sds
        self.top_rgb_iqrmean = np.divide(self.top_rgb_iqr,self.top_rgb_mean) # iqr in means
        self.top_rgb_ratio = self.top_rgb_mean[:3:]/np.sum(self.top_rgb_mean[:3:]) # ratio compared to all bands
    
    def __createGLCMimgs(self): # maxes are used to ensure where there are many 0s (when rounded) that this method works
        glcm_rgb = np.zeros((self.rgb_img.shape[0],self.rgb_img.shape[1],3))
        glcm_rgb[self.rgb_mask[0],self.rgb_mask[1],:]=np.maximum(self.rgb_img[self.rgb_mask[0],self.rgb_mask[1],:3:],0.004*np.ones((self.rgb_mask[0].__len__(),3)))
        glcm_rgb = cv2.cvtColor(glcm_rgb.astype('float32'), cv2.COLOR_RGB2GRAY)
        glcm_rgb = 255*glcm_rgb
        # remove zeros
        glcm_rgb = glcm_rgb[~np.all(glcm_rgb==0,axis=1),:]
        glcm_rgb = glcm_rgb[:,~np.all(glcm_rgb==0,axis=0)]
        self.glcm_rgb_img = glcm_rgb.astype('uint8')
        # MS
        if hasattr(self,'ms_img'):
            glcm_ms= np.zeros((self.ms_img.shape[0],self.ms_img.shape[1],4))
            glcm_ms[self.ms_mask[0],self.ms_mask[1],:]=np.maximum(self.ms_img[self.ms_mask[0],self.ms_mask[1],:4:],0.004*np.ones((self.ms_mask[0].__len__(),4)))
            glcm_ms = 255*glcm_ms
            # remove zeros
            glcm_ms = glcm_ms[~np.all(np.all(glcm_ms==0,axis=1),axis=1),:,:]
            glcm_ms = glcm_ms[:,~np.all(np.all(glcm_ms==0,axis=0),axis=1),:]
            self.glcm_ms_img = glcm_ms.astype('uint8')
    
    def __padRGBGLCM(self,glcm_step): # Pad glcm
        counter = 0
        while(counter < glcm_step and (self.glcm_rgb_img.shape[0]<glcm_step+1 or self.glcm_rgb_img.shape[1]<glcm_step+1)):
            frame = np.zeros((self.glcm_rgb_img.shape[0]+2,self.glcm_rgb_img.shape[1]+2,self.glcm_rgb_img.shape[2]),dtype=self.glcm_rgb_img.dtype)
            frame[1:self.glcm_rgb_img.shape[0]+1,1:self.glcm_rgb_img.shape[1]+1,:]=self.glcm_rgb_img
            self.glcm_rgb_img = frame
            counter = counter + 1
        kern = np.array([[1,1,1],[1,0,1],[1,1,1]])
        tmpimg = self.glcm_rgb_img
        convimg = convolve(self.glcm_rgb_img.astype('float32'),kern,mode='reflect')
        maskimg = np.where(self.glcm_rgb_img!=0,1.0,0.0)
        convmask = convolve(maskimg.astype('float32'),kern,mode='reflect')
        idx1 = np.nonzero(convimg)
        convimg[idx1] = np.maximum(convimg[idx1],np.ones((idx1[0].__len__())))/convmask[idx1]
        convimg = convimg.astype('uint8')
        idx2 = tmpimg==0
        tmpimg[idx2] = convimg[idx2]
        self.glcm_rgb_img = tmpimg
    
    def createRGBGLCMfeats(self,distance=1):
        '''
        Compute features based on GLCM for RGB imagery with offset 
        distance pixels
        '''
        if not hasattr(self,'glcm_rgb_img'):
            self.__createGLCMimgs()
        success = False
        counter = 0
        while(counter < distance and not success):
            try:
                glcm_rgb_vals = mht.haralick(self.glcm_rgb_img,ignore_zeros=True,
                                            return_mean_ptp=True,distance=distance)
                success = True
            except:
                self.__padRGBGLCM(distance)
                counter = counter + 1
        if not hasattr(self,'glcm_rgb_vals'):
            self.glcm_rgb_vals = glcm_rgb_vals
        else:
            self.glcm_rgb_vals = np.concatenate((self.glcm_rgb_vals,
                                                    glcm_rgb_vals))
        if not hasattr(self,'glcm_rgb_dist'):
            self.glcm_rgb_dist = [distance]
        else:
            self.glcm_rgb_dist.append(distance)
    
    def __imgAutocorrelate(self,img,dx,dy):
        if dy >=0:
            im1 = img_as_float(img[:img.shape[0]-dx:,:img.shape[1]-dy:])
            im2 = img_as_float(img[dx:img.shape[0]:,dy:img.shape[1]:])
        else:
            mody = -dy
            im1 = img_as_float(img[:img.shape[0]-dx:,mody:img.shape[1]:])
            im2 = img_as_float(img[dx:img.shape[0]:,:img.shape[1]-mody:])
        # set to mean zero
        im1_mean = im1[np.nonzero(im1)].mean()
        im1[np.nonzero(im1)] -= im1_mean
        im2_mean = im2[np.nonzero(im2)].mean()
        im2[np.nonzero(im2)] -= im2_mean
        nom = np.multiply(im1,im2).sum()
        # average both sub-images
        denom = (np.multiply(im1,im1).sum() + np.multiply(im2,im2).sum())/2
        return nom/denom
    
    def createRGBautoCorFeats(self,distance=1):
        '''
        Compute features based on autocorrelation for RGB imagery with offset 
        distance pixels
        '''
        if not hasattr(self,'glcm_rgb_img'):
            self.__createGLCMimgs()
        N = self.__imgAutocorrelate(self.glcm_rgb_img,0,distance)
        NE = self.__imgAutocorrelate(self.glcm_rgb_img,distance,distance)
        E = self.__imgAutocorrelate(self.glcm_rgb_img,distance,0)
        SE  = self.__imgAutocorrelate(self.glcm_rgb_img,distance,-distance)
        acors = np.array([N,NE,E,SE])
        acfeats = np.array([acors.mean(),acors.max()-acors.min()])
        if not hasattr(self,'acor_rgb_vals'):
            self.acor_rgb_vals = acfeats
        else:
            self.acor_rgb_vals = np.concatenate((self.acor_rgb_vals,
                                                    acfeats))
        if not hasattr(self,'acor_rgb_dist'):
            self.acor_rgb_dist = [distance]
        else:
            self.acor_rgb_dist.append(distance)
    
    def createRGBLBPFeats(self,distance=1):
        '''
        Compute features based on Local binary patterns for RGB imagery with 
        offset distance pixels
        '''
        if not distance in [1,2,3]:
            raise ValueError('distance can only be 1,2 or 3')
        grayimg = cv2.cvtColor(self.rgb_img_clip.astype('float32'), cv2.COLOR_RGB2GRAY)
        lbp_img = local_binary_pattern(grayimg,8*distance,distance,method='uniform')
        lbp_pix = lbp_img[self.rgb_mask_clip]
        unique, counts = np.unique(lbp_pix, return_counts = True)
        count_table = np.zeros([2+distance*8])
        count_table[unique.astype('int')]=counts
        count_table = count_table/count_table.sum()
        if not hasattr(self,'lbp_rgb_vals'):
            self.lbp_rgb_vals = count_table
        else:
            self.lbp_rgb_vals = np.concatenate((self.lbp_rgb_vals,count_table))
        if not hasattr(self,'lbp_rgb_dist'):
            self.lbp_rgb_dist = [distance]
        else:
            self.lbp_rgb_dist.append(distance)
    
    def createRGBLawsFeats(self):
        '''
        Compute features based on Laws' texture energy for RGB imagery with 
        offset distance pixels
        '''
        grayimg = cv2.cvtColor(self.rgb_img_clip.astype('float32'), cv2.COLOR_RGB2GRAY)
        mean_15 = convolve(grayimg,np.ones([15,15])/225,mode='reflect')
        norm_gray = grayimg-mean_15
        del mean_15, grayimg
        # Constuct filter bank
        L5 = np.array([1,4,6,4,1])
        E5 = np.array([-1,-2,0,2,1])
        S5 = np.array([-1,0,2,0,-1])
        R5 = np.array([1,-4,6,-4,1])
        W5 = np.array([-1,2,0,-2,1])
        filtbank = [L5,E5,S5,R5,W5]
        del L5, E5, S5, R5, W5
        filtgrid = np.zeros([5,5,5,5])
        for i in range(5):
            for j in range(5):
                filtgrid[i,j,:,:]=(np.outer(filtbank[i],filtbank[j]))
        del filtbank
        # compute features
        lawsFeat = np.zeros([14,2])
        count_i = 0;
        for i in range(5):
            for j in range(5):
                if j < i or (i==0 and j ==0):
                    continue
                if j==i:
                    convimg = convolve(norm_gray,filtgrid[i,j],mode='reflect')
                    lawsimg = convolve(np.absolute(convimg),np.ones([15,15]),mode='reflect')
                    lawsFeat[count_i,0] = lawsimg[self.rgb_mask_clip].mean()
                    lawsFeat[count_i,1] = lawsimg[self.rgb_mask_clip].std()
                    count_i += 1
                else:
                    convimg1 = np.absolute(convolve(norm_gray,filtgrid[i,j],mode='reflect'))
                    convimg2 = np.absolute(convolve(norm_gray,filtgrid[j,i],mode='reflect'))
                    lawsimg = convolve(convimg1+convimg2,np.ones([15,15])/2,mode='reflect')
                    lawsFeat[count_i,0] = lawsimg[self.rgb_mask_clip].mean()
                    lawsFeat[count_i,1] = lawsimg[self.rgb_mask_clip].std()
                    count_i += 1
        self.laws_rgb_feats = lawsFeat
    
    def createHSVFeats(self,mode=False):
        '''
        Compute HSV stats
        only compute mode value if mode = True
        '''
        hsv_img = cv2.cvtColor(self.rgb_img[:,:,:3:].astype('float32'), cv2.COLOR_RGB2HSV)
        self.hsv_pixels = hsv_img[self.rgb_mask]
        self.hsv_max = self.hsv_pixels.max(axis=0)
        self.hsv_min = self.hsv_pixels.min(axis=0)
        self.hsv_mean = self.hsv_pixels.mean(axis=0)
        self.hsv_std = self.hsv_pixels.std(axis=0)
        self.hsv_median = np.median(self.hsv_pixels,axis=0)
        self.hsv_cov = np.divide(self.hsv_std,self.hsv_mean)
        self.hsv_skew = sp.stats.skew(self.hsv_pixels,axis=0)
        self.hsv_kurt = sp.stats.kurtosis(self.hsv_pixels,axis=0)
        self.hsv_sum = self.hsv_pixels.sum(axis=0)
        self.hsv_rng = self.hsv_max-self.hsv_min
        self.hsv_rngsig = np.divide(self.hsv_rng,self.hsv_std)
        self.hsv_rngmean = np.divide(self.hsv_rng,self.hsv_mean)
        if(mode):
            self.hsv_mode = sp.stats.mode(self.hsv_pixels,axis=0)[0][0]
        self.hsv_deciles = np.percentile(
                                    self.hsv_pixels,np.linspace(10,90,9),axis=0)
        self.hsv_quartiles = np.percentile(self.hsv_pixels,[25,75],axis=0)
        self.hsv_iqr = self.hsv_quartiles[1,:]-self.hsv_quartiles[0,:]
        self.hsv_iqrsig = np.divide(self.hsv_iqr,self.hsv_std)
        self.hsv_iqrmean = np.divide(self.hsv_iqr,self.hsv_mean)
    
    def createMSBandFeats(self,mode=False):
        '''
        Compute stats on each of the MS bands
        only compute mode value if mode = True
        '''
        if hasattr(self,'ms_pixels'):
            self.ms_band_max = self.ms_pixels.max(axis=0)
            self.ms_band_min = self.ms_pixels.min(axis=0)
            self.ms_band_mean = self.ms_pixels.mean(axis=0)
            self.ms_band_std = self.ms_pixels.std(axis=0)
            self.ms_band_median = np.median(self.ms_pixels,axis=0)
            self.ms_band_cov = np.divide(self.ms_band_std,self.ms_band_mean)
            self.ms_band_skew = sp.stats.skew(self.ms_pixels,axis=0)
            self.ms_band_kurt = sp.stats.kurtosis(self.ms_pixels,axis=0)
            self.ms_band_sum = self.ms_pixels.sum(axis=0)
            self.ms_band_rng = self.ms_band_max-self.ms_band_min
            self.ms_band_rngsig = np.divide(self.ms_band_rng,self.ms_band_std)
            self.ms_band_rngmean = np.divide(self.ms_band_rng,self.ms_band_mean)
            if(mode):
                self.ms_band_mode = sp.stats.mode(self.ms_pixels,axis=0)[0][0]
            self.ms_band_deciles = np.percentile(
                                        self.ms_pixels,np.linspace(10,90,9),axis=0)
            self.ms_band_quartiles = np.percentile(self.ms_pixels,[25,75],axis=0)
            self.ms_band_iqr = self.ms_band_quartiles[1,:]-self.ms_band_quartiles[0,:]
            self.ms_band_iqrsig = np.divide(self.ms_band_iqr,self.ms_band_std)
            self.ms_band_iqrmean = np.divide(self.ms_band_iqr,self.ms_band_mean)
            self.ms_band_ratio = self.ms_band_mean[:4:]/np.sum(self.ms_band_mean[:4:])
        else:
            raise AttributeError('No Multispectral imagery provided')

    
    def __padMSGLCM(self,glcm_step): # Pad glcm
        counter = 0
        while(counter < glcm_step and (self.glcm_ms_img.shape[0]<glcm_step+1 or self.glcm_ms_img.shape[1]<glcm_step+1)):
            frame = np.zeros((self.glcm_ms_img.shape[0]+2,self.glcm_ms_img.shape[1]+2,self.glcm_ms_img.shape[2]),dtype=self.glcm_ms_img.dtype)
            frame[1:self.glcm_ms_img.shape[0]+1,1:self.glcm_ms_img.shape[1]+1,:]=self.glcm_ms_img
            self.glcm_ms_img = frame
            counter = counter + 1
        kern = np.array([[1,1,1],[1,0,1],[1,1,1]])
        tmpimg = self.glcm_ms_img
        for band in range(4):
            convimg = convolve(self.glcm_ms_img[:,:,band].astype('float32'),kern,mode='reflect')
            maskimg = np.where(self.glcm_ms_img[:,:,band]!=0,1.0,0.0)
            convmask = convolve(maskimg.astype('float32'),kern,mode='reflect')
            idx1 = np.nonzero(convimg)
            convimg[idx1] = np.maximum(convimg[idx1],np.ones((idx1[0].__len__())))/convmask[idx1]
            convimg = convimg.astype('uint8')
            idx2 = tmpimg[:,:,band]==0
            tmpimg[idx2,band] = convimg[idx2]
        self.glcm_ms_img = tmpimg
    
    def createMSGLCMfeats(self,distance=1):
        '''
        Compute features based on GLCM for MS imagery with offset 
        distance pixels
        '''
        if hasattr(self,'ms_img'):
            if not hasattr(self,'glcm_ms_img'):
                self.__createGLCMimgs()
            success = False
            counter = 0
            while(counter < distance and not success):
                try:
                    glcm_ms_vals = np.vstack((
                        mht.haralick(self.glcm_ms_img[:,:,0],ignore_zeros=True,
                                                return_mean_ptp=True,distance=distance),
                        mht.haralick(self.glcm_ms_img[:,:,1],ignore_zeros=True,
                                                return_mean_ptp=True,distance=distance),
                        mht.haralick(self.glcm_ms_img[:,:,2],ignore_zeros=True,
                                                return_mean_ptp=True,distance=distance),
                        mht.haralick(self.glcm_ms_img[:,:,3],ignore_zeros=True,
                                                return_mean_ptp=True,distance=distance)
                        ))
                    success = True
                except:
                    self.__padMSGLCM(distance)
                    counter = counter + 1
            glcm_ms_vals = np.vstack((glcm_ms_vals,glcm_ms_vals.mean(axis=0))).flatten('C')
            if not hasattr(self,'glcm_ms_vals'):
                self.glcm_ms_vals = glcm_ms_vals
            else:
                self.glcm_ms_vals = np.concatenate((self.glcm_ms_vals,
                                                        glcm_ms_vals))
            if not hasattr(self,'glcm_ms_dist'):
                self.glcm_ms_dist = [distance]
            else:
                self.glcm_ms_dist.append(distance)
        else:
            raise AttributeError('No Multispectral imagery provided')
    
    def createMSautoCorFeats(self,distance=1):
        '''
        Compute features based on autocorrelation for MS imagery with offset 
        distance pixels
        '''
        if hasattr(self,'ms_img'):
            if not hasattr(self,'glcm_ms_img'):
                self.__createGLCMimgs()
            acfeats = np.empty([4,2])
            for acor_i in range(4):
                N = self.__imgAutocorrelate(self.glcm_ms_img[:,:,acor_i],0,distance)
                NE = self.__imgAutocorrelate(self.glcm_ms_img[:,:,acor_i],distance,distance)
                E = self.__imgAutocorrelate(self.glcm_ms_img[:,:,acor_i],distance,0)
                SE  = self.__imgAutocorrelate(self.glcm_ms_img[:,:,acor_i],distance,-distance)
                acors = np.array([N,NE,E,SE])
                acfeats[acor_i,:] = np.array([acors.mean(),acors.max()-acors.min()])
            acfeats = np.vstack((acfeats,acfeats.mean(axis=0))).flatten('C')
            if not hasattr(self,'acor_ms_vals'):
                self.acor_ms_vals = acfeats
            else:
                self.acor_ms_vals = np.concatenate((self.acor_ms_vals,
                                                        acfeats))
            if not hasattr(self,'acor_ms_dist'):
                self.acor_ms_dist = [distance]
            else:
                self.acor_ms_dist.append(distance)
        else:
            raise AttributeError('No Multispectral imagery provided')
    
    def createMSLBPFeats(self,distance=1):
        '''
        Compute features based on Local binary patterns for MS imagery with 
        offset distance pixels
        '''
        if hasattr(self,'ms_img'):
            if not distance in [1,2,3]:
                raise ValueError('distance can only be 1,2 or 3')
            count_table = np.zeros([4,2+distance*8])
            for lbp_i in range(4):
                lbp_img = local_binary_pattern(self.ms_img_clip[:,:,lbp_i],8*distance,distance,method='uniform')
                lbp_pix = lbp_img[self.ms_mask_clip]
                unique, counts = np.unique(lbp_pix, return_counts = True)
                table = np.zeros([2+distance*8])
                table[unique.astype('int')]=counts
                count_table[lbp_i,:] = table/table.sum()
            count_table = np.vstack((count_table,count_table.mean(axis=0))).flatten('C')
            if not hasattr(self,'lbp_ms_vals'):
                self.lbp_ms_vals = count_table
            else:
                self.lbp_ms_vals = np.concatenate((self.lbp_ms_vals,count_table))
            if not hasattr(self,'lbp_ms_dist'):
                self.lbp_ms_dist = [distance]
            else:
                self.lbp_ms_dist.append(distance)
        else:
            raise AttributeError('No Multispectral imagery provided')
    
    def createMSLawsFeats(self):
        '''
        Compute features based on Laws' texture energy for MS imagery with 
        offset distance pixels
        '''
        if hasattr(self,'ms_img'):
            # Construct filter bank
            L5 = np.array([1,4,6,4,1])
            E5 = np.array([-1,-2,0,2,1])
            S5 = np.array([-1,0,2,0,-1])
            R5 = np.array([1,-4,6,-4,1])
            W5 = np.array([-1,2,0,-2,1])
            filtbank = [L5,E5,S5,R5,W5]
            del L5, E5, S5, R5, W5
            filtgrid = np.zeros([5,5,5,5])
            for i in range(5):
                for j in range(5):
                    filtgrid[i,j,:,:]=(np.outer(filtbank[i],filtbank[j]))
            del filtbank
            # compute features
            lawsFeat = np.zeros([4,28])
            for band in range(4):
                mean_15 = convolve(self.ms_img_clip[:,:,band],np.ones([15,15])/225,mode='reflect')
                norm_gray = self.ms_img_clip[:,:,band]-mean_15
                del mean_15
                count_i = 0;
                for i in range(5):
                    for j in range(5):
                        if j < i or (i==0 and j ==0):
                            continue
                        if j==i:
                            convimg = convolve(norm_gray,filtgrid[i,j],mode='reflect')
                            lawsimg = convolve(np.absolute(convimg),np.ones([15,15]),mode='reflect')
                            lawsFeat[band,count_i] = lawsimg[self.ms_mask_clip].mean()
                            lawsFeat[band,count_i+14] = lawsimg[self.ms_mask_clip].std()
                            count_i += 1
                        else:
                            convimg1 = np.absolute(convolve(norm_gray,filtgrid[i,j],mode='reflect'))
                            convimg2 = np.absolute(convolve(norm_gray,filtgrid[j,i],mode='reflect'))
                            lawsimg = convolve(convimg1+convimg2,np.ones([15,15])/2,mode='reflect')
                            lawsFeat[band,count_i] = lawsimg[self.ms_mask_clip].mean()
                            lawsFeat[band,count_i+14] = lawsimg[self.ms_mask_clip].std()
                            count_i += 1
            self.laws_ms_feats = np.vstack((lawsFeat,lawsFeat.mean(axis=0)))
        else:
            raise AttributeError('No Multispectral imagery provided')

    def createSpecIndices(self):
        '''
        Compute RGB and MS spectral indices
        '''
        GRVI_pixels = np.divide(self.rgb_pixels[:,1]-self.rgb_pixels[:,0],
                              self.rgb_pixels[:,1]+self.rgb_pixels[:,0]+1e-15)
        VARI_pixels = np.divide(self.rgb_pixels[:,1]-self.rgb_pixels[:,0],
                              self.rgb_pixels[:,1]+self.rgb_pixels[:,0]\
                              -self.rgb_pixels[:,2]+1e-15)
        GLIr_pixels = np.divide(2*self.rgb_pixels[:,0] - self.rgb_pixels[:,1]\
                                -self.rgb_pixels[:,2],
                                2*self.rgb_pixels[:,0]+self.rgb_pixels[:,1]\
                                +self.rgb_pixels[:,2]+1e-15)
        GLIg_pixels = np.divide(2*self.rgb_pixels[:,1] - self.rgb_pixels[:,0]\
                                -self.rgb_pixels[:,2],
                                2*self.rgb_pixels[:,1]+self.rgb_pixels[:,0]\
                                +self.rgb_pixels[:,2]+1e-15)
        GLIb_pixels = np.divide(2*self.rgb_pixels[:,2] - self.rgb_pixels[:,1]\
                                -self.rgb_pixels[:,0],
                                2*self.rgb_pixels[:,2]+self.rgb_pixels[:,1]\
                                +self.rgb_pixels[:,0]+1e-15)
        ExG_pixels = np.divide(2*self.rgb_pixels[:,1] - self.rgb_pixels[:,0]\
                                -self.rgb_pixels[:,2],
                                self.rgb_pixels[:,2]+self.rgb_pixels[:,1]\
                                +self.rgb_pixels[:,0]+1e-15)
        ExR_pixels = np.divide(2*self.rgb_pixels[:,0] - self.rgb_pixels[:,1]\
                                -self.rgb_pixels[:,2],
                                self.rgb_pixels[:,2]+self.rgb_pixels[:,1]\
                                +self.rgb_pixels[:,0]+1e-15)
        ExB_pixels = np.divide(2*self.rgb_pixels[:,2] - self.rgb_pixels[:,0]\
                                -self.rgb_pixels[:,1],
                                self.rgb_pixels[:,2]+self.rgb_pixels[:,1]\
                                +self.rgb_pixels[:,0]+1e-15)
        ExGveg_pixels = 2*self.rgb_pixels[:,1]- self.rgb_pixels[:,0]\
                                -self.rgb_pixels[:,2]+50
        NegExR_pixels = self.rgb_pixels[:,1]- 1.4*self.rgb_pixels[:,0]
        ExRveg_pixels = np.divide(1.4*self.rgb_pixels[:,1] -\
                                  self.rgb_pixels[:,0],
                                  self.rgb_pixels[:,2]+self.rgb_pixels[:,1]\
                                  +self.rgb_pixels[:,0]+1e-15)
        ExBveg_pixels = np.divide(1.4*self.rgb_pixels[:,2] -\
                                  self.rgb_pixels[:,0],
                                  self.rgb_pixels[:,2]+self.rgb_pixels[:,1]\
                                  +self.rgb_pixels[:,0]+1e-15)
        TGI_pixels = self.rgb_pixels[:,1] -0.39*self.rgb_pixels[:,0]\
                    -0.61*self.rgb_pixels[:,2]
        mGRVI_pixels = np.divide(self.rgb_pixels[:,1]*self.rgb_pixels[:,1] -\
                                 self.rgb_pixels[:,0]*self.rgb_pixels[:,0],
                                self.rgb_pixels[:,1]*self.rgb_pixels[:,1] +\
                                 self.rgb_pixels[:,0]*self.rgb_pixels[:,0]+\
                                 1e-15)
        RGBVI_pixels = np.divide(self.rgb_pixels[:,1]*self.rgb_pixels[:,1] -\
                                 self.rgb_pixels[:,0]*self.rgb_pixels[:,2],
                                self.rgb_pixels[:,1]*self.rgb_pixels[:,1] +\
                                 self.rgb_pixels[:,0]*self.rgb_pixels[:,2]+\
                                 1e-15)
        IKAW_pixels = np.divide(self.rgb_pixels[:,0]-self.rgb_pixels[:,2],
                              self.rgb_pixels[:,0]+self.rgb_pixels[:,2]+1e-15)
        self.rgb_indices = np.stack((
                    GRVI_pixels, VARI_pixels, GLIr_pixels, GLIg_pixels, 
                    GLIb_pixels, ExG_pixels, ExR_pixels, ExB_pixels,
                    ExGveg_pixels, NegExR_pixels, ExRveg_pixels, ExBveg_pixels,
                    TGI_pixels, mGRVI_pixels, RGBVI_pixels, IKAW_pixels
                ),axis=1)
        self.rgbindex_list = ('GRVI','VARI','GLIr','GLIg','GLIb','ExG','ExR',
                              'ExB','ExGveg','NegExR','ExRveg','ExBveg','TGI',
                              'mGRVI','RGBVI','IKAW')
        if hasattr(self,'ms_pixels'):
            NDVI_pixels = np.divide(self.ms_pixels[:,3]-self.ms_pixels[:,1],
                                  self.ms_pixels[:,3]+self.ms_pixels[:,1]+1e-15)
            NDVIg_pixels = np.divide(self.ms_pixels[:,3]-self.ms_pixels[:,0],
                                  self.ms_pixels[:,3]+self.ms_pixels[:,0]+1e-15)
            NDVIre_pixels = np.divide(self.ms_pixels[:,3]-self.ms_pixels[:,2],
                                  self.ms_pixels[:,3]+self.ms_pixels[:,2]+1e-15)
            CIG_pixels = np.divide(self.ms_pixels[:,3],self.ms_pixels[:,0]+1e-15)-1
            CVI_pixels = np.divide(
                        np.multiply(
                            np.multiply(self.ms_pixels[:,3],self.ms_pixels[:,1]),
                            self.ms_pixels[:,1]
                        ),                
                        self.ms_pixels[:,0]+1e-15
                    )
            GRVIms_pixels = np.divide(self.ms_pixels[:,0]-self.ms_pixels[:,1],
                                  self.ms_pixels[:,0]+self.ms_pixels[:,1]+1e-15)
            mGRVIms_pixels = np.divide(self.ms_pixels[:,0]*self.ms_pixels[:,0]-\
                                       self.ms_pixels[:,1]*self.ms_pixels[:,1],
                                       self.ms_pixels[:,0]*self.ms_pixels[:,0]+\
                                       self.ms_pixels[:,1]*self.ms_pixels[:,1]
                                       +1e-15)
            NegExRms_pixels = self.ms_pixels[:,0] - 1.4* self.ms_pixels[:,1]
            self.msindex_list = ('NDVI','NDVIg','NDVIre','CIG','CVI','GRVI',
                                 'mGRVI','NegExR')
            self.ms_indices = np.stack((
                    NDVI_pixels, NDVIg_pixels, NDVIre_pixels, CIG_pixels,
                    CVI_pixels, GRVIms_pixels, mGRVIms_pixels, NegExRms_pixels
                ),axis=1)

    
    def createRGBIndFeats(self,mode=False):
        '''
        Compute features, treating RGB indices as bands
        '''
        if not hasattr(self,'rgb_indices'):
            self.createSpecIndices()
        self.rgb_ind_max = self.rgb_indices.max(axis=0)
        self.rgb_ind_min = self.rgb_indices.min(axis=0)
        self.rgb_ind_mean = self.rgb_indices.mean(axis=0)
        self.rgb_ind_std = self.rgb_indices.std(axis=0)
        self.rgb_ind_median = np.median(self.rgb_indices,axis=0)
        self.rgb_ind_cov = np.divide(self.rgb_ind_std,self.rgb_ind_mean)
        self.rgb_ind_skew = sp.stats.skew(self.rgb_indices,axis=0)
        self.rgb_ind_kurt = sp.stats.kurtosis(self.rgb_indices,axis=0)
        self.rgb_ind_sum = self.rgb_indices.sum(axis=0)
        self.rgb_ind_rng = self.rgb_ind_max-self.rgb_ind_min
        self.rgb_ind_rngsig = np.divide(self.rgb_ind_rng,self.rgb_ind_std)
        self.rgb_ind_rngmean = np.divide(self.rgb_ind_rng,self.rgb_ind_mean)
        if(mode):
            self.rgb_ind_mode = sp.stats.mode(self.rgb_indices,axis=0)[0][0]
        self.rgb_ind_deciles = np.percentile(self.rgb_indices,
                                                np.linspace(10,90,9),axis=0)
        self.rgb_ind_quartiles = np.percentile(self.rgb_indices,[25,75],axis=0)
        self.rgb_ind_iqr = self.rgb_ind_quartiles[1,:]-self.rgb_ind_quartiles[0,:]
        self.rgb_ind_iqrsig = np.divide(self.rgb_ind_iqr,self.rgb_ind_std)
        self.rgb_ind_iqrmean = np.divide(self.rgb_ind_iqr,self.rgb_ind_mean)
    
    def createMSIndFeats(self,mode=False):
        '''
        Compute features, treating MS indices as bands
        '''
        if hasattr(self,'ms_pixels'):
            if not hasattr(self,'ms_indices'):
                self.createSpecIndices()
            self.ms_ind_max = self.ms_indices.max(axis=0)
            self.ms_ind_min = self.ms_indices.min(axis=0)
            self.ms_ind_mean = self.ms_indices.mean(axis=0)
            self.ms_ind_std = self.ms_indices.std(axis=0)
            self.ms_ind_median = np.median(self.ms_indices,axis=0)
            self.ms_ind_cov = np.divide(self.ms_ind_std,self.ms_ind_mean)
            self.ms_ind_skew = sp.stats.skew(self.ms_indices,axis=0)
            self.ms_ind_kurt = sp.stats.kurtosis(self.ms_indices,axis=0)
            self.ms_ind_sum = self.ms_indices.sum(axis=0)
            self.ms_ind_rng = self.ms_ind_max-self.ms_ind_min
            self.ms_ind_rngsig = np.divide(self.ms_ind_rng,self.ms_ind_std)
            self.ms_ind_rngmean = np.divide(self.ms_ind_rng,self.ms_ind_mean)
            if(mode):
                self.ms_ind_mode = sp.stats.mode(self.ms_indices,axis=0)[0][0]
            self.ms_ind_deciles = np.percentile(self.ms_indices,
                                                    np.linspace(10,90,9),axis=0)
            self.ms_ind_quartiles = np.percentile(self.ms_indices,[25,75],axis=0)
            self.ms_ind_iqr = self.ms_ind_quartiles[1,:]-self.ms_ind_quartiles[0,:]
            self.ms_ind_iqrsig = np.divide(self.ms_ind_iqr,self.ms_ind_std)
            self.ms_ind_iqrmean = np.divide(self.ms_ind_iqr,self.ms_ind_mean)
        else:
            raise AttributeError('No Multispectral imagery provided')
   
    def createDSMRawFeats(self,mode=False):
        '''
        Compute stats treating the DSM raster as a band
        only compute mode value if mode = True
        '''
        if hasattr(self,'dsm_pixels'):
            self.dsm_raw_max = np.array([self.dsm_pixels.max(axis=0)])
            self.dsm_raw_min = np.array([self.dsm_pixels.min(axis=0)])
            self.dsm_raw_mean = np.array([self.dsm_pixels.mean(axis=0)])
            self.dsm_raw_std = np.array([self.dsm_pixels.std(axis=0)])
            self.dsm_raw_median = np.array([np.median(self.dsm_pixels,axis=0)])
            self.dsm_raw_cov = np.divide(self.dsm_raw_std,self.dsm_raw_mean)
            self.dsm_raw_skew = np.array([sp.stats.skew(self.dsm_pixels,axis=0)])
            self.dsm_raw_kurt = np.array([sp.stats.kurtosis(self.dsm_pixels,axis=0)])
            self.dsm_raw_sum = np.array([self.dsm_pixels.sum(axis=0)])
            self.dsm_raw_rng = self.dsm_raw_max-self.dsm_raw_min
            self.dsm_raw_rngsig = np.divide(self.dsm_raw_rng,self.dsm_raw_std)
            self.dsm_raw_rngmean = np.divide(self.dsm_raw_rng,self.dsm_raw_mean)
            if(mode):
                self.dsm_raw_mode = sp.stats.mode(self.dsm_pixels,axis=0)[0][0]
            self.dsm_raw_deciles = np.percentile(
                                        self.dsm_pixels,np.linspace(10,90,9),axis=0)
            self.dsm_raw_quartiles = np.percentile(self.dsm_pixels,[25,75],axis=0)
            self.dsm_raw_iqr = np.array([self.dsm_raw_quartiles[1]-self.dsm_raw_quartiles[0]])
            self.dsm_raw_iqrsig = np.divide(self.dsm_raw_iqr,self.dsm_raw_std)
            self.dsm_raw_iqrmean = np.divide(self.dsm_raw_iqr,self.dsm_raw_mean)
            self.dsm_raw_mad = np.array([np.median(np.absolute(self.dsm_pixels - np.median(self.dsm_pixels)))])
            self.dsm_raw_maxmed = self.dsm_raw_max - self.dsm_raw_median
            self.dsm_raw_minmed = self.dsm_raw_min - self.dsm_raw_median
            self.dsm_raw_summed = np.array([(self.dsm_pixels-self.dsm_raw_median).sum(axis=0)])
            self.dsm_raw_decilesmed = self.dsm_raw_deciles - self.dsm_raw_median
            self.dsm_raw_quartilesmed = self.dsm_raw_quartiles - self.dsm_raw_median
        else:
            raise AttributeError('No DSM imagery provided')
    
    def __createDSMGLCMImg(self,levels=32):
        # clamp levels number of height bands, spread uniformly so that
        # 0 means below 5% percentile of H, levels-1 means above top 95%-ile
        # and all else are spread out linearly in this range
        # clamp minimum of 1 in region of interest to avoid issue of mostly zeroes
        if(levels>255):
            raise ValueError('max number of levels is 255')
        lims = np.percentile(self.dsm_pixels,[5,95],axis=0)
        scaleimg = rescale_intensity(self.dsm_img_clip,in_range = (lims[0],lims[1]))
        self.dsm_glcm_img = (scaleimg*(levels-1)).astype('uint8')
        local_img=np.zeros(self.dsm_glcm_img.shape,dtype='uint8')
        local_img[self.dsm_mask_clip[0],self.dsm_mask_clip[1]]=np.maximum(self.dsm_glcm_img[self.dsm_mask_clip[0],self.dsm_mask_clip[1]],np.ones((self.dsm_mask[0].__len__())))
        local_img = local_img[~np.all(local_img==0,axis=1),:]
        local_img = local_img[:,~np.all(local_img==0,axis=0)]
        self.dsm_glcm_img_masked = local_img
    
    def __padDSMGLCM(self,glcm_step): # Pad glcm#
        counter = 0
        while(counter < glcm_step and (self.dsm_glcm_img_masked.shape[0]<glcm_step+1 or self.dsm_glcm_img_masked.shape[1]<glcm_step+1)):
            frame = np.zeros((self.dsm_glcm_img_masked.shape[0]+2,self.dsm_glcm_img_masked.shape[1]+2),dtype=self.dsm_glcm_img_masked.dtype)
            frame[1:self.dsm_glcm_img_masked.shape[0]+1,1:self.dsm_glcm_img_masked.shape[1]+1]=self.dsm_glcm_img_masked
            self.dsm_glcm_img_masked = frame
            counter = counter + 1
        kern = np.array([[1,1,1],[1,0,1],[1,1,1]])
        tmpimg = self.dsm_glcm_img_masked
        convimg = convolve(self.dsm_glcm_img_masked.astype('float32'),kern,mode='reflect')
        maskimg = np.where(self.dsm_glcm_img_masked!=0,1.0,0.0)
        convmask = convolve(maskimg.astype('float32'),kern,mode='reflect')
        idx1 = np.nonzero(convimg)
        convimg[idx1] = np.maximum(convimg[idx1],np.ones((idx1[0].__len__())))/convmask[idx1]
        convimg = convimg.astype('uint8')
        idx2 = tmpimg==0
        tmpimg[idx2] = convimg[idx2]
        self.dsm_glcm_img_masked = tmpimg
    
    def createDSMGLCMfeats(self,distance=1):
        '''
        Compute features based on GLCM for DSM imagery with offset 
        distance pixels
        '''
        if hasattr(self,'dsm_img'):
            if not hasattr(self,'dsm_glcm_img_masked'):
                self.__createDSMGLCMImg()
            success = False
            counter = 0
            while(counter < distance and not success):
                try:
                    glcm_dsm_vals = mht.haralick(self.dsm_glcm_img_masked,ignore_zeros=True,
                                                return_mean_ptp=True,distance=distance)
                    success = True
                except:
                    self.__padDSMGLCM(distance)
                    counter = counter + 1
            if not hasattr(self,'glcm_dsm_vals'):
                self.glcm_dsm_vals = glcm_dsm_vals
            else:
                self.glcm_dsm_vals = np.concatenate((self.glcm_dsm_vals,
                                                        glcm_dsm_vals))
            if not hasattr(self,'glcm_dsm_dist'):
                self.glcm_dsm_dist = [distance]
            else:
                self.glcm_dsm_dist.append(distance)
        else:
            raise AttributeError('No DSM imagery provided')
    
    def createDSMautoCorFeats(self,distance=1):
        '''
        Compute features based on autocorrelation for DSM imagery with offset 
        distance pixels
        '''
        if hasattr(self,'dsm_img'):
            local_img = np.zeros(self.dsm_img_clip.shape)
            local_img[self.dsm_mask_clip[0],self.dsm_mask_clip[1]]=self.dsm_img_clip[self.dsm_mask_clip[0],self.dsm_mask_clip[1]]
            N = self.__imgAutocorrelate(local_img,0,distance)
            NE = self.__imgAutocorrelate(local_img,distance,distance)
            E = self.__imgAutocorrelate(local_img,distance,0)
            SE  = self.__imgAutocorrelate(local_img,distance,-distance)
            acors = np.array([N,NE,E,SE])
            acfeats = np.array([acors.mean(),acors.max()-acors.min()])
            if not hasattr(self,'acor_dsm_vals'):
                self.acor_dsm_vals = acfeats
            else:
                self.acor_dsm_vals = np.concatenate((self.acor_dsm_vals,
                                                        acfeats))
            if not hasattr(self,'acor_dsm_dist'):
                self.acor_dsm_dist = [distance]
            else:
                self.acor_dsm_dist.append(distance)
        else:
            raise AttributeError('No DSM imagery provided')
    
    def createDSMLBPFeats(self,distance=1):
        '''
        Compute features based on Local binary patterns for DSM imagery with 
        offset distance pixels
        '''
        if hasattr(self,'dsm_img'):
            if not distance in [1,2,3]:
                raise ValueError('distance can only be 1,2 or 3')
            if not hasattr(self,'dsm_glcm_img'):
                self.__createDSMGLCMImg()
            lbp_img = local_binary_pattern(self.dsm_glcm_img,8*distance,distance,method='uniform')
            lbp_pix = lbp_img[self.dsm_mask_clip]
            unique, counts = np.unique(lbp_pix, return_counts = True)
            count_table = np.zeros([2+distance*8])
            count_table[unique.astype('int')]=counts
            count_table = count_table/count_table.sum()
            if not hasattr(self,'lbp_dsm_vals'):
                self.lbp_dsm_vals = count_table
            else:
                self.lbp_dsm_vals = np.concatenate((self.lbp_dsm_vals,count_table))
            if not hasattr(self,'lbp_dsm_dist'):
                self.lbp_dsm_dist = [distance]
            else:
                self.lbp_dsm_dist.append(distance)
        else:
            raise AttributeError('No DSM imagery provided')
    
    def createDSMLawsFeats(self):
        '''
        Compute features based on Laws' texture energy for DSM imagery with 
        offset distance pixels
        '''
        if hasattr(self,'dsm_img'):
            mean_15 = convolve(self.dsm_img_clip,np.ones([15,15])/225,mode='reflect')
            norm_gray = self.dsm_img_clip-mean_15
            del mean_15
            # Constuct filter bank
            L5 = np.array([1,4,6,4,1])
            E5 = np.array([-1,-2,0,2,1])
            S5 = np.array([-1,0,2,0,-1])
            R5 = np.array([1,-4,6,-4,1])
            W5 = np.array([-1,2,0,-2,1])
            filtbank = [L5,E5,S5,R5,W5]
            del L5, E5, S5, R5, W5
            filtgrid = np.zeros([5,5,5,5])
            for i in range(5):
                for j in range(5):
                    filtgrid[i,j,:,:]=(np.outer(filtbank[i],filtbank[j]))
            del filtbank
            # compute features
            lawsFeat = np.zeros([14,2])
            count_i = 0;
            for i in range(5):
                for j in range(5):
                    if j < i or (i==0 and j ==0):
                        continue
                    if j==i:
                        convimg = convolve(norm_gray,filtgrid[i,j],mode='reflect')
                        lawsimg = convolve(np.absolute(convimg),np.ones([15,15]),mode='reflect')
                        lawsFeat[count_i,0] = lawsimg[self.dsm_mask_clip].mean()
                        lawsFeat[count_i,1] = lawsimg[self.dsm_mask_clip].std()
                        count_i += 1
                    else:
                        convimg1 = np.absolute(convolve(norm_gray,filtgrid[i,j],mode='reflect'))
                        convimg2 = np.absolute(convolve(norm_gray,filtgrid[j,i],mode='reflect'))
                        lawsimg = convolve(convimg1+convimg2,np.ones([15,15])/2,mode='reflect')
                        lawsFeat[count_i,0] = lawsimg[self.dsm_mask_clip].mean()
                        lawsFeat[count_i,1] = lawsimg[self.dsm_mask_clip].std()
                        count_i += 1
            self.laws_dsm_feats = lawsFeat
        else:
            raise AttributeError('No DSM imagery provided')
    
    def stackFeats(self):
        '''
        Stack all compured features to be accessed by the attribute featStack
        '''
        featStack = np.array([])
        featList = []
        featClass = []
        featSizeInvar = []
        featHeightInvar=[]
        featScale = []
        if hasattr(self,'rgb_band_max'):
            featList.extend(['rgb_band_max_R','rgb_band_max_G','rgb_band_max_B'])
            featClass.extend(['rgb_band']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.rgb_band_max
            else:
                featStack = np.concatenate((featStack,self.rgb_band_max))
        if hasattr(self,'rgb_band_min'):
            featList.extend(['rgb_band_min_R','rgb_band_min_G','rgb_band_min_B'])
            featClass.extend(['rgb_band']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.rgb_band_min
            else:
                featStack = np.concatenate((featStack,self.rgb_band_min))
        if hasattr(self,'rgb_band_mean'):
            featList.extend(['rgb_band_mean_R','rgb_band_mean_G','rgb_band_mean_B'])
            featClass.extend(['rgb_band']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.rgb_band_mean
            else:
                featStack = np.concatenate((featStack,self.rgb_band_mean))
        if hasattr(self,'rgb_band_std'):
            featList.extend(['rgb_band_std_R','rgb_band_std_G','rgb_band_std_B'])
            featClass.extend(['rgb_band']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.rgb_band_std
            else:
                featStack = np.concatenate((featStack,self.rgb_band_std))
        if hasattr(self,'rgb_band_median'):
            featList.extend(['rgb_band_median_R','rgb_band_median_G','rgb_band_median_B'])
            featClass.extend(['rgb_band']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.rgb_band_median
            else:
                featStack = np.concatenate((featStack,self.rgb_band_median))
        if hasattr(self,'rgb_band_cov'):
            featList.extend(['rgb_band_cov_R','rgb_band_cov_G','rgb_band_cov_B'])
            featClass.extend(['rgb_band']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.rgb_band_cov
            else:
                featStack = np.concatenate((featStack,self.rgb_band_cov))
        if hasattr(self,'rgb_band_skew'):
            featList.extend(['rgb_band_skew_R','rgb_band_skew_G','rgb_band_skew_B'])
            featClass.extend(['rgb_band']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.rgb_band_skew
            else:
                featStack = np.concatenate((featStack,self.rgb_band_skew))
        if hasattr(self,'rgb_band_kurt'):
            featList.extend(['rgb_band_kurt_R','rgb_band_kurt_G','rgb_band_kurt_B'])
            featClass.extend(['rgb_band']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.rgb_band_kurt
            else:
                featStack = np.concatenate((featStack,self.rgb_band_kurt))
        if hasattr(self,'rgb_band_sum'):
            featList.extend(['rgb_band_sum_R','rgb_band_sum_G','rgb_band_sum_B'])
            featClass.extend(['rgb_band']*3)
            featSizeInvar.extend([False]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.rgb_band_sum
            else:
                featStack = np.concatenate((featStack,self.rgb_band_sum))
        if hasattr(self,'rgb_band_rng'):
            featList.extend(['rgb_band_rng_R','rgb_band_rng_G','rgb_band_rng_B'])
            featClass.extend(['rgb_band']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.band_rng
            else:
                featStack = np.concatenate((featStack,self.rgb_band_rng))
        if hasattr(self,'rgb_band_rngsig'):
            featList.extend(['rgb_band_rngsig_R','rgb_band_rngsig_G','rgb_band_rngsig_B'])
            featClass.extend(['rgb_band']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.rgb_band_rngsig
            else:
                featStack = np.concatenate((featStack,self.rgb_band_rngsig))
        if hasattr(self,'rgb_band_rngmean'):
            featList.extend(['rgb_band_rngmean_R','rgb_band_rngmean_G','rgb_band_rngmean_B'])
            featClass.extend(['rgb_band']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.rgb_band_rngmean
            else:
                featStack = np.concatenate((featStack,self.rgb_band_rngmean))
        if hasattr(self,'rgb_band_mode'):
            featList.extend(['rgb_band_mode_R','rgb_band_mode_G','rgb_band_mode_B'])
            featClass.extend(['rgb_band']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.rgb_band_mode
            else:
                featStack = np.concatenate((featStack,self.rgb_band_mode))
        if hasattr(self,'rgb_band_deciles'):
            featList.extend(['rgb_band_decile_R_1','rgb_band_decile_R_2',
                            'rgb_band_decile_R_3','rgb_band_decile_R_4',
                            'rgb_band_decile_R_5','rgb_band_decile_R_6',
                            'rgb_band_decile_R_7','rgb_band_decile_R_8',
                            'rgb_band_decile_R_9','rgb_band_decile_G_1',
                            'rgb_band_decile_G_2','rgb_band_decile_G_3',
                            'rgb_band_decile_G_4','rgb_band_decile_G_5',
                            'rgb_band_decile_G_6','rgb_band_decile_G_7',
                            'rgb_band_decile_G_8','rgb_band_decile_G_9',
                            'rgb_band_decile_B_1','rgb_band_decile_B_2',
                            'rgb_band_decile_B_3','rgb_band_decile_B_4',
                            'rgb_band_decile_B_5','rgb_band_decile_B_6',
                            'rgb_band_decile_B_7','rgb_band_decile_B_8',
                            'rgb_band_decile_B_9'])
            featClass.extend(['rgb_band']*27)
            featSizeInvar.extend([True]*27)
            featHeightInvar.extend([True]*27)
            featScale.extend([0]*27)
            if featStack.size==0:
                featStack = self.rgb_band_deciles.flatten('F')
            else:
                featStack = np.concatenate((featStack,
                                            self.rgb_band_deciles.flatten('F')))
        if hasattr(self,'rgb_band_quartiles'):
            featList.extend(['rgb_band_quartile_R_1','rgb_band_quartile_R_3',
                            'rgb_band_quartile_G_1','rgb_band_quartile_G_3',
                            'rgb_band_quartile_B_1','rgb_band_quartile_B_3'])
            featClass.extend(['rgb_band']*6)
            featSizeInvar.extend([True]*6)
            featHeightInvar.extend([True]*6)
            featScale.extend([0]*6)
            if featStack.size==0:
                featStack = self.rgb_band_quartiles.flatten('F')
            else:
                featStack = np.concatenate((featStack,
                                            self.rgb_band_quartiles.flatten('F')))
        if hasattr(self,'rgb_band_iqr'):
            featList.extend(['rgb_band_iqr_R','rgb_band_iqr_G','rgb_band_iqr_B'])
            featClass.extend(['rgb_band']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.rgb_band_iqr
            else:
                featStack = np.concatenate((featStack,self.rgb_band_iqr))
        if hasattr(self,'rgb_band_iqrsig'):
            featList.extend(['rgb_band_iqrsig_R','rgb_band_iqrsig_G','rgb_band_iqrsig_B'])
            featClass.extend(['rgb_band']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.rgb_band_iqrsig
            else:
                featStack = np.concatenate((featStack,self.rgb_band_iqrsig))
        if hasattr(self,'rgb_band_iqrmean'):
            featList.extend(['rgb_band_iqrmean_R','rgb_band_iqrmean_G','rgb_band_iqrmean_B'])
            featClass.extend(['rgb_band']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.rgb_band_iqrmean
            else:
                featStack = np.concatenate((featStack,self.rgb_band_iqrmean))
        if hasattr(self,'rgb_band_ratio'):
            featList.extend(['rgb_band_ratio_R','rgb_band_ratio_G','rgb_band_ratio_B'])
            featClass.extend(['rgb_band']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.rgb_band_ratio
            else:
                featStack = np.concatenate((featStack,self.rgb_band_ratio))
        if hasattr(self,'top_rgb_max'):
            featList.extend(['top_rgb_max_R','top_rgb_max_G','top_rgb_max_B'])
            featClass.extend(['rgb_top']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.top_rgb_max
            else:
                featStack = np.concatenate((featStack,self.top_rgb_max))
        if hasattr(self,'top_rgb_min'):
            featList.extend(['top_rgb_min_R','top_rgb_min_G','top_rgb_min_B'])
            featClass.extend(['rgb_top']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.top_rgb_min
            else:
                featStack = np.concatenate((featStack,self.top_rgb_min))
        if hasattr(self,'top_rgb_mean'):
            featList.extend(['top_rgb_mean_R','top_rgb_mean_G','top_rgb_mean_B'])
            featClass.extend(['rgb_top']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.top_rgb_mean
            else:
                featStack = np.concatenate((featStack,self.top_rgb_mean))
        if hasattr(self,'top_rgb_std'):
            featList.extend(['top_rgb_std_R','top_rgb_std_G','top_rgb_std_B'])
            featClass.extend(['rgb_top']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.top_rgb_std
            else:
                featStack = np.concatenate((featStack,self.top_rgb_std))
        if hasattr(self,'top_rgb_median'):
            featList.extend(['top_rgb_median_R','top_rgb_median_G','top_rgb_median_B'])
            featClass.extend(['rgb_top']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.top_rgb_median
            else:
                featStack = np.concatenate((featStack,self.top_rgb_median))
        if hasattr(self,'top_rgb_cov'):
            featList.extend(['top_rgb_cov_R','top_rgb_cov_G','top_rgb_cov_B'])
            featClass.extend(['rgb_top']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.top_rgb_cov
            else:
                featStack = np.concatenate((featStack,self.top_rgb_cov))
        if hasattr(self,'top_rgb_skew'):
            featList.extend(['top_rgb_skew_R','top_rgb_skew_G','top_rgb_skew_B'])
            featClass.extend(['rgb_top']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.top_rgb_skew
            else:
                featStack = np.concatenate((featStack,self.top_rgb_skew))
        if hasattr(self,'top_rgb_kurt'):
            featList.extend(['top_rgb_kurt_R','top_rgb_kurt_G','top_rgb_kurt_B'])
            featClass.extend(['rgb_top']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.top_rgb_kurt
            else:
                featStack = np.concatenate((featStack,self.top_rgb_kurt))
        if hasattr(self,'top_rgb_sum'):
            featList.extend(['top_rgb_sum_R','top_rgb_sum_G','top_rgb_sum_B'])
            featClass.extend(['rgb_top']*3)
            featSizeInvar.extend([False]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.top_rgb_sum
            else:
                featStack = np.concatenate((featStack,self.top_rgb_sum))
        if hasattr(self,'top_rgb_rng'):
            featList.extend(['top_rgb_rng_R','top_rgb_rng_G','top_rgb_rng_B'])
            featClass.extend(['rgb_top']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.band_rng
            else:
                featStack = np.concatenate((featStack,self.top_rgb_rng))
        if hasattr(self,'top_rgb_rngsig'):
            featList.extend(['top_rgb_rngsig_R','top_rgb_rngsig_G','top_rgb_rngsig_B'])
            featClass.extend(['rgb_top']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.top_rgb_rngsig
            else:
                featStack = np.concatenate((featStack,self.top_rgb_rngsig))
        if hasattr(self,'top_rgb_rngmean'):
            featList.extend(['top_rgb_rngmean_R','top_rgb_rngmean_G','top_rgb_rngmean_B'])
            featClass.extend(['rgb_top']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.top_rgb_rngmean
            else:
                featStack = np.concatenate((featStack,self.top_rgb_rngmean))
        if hasattr(self,'top_rgb_mode'):
            featList.extend(['top_rgb_mode_R','top_rgb_mode_G','top_rgb_mode_B'])
            featClass.extend(['rgb_top']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.top_rgb_mode
            else:
                featStack = np.concatenate((featStack,self.top_rgb_mode))
        if hasattr(self,'top_rgb_deciles'):
            featList.extend(['top_rgb_decile_R_1','top_rgb_decile_R_2',
                            'top_rgb_decile_R_3','top_rgb_decile_R_4',
                            'top_rgb_decile_R_5','top_rgb_decile_R_6',
                            'top_rgb_decile_R_7','top_rgb_decile_R_8',
                            'top_rgb_decile_R_9','top_rgb_decile_G_1',
                            'top_rgb_decile_G_2','top_rgb_decile_G_3',
                            'top_rgb_decile_G_4','top_rgb_decile_G_5',
                            'top_rgb_decile_G_6','top_rgb_decile_G_7',
                            'top_rgb_decile_G_8','top_rgb_decile_G_9',
                            'top_rgb_decile_B_1','top_rgb_decile_B_2',
                            'top_rgb_decile_B_3','top_rgb_decile_B_4',
                            'top_rgb_decile_B_5','top_rgb_decile_B_6',
                            'top_rgb_decile_B_7','top_rgb_decile_B_8',
                            'top_rgb_decile_B_9'])
            featClass.extend(['rgb_top']*27)
            featSizeInvar.extend([True]*27)
            featHeightInvar.extend([True]*27)
            featScale.extend([0]*27)
            if featStack.size==0:
                featStack = self.top_rgb_deciles.flatten('F')
            else:
                featStack = np.concatenate((featStack,
                                            self.top_rgb_deciles.flatten('F')))
        if hasattr(self,'top_rgb_quartiles'):
            featList.extend(['top_rgb_quartile_R_1','top_rgb_quartile_R_3',
                            'top_rgb_quartile_G_1','top_rgb_quartile_G_3',
                            'top_rgb_quartile_B_1','top_rgb_quartile_B_3'])
            featClass.extend(['rgb_top']*6)
            featSizeInvar.extend([True]*6)
            featHeightInvar.extend([True]*6)
            featScale.extend([0]*6)
            if featStack.size==0:
                featStack = self.top_rgb_quartiles.flatten('F')
            else:
                featStack = np.concatenate((featStack,
                                            self.top_rgb_quartiles.flatten('F')))
        if hasattr(self,'top_rgb_iqr'):
            featList.extend(['top_rgb_iqr_R','top_rgb_iqr_G','top_rgb_iqr_B'])
            featClass.extend(['rgb_top']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.top_rgb_iqr
            else:
                featStack = np.concatenate((featStack,self.top_rgb_iqr))
        if hasattr(self,'top_rgb_iqrsig'):
            featList.extend(['top_rgb_iqrsig_R','top_rgb_iqrsig_G','top_rgb_iqrsig_B'])
            featClass.extend(['rgb_top']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.top_rgb_iqrsig
            else:
                featStack = np.concatenate((featStack,self.top_rgb_iqrsig))
        if hasattr(self,'top_rgb_iqrmean'):
            featList.extend(['top_rgb_iqrmean_R','top_rgb_iqrmean_G','top_rgb_iqrmean_B'])
            featClass.extend(['rgb_top']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.top_rgb_iqrmean
            else:
                featStack = np.concatenate((featStack,self.top_rgb_iqrmean))
        if hasattr(self,'top_rgb_ratio'):
            featList.extend(['top_rgb_ratio_R','top_rgb_ratio_G','top_rgb_ratio_B'])
            featClass.extend(['rgb_top']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.top_rgb_ratio
            else:
                featStack = np.concatenate((featStack,self.top_rgb_ratio))
        if hasattr(self,'rgb_ind_max'):
            featList.extend(['rgb_ind_max_GRVI','rgb_ind_max_VARI',
                            'rgb_ind_max_GLIr','rgb_ind_max_GLIg',
                            'rgb_ind_max_GLIb','rgb_ind_max_ExG',
                            'rgb_ind_max_ExR','rgb_ind_max_ExB',
                            'rgb_ind_max_ExGveg','rgb_ind_max_NegExR',
                            'rgb_ind_max_ExRveg','rgb_ind_max_ExBveg',
                            'rgb_ind_max_TGI','rgb_ind_max_mGRVI',
                            'rgb_ind_max_RGBVI','rgb_ind_max_IKAW'])
            featClass.extend(['rgb_ind']*16)
            featSizeInvar.extend([True]*16)
            featHeightInvar.extend([True]*16)
            featScale.extend([0]*16)
            if featStack.size==0:
                featStack = self.rgb_ind_max
            else:
                featStack = np.concatenate((featStack,self.rgb_ind_max))
        if hasattr(self,'rgb_ind_min'):
            featList.extend(['rgb_ind_min_GRVI','rgb_ind_min_VARI',
                            'rgb_ind_min_GLIr','rgb_ind_min_GLIg',
                            'rgb_ind_min_GLIb','rgb_ind_min_ExG',
                            'rgb_ind_min_ExR','rgb_ind_min_ExB',
                            'rgb_ind_min_ExGveg','rgb_ind_min_NegExR',
                            'rgb_ind_min_ExRveg','rgb_ind_min_ExBveg',
                            'rgb_ind_min_TGI','rgb_ind_min_mGRVI',
                            'rgb_ind_min_RGBVI','rgb_ind_min_IKAW'])
            featClass.extend(['rgb_ind']*16)
            featSizeInvar.extend([True]*16)
            featHeightInvar.extend([True]*16)
            featScale.extend([0]*16)
            if featStack.size==0:
                featStack = self.rgb_ind_min
            else:
                featStack = np.concatenate((featStack,self.rgb_ind_min))
        if hasattr(self,'rgb_ind_mean'):
            featList.extend(['rgb_ind_mean_GRVI','rgb_ind_mean_VARI',
                            'rgb_ind_mean_GLIr','rgb_ind_mean_GLIg',
                            'rgb_ind_mean_GLIb','rgb_ind_mean_ExG',
                            'rgb_ind_mean_ExR','rgb_ind_mean_ExB',
                            'rgb_ind_mean_ExGveg','rgb_ind_mean_NegExR',
                            'rgb_ind_mean_ExRveg','rgb_ind_mean_ExBveg',
                            'rgb_ind_mean_TGI','rgb_ind_mean_mGRVI',
                            'rgb_ind_mean_RGBVI','rgb_ind_mean_IKAW'])
            featClass.extend(['rgb_ind']*16)
            featSizeInvar.extend([True]*16)
            featHeightInvar.extend([True]*16)
            featScale.extend([0]*16)
            if featStack.size==0:
                featStack = self.rgb_ind_mean
            else:
                featStack = np.concatenate((featStack,self.rgb_ind_mean))
        if hasattr(self,'rgb_ind_std'):
            featList.extend(['rgb_ind_std_GRVI','rgb_ind_std_VARI',
                            'rgb_ind_std_GLIr','rgb_ind_std_GLIg',
                            'rgb_ind_std_GLIb','rgb_ind_std_ExG',
                            'rgb_ind_std_ExR','rgb_ind_std_ExB',
                            'rgb_ind_std_ExGveg','rgb_ind_std_NegExR',
                            'rgb_ind_std_ExRveg','rgb_ind_std_ExBveg',
                            'rgb_ind_std_TGI','rgb_ind_std_mGRVI',
                            'rgb_ind_std_RGBVI','rgb_ind_std_IKAW'])
            featClass.extend(['rgb_ind']*16)
            featSizeInvar.extend([True]*16)
            featHeightInvar.extend([True]*16)
            featScale.extend([0]*16)
            if featStack.size==0:
                featStack = self.rgb_ind_std
            else:
                featStack = np.concatenate((featStack,self.rgb_ind_std))
        if hasattr(self,'rgb_ind_median'):
            featList.extend(['rgb_ind_median_GRVI','rgb_ind_median_VARI',
                            'rgb_ind_median_GLIr','rgb_ind_median_GLIg',
                            'rgb_ind_median_GLIb','rgb_ind_median_ExG',
                            'rgb_ind_median_ExR','rgb_ind_median_ExB',
                            'rgb_ind_median_ExGveg','rgb_ind_median_NegExR',
                            'rgb_ind_median_ExRveg','rgb_ind_median_ExBveg',
                            'rgb_ind_median_TGI','rgb_ind_median_mGRVI',
                            'rgb_ind_median_RGBVI','rgb_ind_median_IKAW'])
            featClass.extend(['rgb_ind']*16)
            featSizeInvar.extend([True]*16)
            featHeightInvar.extend([True]*16)
            featScale.extend([0]*16)
            if featStack.size==0:
                featStack = self.rgb_ind_median
            else:
                featStack = np.concatenate((featStack,self.rgb_ind_median))
        if hasattr(self,'rgb_ind_cov'):
            featList.extend(['rgb_ind_cov_GRVI','rgb_ind_cov_VARI',
                            'rgb_ind_cov_GLIr','rgb_ind_cov_GLIg',
                            'rgb_ind_cov_GLIb','rgb_ind_cov_ExG',
                            'rgb_ind_cov_ExR','rgb_ind_cov_ExB',
                            'rgb_ind_cov_ExGveg','rgb_ind_cov_NegExR',
                            'rgb_ind_cov_ExRveg','rgb_ind_cov_ExBveg',
                            'rgb_ind_cov_TGI','rgb_ind_cov_mGRVI',
                            'rgb_ind_cov_RGBVI','rgb_ind_cov_IKAW'])
            featClass.extend(['rgb_ind']*16)
            featSizeInvar.extend([True]*16)
            featHeightInvar.extend([True]*16)
            featScale.extend([0]*16)
            if featStack.size==0:
                featStack = self.rgb_ind_cov
            else:
                featStack = np.concatenate((featStack,self.rgb_ind_cov))
        if hasattr(self,'rgb_ind_skew'):
            featList.extend(['rgb_ind_skew_GRVI','rgb_ind_skew_VARI',
                            'rgb_ind_skew_GLIr','rgb_ind_skew_GLIg',
                            'rgb_ind_skew_GLIb','rgb_ind_skew_ExG',
                            'rgb_ind_skew_ExR','rgb_ind_skew_ExB',
                            'rgb_ind_skew_ExGveg','rgb_ind_skew_NegExR',
                            'rgb_ind_skew_ExRveg','rgb_ind_skew_ExBveg',
                            'rgb_ind_skew_TGI','rgb_ind_skew_mGRVI',
                            'rgb_ind_skew_RGBVI','rgb_ind_skew_IKAW'])
            featClass.extend(['rgb_ind']*16)
            featSizeInvar.extend([True]*16)
            featHeightInvar.extend([True]*16)
            featScale.extend([0]*16)
            if featStack.size==0:
                featStack = self.rgb_ind_skew
            else:
                featStack = np.concatenate((featStack,self.rgb_ind_skew))
        if hasattr(self,'rgb_ind_kurt'):
            featList.extend(['rgb_ind_kurt_GRVI','rgb_ind_kurt_VARI',
                            'rgb_ind_kurt_GLIr','rgb_ind_kurt_GLIg',
                            'rgb_ind_kurt_GLIb','rgb_ind_kurt_ExG',
                            'rgb_ind_kurt_ExR','rgb_ind_kurt_ExB',
                            'rgb_ind_kurt_ExGveg','rgb_ind_kurt_NegExR',
                            'rgb_ind_kurt_ExRveg','rgb_ind_kurt_ExBveg',
                            'rgb_ind_kurt_TGI','rgb_ind_kurt_mGRVI',
                            'rgb_ind_kurt_RGBVI','rgb_ind_kurt_IKAW'])
            featClass.extend(['rgb_ind']*16)
            featSizeInvar.extend([True]*16)
            featHeightInvar.extend([True]*16)
            featScale.extend([0]*16)
            if featStack.size==0:
                featStack = self.rgb_ind_kurt
            else:
                featStack = np.concatenate((featStack,self.rgb_ind_kurt))
        if hasattr(self,'rgb_ind_sum'):
            featList.extend(['rgb_ind_sum_GRVI','rgb_ind_sum_VARI',
                            'rgb_ind_sum_GLIr','rgb_ind_sum_GLIg',
                            'rgb_ind_sum_GLIb','rgb_ind_sum_ExG',
                            'rgb_ind_sum_ExR','rgb_ind_sum_ExB',
                            'rgb_ind_sum_ExGveg','rgb_ind_sum_NegExR',
                            'rgb_ind_sum_ExRveg','rgb_ind_sum_ExBveg',
                            'rgb_ind_sum_TGI','rgb_ind_sum_mGRVI',
                            'rgb_ind_sum_RGBVI','rgb_ind_sum_IKAW'])
            featClass.extend(['rgb_ind']*16)
            featSizeInvar.extend([False]*16)
            featHeightInvar.extend([True]*16)
            featScale.extend([0]*16)
            if featStack.size==0:
                featStack = self.rgb_ind_sum
            else:
                featStack = np.concatenate((featStack,self.rgb_ind_sum))
        if hasattr(self,'rgb_ind_rng'):
            featList.extend(['rgb_ind_rng_GRVI','rgb_ind_rng_VARI',
                            'rgb_ind_rng_GLIr','rgb_ind_rng_GLIg',
                            'rgb_ind_rng_GLIb','rgb_ind_rng_ExG',
                            'rgb_ind_rng_ExR','rgb_ind_rng_ExB',
                            'rgb_ind_rng_ExGveg','rgb_ind_rng_NegExR',
                            'rgb_ind_rng_ExRveg','rgb_ind_rng_ExBveg',
                            'rgb_ind_rng_TGI','rgb_ind_rng_mGRVI',
                            'rgb_ind_rng_RGBVI','rgb_ind_rng_IKAW'])
            featClass.extend(['rgb_ind']*16)
            featSizeInvar.extend([True]*16)
            featHeightInvar.extend([True]*16)
            featScale.extend([0]*16)
            if featStack.size==0:
                featStack = self.rgb_ind_rng
            else:
                featStack = np.concatenate((featStack,self.rgb_ind_rng))
        if hasattr(self,'rgb_ind_rngsig'):
            featList.extend(['rgb_ind_rngsig_GRVI','rgb_ind_rngsig_VARI',
                            'rgb_ind_rngsig_GLIr','rgb_ind_rngsig_GLIg',
                            'rgb_ind_rngsig_GLIb','rgb_ind_rngsig_ExG',
                            'rgb_ind_rngsig_ExR','rgb_ind_rngsig_ExB',
                            'rgb_ind_rngsig_ExGveg','rgb_ind_rngsig_NegExR',
                            'rgb_ind_rngsig_ExRveg','rgb_ind_rngsig_ExBveg',
                            'rgb_ind_rngsig_TGI','rgb_ind_rngsig_mGRVI',
                            'rgb_ind_rngsig_RGBVI','rgb_ind_rngsig_IKAW'])
            featClass.extend(['rgb_ind']*16)
            featSizeInvar.extend([True]*16)
            featHeightInvar.extend([True]*16)
            featScale.extend([0]*16)
            if featStack.size==0:
                featStack = self.rgb_ind_rngsig
            else:
                featStack = np.concatenate((featStack,self.rgb_ind_rngsig))
        if hasattr(self,'rgb_ind_rngmean'):
            featList.extend(['rgb_ind_rngmean_GRVI','rgb_ind_rngmean_VARI',
                            'rgb_ind_rngmean_GLIr','rgb_ind_rngmean_GLIg',
                            'rgb_ind_rngmean_GLIb','rgb_ind_rngmean_ExG',
                            'rgb_ind_rngmean_ExR','rgb_ind_rngmean_ExB',
                            'rgb_ind_rngmean_ExGveg','rgb_ind_rngmean_NegExR',
                            'rgb_ind_rngmean_ExRveg','rgb_ind_rngmean_ExBveg',
                            'rgb_ind_rngmean_TGI','rgb_ind_rngmean_mGRVI',
                            'rgb_ind_rngmean_RGBVI','rgb_ind_rngmean_IKAW'])
            featClass.extend(['rgb_ind']*16)
            featSizeInvar.extend([True]*16)
            featHeightInvar.extend([True]*16)
            featScale.extend([0]*16)
            if featStack.size==0:
                featStack = self.rgb_ind_rngmean
            else:
                featStack = np.concatenate((featStack,self.rgb_ind_rngmean))
        if hasattr(self,'rgb_ind_mode'):
            featList.extend(['rgb_ind_mode_GRVI','rgb_ind_mode_VARI',
                            'rgb_ind_mode_GLIr','rgb_ind_mode_GLIg',
                            'rgb_ind_mode_GLIb','rgb_ind_mode_ExG',
                            'rgb_ind_mode_ExR','rgb_ind_mode_ExB',
                            'rgb_ind_mode_ExGveg','rgb_ind_mode_NegExR',
                            'rgb_ind_mode_ExRveg','rgb_ind_mode_ExBveg',
                            'rgb_ind_mode_TGI','rgb_ind_mode_mGRVI',
                            'rgb_ind_mode_RGBVI','rgb_ind_mode_IKAW'])
            featClass.extend(['rgb_ind']*16)
            featSizeInvar.extend([True]*16)
            featHeightInvar.extend([True]*16)
            featScale.extend([0]*16)
            if featStack.size==0:
                featStack = self.rgb_ind_mode
            else:
                featStack = np.concatenate((featStack,self.rgb_ind_mode))
        if hasattr(self,'rgb_ind_deciles'):
            featList.extend(['rgb_ind_decile_GRVI_1','rgb_ind_decile_GRVI_2',
                            'rgb_ind_decile_GRVI_3','rgb_ind_decile_GRVI_4',
                            'rgb_ind_decile_GRVI_5','rgb_ind_decile_GRVI_6',
                            'rgb_ind_decile_GRVI_7','rgb_ind_decile_GRVI_8',
                            'rgb_ind_decile_GRVI_9','rgb_ind_decile_VARI_1',
                            'rgb_ind_decile_VARI_2','rgb_ind_decile_VARI_3',
                            'rgb_ind_decile_VARI_4','rgb_ind_decile_VARI_5',
                            'rgb_ind_decile_VARI_6','rgb_ind_decile_VARI_7',
                            'rgb_ind_decile_VARI_8','rgb_ind_decile_VARI_9',
                            'rgb_ind_decile_GLIr_1','rgb_ind_decile_GLIr_2',
                            'rgb_ind_decile_GLIr_3','rgb_ind_decile_GLIr_4',
                            'rgb_ind_decile_GLIr_5','rgb_ind_decile_GLIr_6',
                            'rgb_ind_decile_GLIr_7','rgb_ind_decile_GLIr_8',
                            'rgb_ind_decile_GLIr_9','rgb_ind_decile_GLIg_1',
                            'rgb_ind_decile_GLIg_2','rgb_ind_decile_GLIg_3',
                            'rgb_ind_decile_GLIg_4','rgb_ind_decile_GLIg_5',
                            'rgb_ind_decile_GLIg_6','rgb_ind_decile_GLIg_7',
                            'rgb_ind_decile_GLIg_8','rgb_ind_decile_GLIg_9',
                            'rgb_ind_decile_GLIb_1','rgb_ind_decile_GLIb_2',
                            'rgb_ind_decile_GLIb_3','rgb_ind_decile_GLIb_4',
                            'rgb_ind_decile_GLIb_5','rgb_ind_decile_GLIb_6',
                            'rgb_ind_decile_GLIb_7','rgb_ind_decile_GLIb_8',
                            'rgb_ind_decile_GLIb_9','rgb_ind_decile_ExG_1',
                            'rgb_ind_decile_ExG_2','rgb_ind_decile_ExG_3',
                            'rgb_ind_decile_ExG_4','rgb_ind_decile_ExG_5',
                            'rgb_ind_decile_ExG_6','rgb_ind_decile_ExG_7',
                            'rgb_ind_decile_ExG_8','rgb_ind_decile_ExG_9',
                            'rgb_ind_decile_ExR_1','rgb_ind_decile_ExR_2',
                            'rgb_ind_decile_ExR_3','rgb_ind_decile_ExR_4',
                            'rgb_ind_decile_ExR_5','rgb_ind_decile_ExR_6',
                            'rgb_ind_decile_ExR_7','rgb_ind_decile_ExR_8',
                            'rgb_ind_decile_ExR_9','rgb_ind_decile_ExB_1',
                            'rgb_ind_decile_ExB_2','rgb_ind_decile_ExB_3',
                            'rgb_ind_decile_ExB_4','rgb_ind_decile_ExB_5',
                            'rgb_ind_decile_ExB_6','rgb_ind_decile_ExB_7',
                            'rgb_ind_decile_ExB_8','rgb_ind_decile_ExB_9',
                            'rgb_ind_decile_ExGveg_1','rgb_ind_decile_ExGveg_2',
                            'rgb_ind_decile_ExGveg_3','rgb_ind_decile_ExGveg_4',
                            'rgb_ind_decile_ExGveg_5','rgb_ind_decile_ExGveg_6',
                            'rgb_ind_decile_ExGveg_7','rgb_ind_decile_ExGveg_8',
                            'rgb_ind_decile_ExGveg_9','rgb_ind_decile_NegExR_1',
                            'rgb_ind_decile_NegExR_2','rgb_ind_decile_NegExR_3',
                            'rgb_ind_decile_NegExR_4','rgb_ind_decile_NegExR_5',
                            'rgb_ind_decile_NegExR_6','rgb_ind_decile_NegExR_7',
                            'rgb_ind_decile_NegExR_8','rgb_ind_decile_NegExR_9',
                            'rgb_ind_decile_ExRveg_1','rgb_ind_decile_ExRveg_2',
                            'rgb_ind_decile_ExRveg_3','rgb_ind_decile_ExRveg_4',
                            'rgb_ind_decile_ExRveg_5','rgb_ind_decile_ExRveg_6',
                            'rgb_ind_decile_ExRveg_7','rgb_ind_decile_ExRveg_8',
                            'rgb_ind_decile_ExRveg_9','rgb_ind_decile_ExBveg_1',
                            'rgb_ind_decile_ExBveg_2','rgb_ind_decile_ExBveg_3',
                            'rgb_ind_decile_ExBveg_4','rgb_ind_decile_ExBveg_5',
                            'rgb_ind_decile_ExBveg_6','rgb_ind_decile_ExBveg_7',
                            'rgb_ind_decile_ExBveg_8','rgb_ind_decile_ExBveg_9',
                            'rgb_ind_decile_TGI_1','rgb_ind_decile_TGI_2',
                            'rgb_ind_decile_TGI_3','rgb_ind_decile_TGI_4',
                            'rgb_ind_decile_TGI_5','rgb_ind_decile_TGI_6',
                            'rgb_ind_decile_TGI_7','rgb_ind_decile_TGI_8',
                            'rgb_ind_decile_TGI_9','rgb_ind_decile_mGRVI_1',
                            'rgb_ind_decile_mGRVI_2','rgb_ind_decile_mGRVI_3',
                            'rgb_ind_decile_mGRVI_4','rgb_ind_decile_mGRVI_5',
                            'rgb_ind_decile_mGRVI_6','rgb_ind_decile_mGRVI_7',
                            'rgb_ind_decile_mGRVI_8','rgb_ind_decile_mGRVI_9',
                            'rgb_ind_decile_RGBVI_1','rgb_ind_decile_RGBVI_2',
                            'rgb_ind_decile_RGBVI_3','rgb_ind_decile_RGBVI_4',
                            'rgb_ind_decile_RGBVI_5','rgb_ind_decile_RGBVI_6',
                            'rgb_ind_decile_RGBVI_7','rgb_ind_decile_RGBVI_8',
                            'rgb_ind_decile_RGBVI_9','rgb_ind_decile_IKAW_1',
                            'rgb_ind_decile_IKAW_2','rgb_ind_decile_IKAW_3',
                            'rgb_ind_decile_IKAW_4','rgb_ind_decile_IKAW_5',
                            'rgb_ind_decile_IKAW_6','rgb_ind_decile_IKAW_7',
                            'rgb_ind_decile_IKAW_8','rgb_ind_decile_IKAW_9'])
            featClass.extend(['rgb_ind']*144)
            featSizeInvar.extend([True]*144)
            featHeightInvar.extend([True]*144)
            featScale.extend([0]*144)
            if featStack.size==0:
                featStack = self.rgb_ind_deciles.flatten('F')
            else:
                featStack = np.concatenate((featStack,
                                            self.rgb_ind_deciles.flatten('F')))
        if hasattr(self,'rgb_ind_quartiles'):
            featList.extend(['rgb_ind_quartile_GRVI_1','rgb_ind_quartile_GRVI_3',
                            'rgb_ind_quartile_VARI_1','rgb_ind_quartile_VARI_3',
                            'rgb_ind_quartile_GLIr_1','rgb_ind_quartile_GLIr_3',
                            'rgb_ind_quartile_GLIg_1','rgb_ind_quartile_GLIg_3',
                            'rgb_ind_quartile_GLIb_1','rgb_ind_quartile_GLIb_3',
                            'rgb_ind_quartile_ExG_1','rgb_ind_quartile_ExG_3',
                            'rgb_ind_quartile_ExR_1','rgb_ind_quartile_ExR_3',
                            'rgb_ind_quartile_ExB_1','rgb_ind_quartile_ExB_3',
                            'rgb_ind_quartile_ExGveg_1','rgb_ind_quartile_ExGveg_3',
                            'rgb_ind_quartile_NegExR_1','rgb_ind_quartile_NegExR_3',
                            'rgb_ind_quartile_ExRveg_1','rgb_ind_quartile_ExRveg_3',
                            'rgb_ind_quartile_ExBveg_1','rgb_ind_quartile_ExBveg_3',
                            'rgb_ind_quartile_TGI_1','rgb_ind_quartile_TGI_3',
                            'rgb_ind_quartile_mGRVI_1','rgb_ind_quartile_mGRVI_3',
                            'rgb_ind_quartile_RGBVI_1','rgb_ind_quartile_RGBVI_3',
                            'rgb_ind_quartile_IKAW_1','rgb_ind_quartile_IKAW_3'])
            featClass.extend(['rgb_ind']*32)
            featSizeInvar.extend([True]*32)
            featHeightInvar.extend([True]*32)
            featScale.extend([0]*32)
            if featStack.size==0:
                featStack = self.rgb_ind_quartiles.flatten('F')
            else:
                featStack = np.concatenate((featStack,
                                            self.rgb_ind_quartiles.flatten('F')))
        if hasattr(self,'rgb_ind_iqr'):
            featList.extend(['rgb_ind_iqr_GRVI','rgb_ind_iqr_VARI',
                            'rgb_ind_iqr_GLIr','rgb_ind_iqr_GLIg',
                            'rgb_ind_iqr_GLIb','rgb_ind_iqr_ExG',
                            'rgb_ind_iqr_ExR','rgb_ind_iqr_ExB',
                            'rgb_ind_iqr_ExGveg','rgb_ind_iqr_NegExR',
                            'rgb_ind_iqr_ExRveg','rgb_ind_iqr_ExBveg',
                            'rgb_ind_iqr_TGI','rgb_ind_iqr_mGRVI',
                            'rgb_ind_iqr_RGBVI','rgb_ind_iqr_IKAW'])
            featClass.extend(['rgb_ind']*16)
            featSizeInvar.extend([True]*16)
            featHeightInvar.extend([True]*16)
            featScale.extend([0]*16)
            if featStack.size==0:
                featStack = self.rgb_ind_iqr
            else:
                featStack = np.concatenate((featStack,self.rgb_ind_iqr))
        if hasattr(self,'rgb_ind_iqrsig'):
            featList.extend(['rgb_ind_iqrsig_GRVI','rgb_ind_iqrsig_VARI',
                            'rgb_ind_iqrsig_GLIr','rgb_ind_iqrsig_GLIg',
                            'rgb_ind_iqrsig_GLIb','rgb_ind_iqrsig_ExG',
                            'rgb_ind_iqrsig_ExR','rgb_ind_iqrsig_ExB',
                            'rgb_ind_iqrsig_ExGveg','rgb_ind_iqrsig_NegExR',
                            'rgb_ind_iqrsig_ExRveg','rgb_ind_iqrsig_ExBveg',
                            'rgb_ind_iqrsig_TGI','rgb_ind_iqrsig_mGRVI',
                            'rgb_ind_iqrsig_RGBVI','rgb_ind_iqrsig_IKAW'])
            featClass.extend(['rgb_ind']*16)
            featSizeInvar.extend([True]*16)
            featHeightInvar.extend([True]*16)
            featScale.extend([0]*16)
            if featStack.size==0:
                featStack = self.rgb_ind_iqrsig
            else:
                featStack = np.concatenate((featStack,self.rgb_ind_iqrsig))
        if hasattr(self,'rgb_ind_iqrmean'):
            featList.extend(['rgb_ind_iqrmean_GRVI','rgb_ind_iqrmean_VARI',
                            'rgb_ind_iqrmean_GLIr','rgb_ind_iqrmean_GLIg',
                            'rgb_ind_iqrmean_GLIb','rgb_ind_iqrmean_ExG',
                            'rgb_ind_iqrmean_ExR','rgb_ind_iqrmean_ExB',
                            'rgb_ind_iqrmean_ExGveg','rgb_ind_iqrmean_NegExR',
                            'rgb_ind_iqrmean_ExRveg','rgb_ind_iqrmean_ExBveg',
                            'rgb_ind_iqrmean_TGI','rgb_ind_iqrmean_mGRVI',
                            'rgb_ind_iqrmean_RGBVI','rgb_ind_iqrmean_IKAW'])
            featClass.extend(['rgb_ind']*16)
            featSizeInvar.extend([True]*16)
            featHeightInvar.extend([True]*16)
            featScale.extend([0]*16)
            if featStack.size==0:
                featStack = self.rgb_ind_iqrmean
            else:
                featStack = np.concatenate((featStack,self.rgb_ind_iqrmean))
        if hasattr(self,'glcm_rgb_vals'):
            for rgb_glcm_d in self.glcm_rgb_dist:
                glcm_list = ['glcm_rgb_asm_' + str(rgb_glcm_d),
                            'glcm_rgb_con_' + str(rgb_glcm_d),
                            'glcm_rgb_cor_' + str(rgb_glcm_d),
                            'glcm_rgb_var_' + str(rgb_glcm_d),
                            'glcm_rgb_idm_' + str(rgb_glcm_d),
                            'glcm_rgb_sumav_' + str(rgb_glcm_d),
                            'glcm_rgb_sumvar_' + str(rgb_glcm_d),
                            'glcm_rgb_sument_' + str(rgb_glcm_d),
                            'glcm_rgb_ent_' + str(rgb_glcm_d),
                            'glcm_rgb_difvar_' + str(rgb_glcm_d),
                            'glcm_rgb_difent_' + str(rgb_glcm_d),
                            'glcm_rgb_infcor1_' + str(rgb_glcm_d),
                            'glcm_rgb_infcor2_' + str(rgb_glcm_d),
                            'glcm_rgb_asm_rng_' + str(rgb_glcm_d),
                            'glcm_rgb_con_rng_' + str(rgb_glcm_d),
                            'glcm_rgb_cor_rng_' + str(rgb_glcm_d),
                            'glcm_rgb_var_rng_' + str(rgb_glcm_d),
                            'glcm_rgb_idm_rng_' + str(rgb_glcm_d),
                            'glcm_rgb_sumav_rng_' + str(rgb_glcm_d),
                            'glcm_rgb_sumvar_rng_' + str(rgb_glcm_d),
                            'glcm_rgb_sument_rng_' + str(rgb_glcm_d),
                            'glcm_rgb_ent_rng_' + str(rgb_glcm_d),
                            'glcm_rgb_difvar_rng_' + str(rgb_glcm_d),
                            'glcm_rgb_difent_rng_' + str(rgb_glcm_d),
                            'glcm_rgb_infcor1_rng_' + str(rgb_glcm_d),
                            'glcm_rgb_infcor2_rng_' + str(rgb_glcm_d)]
                featList.extend(glcm_list)
                featClass.extend(['rgb_glcm']*26)
                featSizeInvar.extend([True]*26)
                featHeightInvar.extend([True]*26)
                featScale.extend([rgb_glcm_d]*26)
            if featStack.size==0:
                featStack = self.glcm_rgb_vals
            else:
                featStack = np.concatenate((featStack,self.glcm_rgb_vals))
        if hasattr(self,'acor_rgb_vals'):
            for rgb_acor_d in self.acor_rgb_dist:
                acor_list = ['acor_rgb_mean_' + str(rgb_acor_d),
                            'acor_rgb_rng_' + str(rgb_acor_d)]
                featList.extend(acor_list)
                featClass.extend(['rgb_acor']*2)
                featSizeInvar.extend([True]*2)
                featHeightInvar.extend([True]*2)
                featScale.extend([rgb_acor_d]*2)
            if featStack.size==0:
                featStack = self.acor_rgb_vals
            else:
                featStack = np.concatenate((featStack,self.acor_rgb_vals))
        if hasattr(self,'lbp_rgb_vals'):
            for rgb_lbp_d in self.lbp_rgb_dist:
                for ft_i in range(2+8*rgb_lbp_d):
                    featList.extend(
                        ['lbp_rgb_d_' + str(rgb_lbp_d) + '_feat_' + str(ft_i)]
                    )
                    featClass.extend(['rgb_lbp'])
                    featSizeInvar.extend([True])
                    featHeightInvar.extend([True])
                    featScale.extend([rgb_lbp_d])
            if featStack.size==0:
                featStack = self.lbp_rgb_vals
            else:
                featStack = np.concatenate((featStack,self.lbp_rgb_vals))
        if hasattr(self,'laws_rgb_feats'):
            laws_list = []
            filtbank = ['L5','E5','S5','R5','W5']
            for stat in ['mean','std']:
                for i in range(5):
                    for j in range(5):
                            if j < i or (i==0 and j ==0):
                                continue
                            else:
                                featList.append('laws_' + filtbank[i] + filtbank[j] +'_RGB_' + stat)
                                featClass.extend(['rgb_laws'])
                                featSizeInvar.extend([True])
                                featHeightInvar.extend([True])
                                featScale.extend([0])
            if featStack.size==0:
                featStack = self.laws_rgb_feats.flatten('F')
            else:
                featStack = np.concatenate((featStack,self.laws_rgb_feats.flatten('F')))
        if hasattr(self,'ms_band_max'):
            featList.extend(['ms_band_max_G','ms_band_max_R','ms_band_max_RE','ms_band_max_NIR'])
            featClass.extend(['ms_band']*4)
            featSizeInvar.extend([True]*4)
            featHeightInvar.extend([True]*4)
            featScale.extend([0]*4)
            if featStack.size==0:
                featStack = self.ms_band_max
            else:
                featStack = np.concatenate((featStack,self.ms_band_max))
        if hasattr(self,'ms_band_min'):
            featList.extend(['ms_band_min_G','ms_band_min_R','ms_band_min_RE','ms_band_min_NIR'])
            featClass.extend(['ms_band']*4)
            featSizeInvar.extend([True]*4)
            featHeightInvar.extend([True]*4)
            featScale.extend([0]*4)
            if featStack.size==0:
                featStack = self.ms_band_min
            else:
                featStack = np.concatenate((featStack,self.ms_band_min))
        if hasattr(self,'ms_band_mean'):
            featList.extend(['ms_band_mean_G','ms_band_mean_R','ms_band_mean_RE','ms_band_mean_NIR'])
            featClass.extend(['ms_band']*4)
            featSizeInvar.extend([True]*4)
            featHeightInvar.extend([True]*4)
            featScale.extend([0]*4)
            if featStack.size==0:
                featStack = self.ms_band_mean
            else:
                featStack = np.concatenate((featStack,self.ms_band_mean))
        if hasattr(self,'ms_band_std'):
            featList.extend(['ms_band_std_G','ms_band_std_R','ms_band_std_RE','ms_band_std_NIR'])
            featClass.extend(['ms_band']*4)
            featSizeInvar.extend([True]*4)
            featHeightInvar.extend([True]*4)
            featScale.extend([0]*4)
            if featStack.size==0:
                featStack = self.ms_band_std
            else:
                featStack = np.concatenate((featStack,self.ms_band_std))
        if hasattr(self,'ms_band_median'):
            featList.extend(['ms_band_median_G','ms_band_median_R','ms_band_median_RE','ms_band_median_NIR'])
            featClass.extend(['ms_band']*4)
            featSizeInvar.extend([True]*4)
            featHeightInvar.extend([True]*4)
            featScale.extend([0]*4)
            if featStack.size==0:
                featStack = self.ms_band_median
            else:
                featStack = np.concatenate((featStack,self.ms_band_median))
        if hasattr(self,'ms_band_cov'):
            featList.extend(['ms_band_cov_G','ms_band_cov_R','ms_band_cov_RE','ms_band_cov_NIR'])
            featClass.extend(['ms_band']*4)
            featSizeInvar.extend([True]*4)
            featHeightInvar.extend([True]*4)
            featScale.extend([0]*4)
            if featStack.size==0:
                featStack = self.ms_band_cov
            else:
                featStack = np.concatenate((featStack,self.ms_band_cov))
        if hasattr(self,'ms_band_skew'):
            featList.extend(['ms_band_skew_G','ms_band_skew_R','ms_band_skew_RE','ms_band_skew_NIR'])
            featClass.extend(['ms_band']*4)
            featSizeInvar.extend([True]*4)
            featHeightInvar.extend([True]*4)
            featScale.extend([0]*4)
            if featStack.size==0:
                featStack = self.ms_band_skew
            else:
                featStack = np.concatenate((featStack,self.ms_band_skew))
        if hasattr(self,'ms_band_kurt'):
            featList.extend(['ms_band_kurt_G','ms_band_kurt_R','ms_band_kurt_RE','ms_band_kurt_NIR'])
            featClass.extend(['ms_band']*4)
            featSizeInvar.extend([True]*4)
            featHeightInvar.extend([True]*4)
            featScale.extend([0]*4)
            if featStack.size==0:
                featStack = self.ms_band_kurt
            else:
                featStack = np.concatenate((featStack,self.ms_band_kurt))
        if hasattr(self,'ms_band_sum'):
            featList.extend(['ms_band_sum_G','ms_band_sum_R','ms_band_sum_RE','ms_band_sum_NIR'])
            featClass.extend(['ms_band']*4)
            featSizeInvar.extend([False]*4)
            featHeightInvar.extend([True]*4)
            featScale.extend([0]*4)
            if featStack.size==0:
                featStack = self.ms_band_sum
            else:
                featStack = np.concatenate((featStack,self.ms_band_sum))
        if hasattr(self,'ms_band_rng'):
            featList.extend(['ms_band_rng_G','ms_band_rng_R','ms_band_rng_RE','ms_band_rng_NIR'])
            featClass.extend(['ms_band']*4)
            featSizeInvar.extend([True]*4)
            featHeightInvar.extend([True]*4)
            featScale.extend([0]*4)
            if featStack.size==0:
                featStack = self.band_rng
            else:
                featStack = np.concatenate((featStack,self.ms_band_rng))
        if hasattr(self,'ms_band_rngsig'):
            featList.extend(['ms_band_rngsig_G','ms_band_rngsig_R','ms_band_rngsig_RE','ms_band_rngsig_NIR'])
            featClass.extend(['ms_band']*4)
            featSizeInvar.extend([True]*4)
            featHeightInvar.extend([True]*4)
            featScale.extend([0]*4)
            if featStack.size==0:
                featStack = self.ms_band_rngsig
            else:
                featStack = np.concatenate((featStack,self.ms_band_rngsig))
        if hasattr(self,'ms_band_rngmean'):
            featList.extend(['ms_band_rngmean_G','ms_band_rngmean_R','ms_band_rngmean_RE','ms_band_rngmean_NIR'])
            featClass.extend(['ms_band']*4)
            featSizeInvar.extend([True]*4)
            featHeightInvar.extend([True]*4)
            featScale.extend([0]*4)
            if featStack.size==0:
                featStack = self.ms_band_rngmean
            else:
                featStack = np.concatenate((featStack,self.ms_band_rngmean))
        if hasattr(self,'ms_band_mode'):
            featList.extend(['ms_band_mode_G','ms_band_mode_R','ms_band_mode_RE','ms_band_mode_NIR'])
            featClass.extend(['ms_band']*4)
            featSizeInvar.extend([True]*4)
            featHeightInvar.extend([True]*4)
            featScale.extend([0]*4)
            if featStack.size==0:
                featStack = self.ms_band_mode
            else:
                featStack = np.concatenate((featStack,self.ms_band_mode))
        if hasattr(self,'ms_band_deciles'):
            featList.extend(['ms_band_decile_G_1','ms_band_decile_G_2',
                            'ms_band_decile_G_3','ms_band_decile_G_4',
                            'ms_band_decile_G_5','ms_band_decile_G_6',
                            'ms_band_decile_G_7','ms_band_decile_G_8',
                            'ms_band_decile_G_9','ms_band_decile_R_1',
                            'ms_band_decile_R_2','ms_band_decile_R_3',
                            'ms_band_decile_R_4','ms_band_decile_R_5',
                            'ms_band_decile_R_6','ms_band_decile_R_7',
                            'ms_band_decile_R_8','ms_band_decile_R_9',
                            'ms_band_decile_RE_1','ms_band_decile_RE_2',
                            'ms_band_decile_RE_3','ms_band_decile_RE_4',
                            'ms_band_decile_RE_5','ms_band_decile_RE_6',
                            'ms_band_decile_RE_7','ms_band_decile_RE_8',
                            'ms_band_decile_RE_9','ms_band_decile_NIR_1',
                            'ms_band_decile_NIR_2','ms_band_decile_NIR_3',
                            'ms_band_decile_NIR_4','ms_band_decile_NIR_5',
                            'ms_band_decile_NIR_6','ms_band_decile_NIR_7',
                            'ms_band_decile_NIR_8','ms_band_decile_NIR_9'])
            featClass.extend(['ms_band']*36)
            featSizeInvar.extend([True]*36)
            featHeightInvar.extend([True]*36)
            featScale.extend([0]*36)
            if featStack.size==0:
                featStack = self.ms_band_deciles.flatten('F')
            else:
                featStack = np.concatenate((featStack,
                                            self.ms_band_deciles.flatten('F')))
        if hasattr(self,'ms_band_quartiles'):
            featList.extend(['ms_band_quartile_G_1','ms_band_quartile_G_3',
                            'ms_band_quartile_R_1','ms_band_quartile_R_3',
                            'ms_band_quartile_RE_1','ms_band_quartile_RE_3',
                            'ms_band_quartile_NIR_1','ms_band_quartile_NIR_3'])
            featClass.extend(['ms_band']*8)
            featSizeInvar.extend([True]*8)
            featHeightInvar.extend([True]*8)
            featScale.extend([0]*8)
            if featStack.size==0:
                featStack = self.ms_band_quartiles.flatten('F')
            else:
                featStack = np.concatenate((featStack,
                                            self.ms_band_quartiles.flatten('F')))
        if hasattr(self,'ms_band_iqr'):
            featList.extend(['ms_band_iqr_G','ms_band_iqr_R','ms_band_iqr_RE','ms_band_iqr_NIR'])
            featClass.extend(['ms_band']*4)
            featSizeInvar.extend([True]*4)
            featHeightInvar.extend([True]*4)
            featScale.extend([0]*4)
            if featStack.size==0:
                featStack = self.ms_band_iqr
            else:
                featStack = np.concatenate((featStack,self.ms_band_iqr))
        if hasattr(self,'ms_band_iqrsig'):
            featList.extend(['ms_band_iqrsig_G','ms_band_iqrsig_R','ms_band_iqrsig_RE','ms_band_iqrsig_NIR'])
            featClass.extend(['ms_band']*4)
            featSizeInvar.extend([True]*4)
            featHeightInvar.extend([True]*4)
            featScale.extend([0]*4)
            if featStack.size==0:
                featStack = self.ms_band_iqrsig
            else:
                featStack = np.concatenate((featStack,self.ms_band_iqrsig))
        if hasattr(self,'ms_band_iqrmean'):
            featList.extend(['ms_band_iqrmean_G','ms_band_iqrmean_R','ms_band_iqrmean_RE','ms_band_iqrmean_NIR'])
            featClass.extend(['ms_band']*4)
            featSizeInvar.extend([True]*4)
            featHeightInvar.extend([True]*4)
            featScale.extend([0]*4)
            if featStack.size==0:
                featStack = self.ms_band_iqrmean
            else:
                featStack = np.concatenate((featStack,self.ms_band_iqrmean))
        if hasattr(self,'ms_band_ratio'):
            featList.extend(['ms_band_ratio_G','ms_band_ratio_R','ms_band_ratio_RE','ms_band_ratio_NIR'])
            featClass.extend(['ms_band']*4)
            featSizeInvar.extend([True]*4)
            featHeightInvar.extend([True]*4)
            featScale.extend([0]*4)
            if featStack.size==0:
                featStack = self.ms_band_ratio
            else:
                featStack = np.concatenate((featStack,self.ms_band_ratio))
        if hasattr(self,'ms_ind_max'):
            featList.extend(['ms_ind_max_NDVI','ms_ind_max_NDVIg',
                            'ms_ind_max_NDVIre','ms_ind_max_CIG',
                            'ms_ind_max_CVI','ms_ind_max_GRVI',
                            'ms_ind_max_mGRVI','ms_ind_max_NegExR'])
            featClass.extend(['ms_ind']*8)
            featSizeInvar.extend([True]*8)
            featHeightInvar.extend([True]*8)
            featScale.extend([0]*8)
            if featStack.size==0:
                featStack = self.ms_ind_max
            else:
                featStack = np.concatenate((featStack,self.ms_ind_max))
        if hasattr(self,'ms_ind_min'):
            featList.extend(['ms_ind_min_NDVI','ms_ind_min_NDVIg',
                            'ms_ind_min_NDVIre','ms_ind_min_CIG',
                            'ms_ind_min_CVI','ms_ind_min_GRVI',
                            'ms_ind_min_mGRVI','ms_ind_min_NegExR'])
            featClass.extend(['ms_ind']*8)
            featSizeInvar.extend([True]*8)
            featHeightInvar.extend([True]*8)
            featScale.extend([0]*8)
            if featStack.size==0:
                featStack = self.ms_ind_min
            else:
                featStack = np.concatenate((featStack,self.ms_ind_min))
        if hasattr(self,'ms_ind_mean'):
            featList.extend(['ms_ind_mean_NDVI','ms_ind_mean_NDVIg',
                            'ms_ind_mean_NDVIre','ms_ind_mean_CIG',
                            'ms_ind_mean_CVI','ms_ind_mean_GRVI',
                            'ms_ind_mean_mGRVI','ms_ind_mean_NegExR'])
            featClass.extend(['ms_ind']*8)
            featSizeInvar.extend([True]*8)
            featHeightInvar.extend([True]*8)
            featScale.extend([0]*8)
            if featStack.size==0:
                featStack = self.ms_ind_mean
            else:
                featStack = np.concatenate((featStack,self.ms_ind_mean))
        if hasattr(self,'ms_ind_std'):
            featList.extend(['ms_ind_std_NDVI','ms_ind_std_NDVIg',
                            'ms_ind_std_NDVIre','ms_ind_std_CIG',
                            'ms_ind_std_CVI','ms_ind_std_GRVI',
                            'ms_ind_std_mGRVI','ms_ind_std_NegExR'])
            featClass.extend(['ms_ind']*8)
            featSizeInvar.extend([True]*8)
            featHeightInvar.extend([True]*8)
            featScale.extend([0]*8)
            if featStack.size==0:
                featStack = self.ms_ind_std
            else:
                featStack = np.concatenate((featStack,self.ms_ind_std))
        if hasattr(self,'ms_ind_median'):
            featList.extend(['ms_ind_median_NDVI','ms_ind_median_NDVIg',
                            'ms_ind_median_NDVIre','ms_ind_median_CIG',
                            'ms_ind_median_CVI','ms_ind_median_GRVI',
                            'ms_ind_median_mGRVI','ms_ind_median_NegExR'])
            featClass.extend(['ms_ind']*8)
            featSizeInvar.extend([True]*8)
            featHeightInvar.extend([True]*8)
            featScale.extend([0]*8)
            if featStack.size==0:
                featStack = self.ms_ind_median
            else:
                featStack = np.concatenate((featStack,self.ms_ind_median))
        if hasattr(self,'ms_ind_cov'):
            featList.extend(['ms_ind_cov_NDVI','ms_ind_cov_NDVIg',
                            'ms_ind_cov_NDVIre','ms_ind_cov_CIG',
                            'ms_ind_cov_CVI','ms_ind_cov_GRVI',
                            'ms_ind_cov_mGRVI','ms_ind_cov_NegExR'])
            featClass.extend(['ms_ind']*8)
            featSizeInvar.extend([True]*8)
            featHeightInvar.extend([True]*8)
            featScale.extend([0]*8)
            if featStack.size==0:
                featStack = self.ms_ind_cov
            else:
                featStack = np.concatenate((featStack,self.ms_ind_cov))
        if hasattr(self,'ms_ind_skew'):
            featList.extend(['ms_ind_skew_NDVI','ms_ind_skew_NDVIg',
                            'ms_ind_skew_NDVIre','ms_ind_skew_CIG',
                            'ms_ind_skew_CVI','ms_ind_skew_GRVI',
                            'ms_ind_skew_mGRVI','ms_ind_skew_NegExR'])
            featClass.extend(['ms_ind']*8)
            featSizeInvar.extend([True]*8)
            featHeightInvar.extend([True]*8)
            featScale.extend([0]*8)
            if featStack.size==0:
                featStack = self.ms_ind_skew
            else:
                featStack = np.concatenate((featStack,self.ms_ind_skew))
        if hasattr(self,'ms_ind_kurt'):
            featList.extend(['ms_ind_kurt_NDVI','ms_ind_kurt_NDVIg',
                            'ms_ind_kurt_NDVIre','ms_ind_kurt_CIG',
                            'ms_ind_kurt_CVI','ms_ind_kurt_GRVI',
                            'ms_ind_kurt_mGRVI','ms_ind_kurt_NegExR'])
            featClass.extend(['ms_ind']*8)
            featSizeInvar.extend([True]*8)
            featHeightInvar.extend([True]*8)
            featScale.extend([0]*8)
            if featStack.size==0:
                featStack = self.ms_ind_kurt
            else:
                featStack = np.concatenate((featStack,self.ms_ind_kurt))
        if hasattr(self,'ms_ind_sum'):
            featList.extend(['ms_ind_sum_NDVI','ms_ind_sum_NDVIg',
                            'ms_ind_sum_NDVIre','ms_ind_sum_CIG',
                            'ms_ind_sum_CVI','ms_ind_sum_GRVI',
                            'ms_ind_sum_mGRVI','ms_ind_sum_NegExR'])
            featClass.extend(['ms_ind']*8)
            featSizeInvar.extend([False]*8)
            featHeightInvar.extend([True]*8)
            featScale.extend([0]*8)
            if featStack.size==0:
                featStack = self.ms_ind_sum
            else:
                featStack = np.concatenate((featStack,self.ms_ind_sum))
        if hasattr(self,'ms_ind_rng'):
            featList.extend(['ms_ind_rng_NDVI','ms_ind_rng_NDVIg',
                            'ms_ind_rng_NDVIre','ms_ind_rng_CIG',
                            'ms_ind_rng_CVI','ms_ind_rng_GRVI',
                            'ms_ind_rng_mGRVI','ms_ind_rng_NegExR'])
            featClass.extend(['ms_ind']*8)
            featSizeInvar.extend([True]*8)
            featHeightInvar.extend([True]*8)
            featScale.extend([0]*8)
            if featStack.size==0:
                featStack = self.ms_ind_rng
            else:
                featStack = np.concatenate((featStack,self.ms_ind_rng))
        if hasattr(self,'ms_ind_rngsig'):
            featList.extend(['ms_ind_rngsig_NDVI','ms_ind_rngsig_NDVIg',
                            'ms_ind_rngsig_NDVIre','ms_ind_rngsig_CIG',
                            'ms_ind_rngsig_CVI','ms_ind_rngsig_GRVI',
                            'ms_ind_rngsig_mGRVI','ms_ind_rngsig_NegExR'])
            featClass.extend(['ms_ind']*8)
            featSizeInvar.extend([True]*8)
            featHeightInvar.extend([True]*8)
            featScale.extend([0]*8)
            if featStack.size==0:
                featStack = self.ms_ind_rngsig
            else:
                featStack = np.concatenate((featStack,self.ms_ind_rngsig))
        if hasattr(self,'ms_ind_rngmean'):
            featList.extend(['ms_ind_rngsmean_NDVI','ms_ind_rngsmean_NDVIg',
                            'ms_ind_rngsmean_NDVIre','ms_ind_rngsmean_CIG',
                            'ms_ind_rngsmean_CVI','ms_ind_rngsmean_GRVI',
                            'ms_ind_rngsmean_mGRVI','ms_ind_rngsmean_NegExR'])
            featClass.extend(['ms_ind']*8)
            featSizeInvar.extend([True]*8)
            featHeightInvar.extend([True]*8)
            featScale.extend([0]*8)
            if featStack.size==0:
                featStack = self.ms_ind_rngmean
            else:
                featStack = np.concatenate((featStack,self.ms_ind_rngmean))
        if hasattr(self,'ms_ind_mode'):
            featList.extend(['ms_ind_mode_NDVI','ms_ind_mode_NDVIg',
                            'ms_ind_mode_NDVIre','ms_ind_mode_CIG',
                            'ms_ind_mode_CVI','ms_ind_mode_GRVI',
                            'ms_ind_mode_mGRVI','ms_ind_mode_NegExR'])
            featClass.extend(['ms_ind']*8)
            featSizeInvar.extend([True]*8)
            featHeightInvar.extend([True]*8)
            featScale.extend([0]*8)
            if featStack.size==0:
                featStack = self.ms_ind_mode
            else:
                featStack = np.concatenate((featStack,self.ms_ind_mode))
        if hasattr(self,'ms_ind_deciles'):
            featList.extend(['ms_ind_decile_NDVI_1','ms_ind_decile_NDVI_2',
                            'ms_ind_decile_NDVI_3','ms_ind_decile_NDVI_4',
                            'ms_ind_decile_NDVI_5','ms_ind_decile_NDVI_6',
                            'ms_ind_decile_NDVI_7','ms_ind_decile_NDVI_8',
                            'ms_ind_decile_NDVI_9','ms_ind_decile_NDVIg_1',
                            'ms_ind_decile_NDVIg_2','ms_ind_decile_NDVIg_3',
                            'ms_ind_decile_NDVIg_4','ms_ind_decile_NDVIg_5',
                            'ms_ind_decile_NDVIg_6','ms_ind_decile_NDVIg_7',
                            'ms_ind_decile_NDVIg_8','ms_ind_decile_NDVIg_9',
                            'ms_ind_decile_NDVIre_1','ms_ind_decile_NDVIre_2',
                            'ms_ind_decile_NDVIre_3','ms_ind_decile_NDVIre_4',
                            'ms_ind_decile_NDVIre_5','ms_ind_decile_NDVIre_6',
                            'ms_ind_decile_NDVIre_7','ms_ind_decile_NDVIre_8',
                            'ms_ind_decile_NDVIre_9','ms_ind_decile_CIG_1',
                            'ms_ind_decile_CIG_2','ms_ind_decile_CIG_3',
                            'ms_ind_decile_CIG_4','ms_ind_decile_CIG_5',
                            'ms_ind_decile_CIG_6','ms_ind_decile_CIG_7',
                            'ms_ind_decile_CIG_8','ms_ind_decile_CIG_9',
                            'ms_ind_decile_CVI_1','ms_ind_decile_CVI_2',
                            'ms_ind_decile_CVI_3','ms_ind_decile_CVI_4',
                            'ms_ind_decile_CVI_5','ms_ind_decile_CVI_6',
                            'ms_ind_decile_CVI_7','ms_ind_decile_CVI_8',
                            'ms_ind_decile_CVI_9','ms_ind_decile_GRVI_1',
                            'ms_ind_decile_GRVI_2','ms_ind_decile_GRVI_3',
                            'ms_ind_decile_GRVI_4','ms_ind_decile_GRVI_5',
                            'ms_ind_decile_GRVI_6','ms_ind_decile_GRVI_7',
                            'ms_ind_decile_GRVI_8','ms_ind_decile_GRVI_9',
                            'ms_ind_decile_mGRVI_1','ms_ind_decile_mGRVI_2',
                            'ms_ind_decile_mGRVI_3','ms_ind_decile_mGRVI_4',
                            'ms_ind_decile_mGRVI_5','ms_ind_decile_mGRVI_6',
                            'ms_ind_decile_mGRVI_7','ms_ind_decile_mGRVI_8',
                            'ms_ind_decile_mGRVI_9','ms_ind_decile_NegExR_1',
                            'ms_ind_decile_NegExR_2','ms_ind_decile_NegExR_3',
                            'ms_ind_decile_NegExR_4','ms_ind_decile_NegExR_5',
                            'ms_ind_decile_NegExR_6','ms_ind_decile_NegExR_7',
                            'ms_ind_decile_NegExR_8','ms_ind_decile_NegExR_9'])
            featClass.extend(['ms_ind']*72)
            featSizeInvar.extend([True]*72)
            featHeightInvar.extend([True]*72)
            featScale.extend([0]*72)
            if featStack.size==0:
                featStack = self.ms_ind_deciles.flatten('F')
            else:
                featStack = np.concatenate((featStack,
                                            self.ms_ind_deciles.flatten('F')))
        if hasattr(self,'ms_ind_quartiles'):
            featList.extend(['ms_ind_quartile_NDVI_1','ms_ind_quartile_NDVI_3',
                            'ms_ind_quartile_NDVIg_1','ms_ind_quartile_NDVIg_3',
                            'ms_ind_quartile_NDVIre_1','ms_ind_quartile_NDVIre_3',
                            'ms_ind_quartile_CIG_1','ms_ind_quartile_CIG_3',
                            'ms_ind_quartile_CVI_1','ms_ind_quartile_CVI_3',
                            'ms_ind_quartile_GRVI_1','ms_ind_quartile_GRVI_3',
                            'ms_ind_quartile_mGRVI_1','ms_ind_quartile_mGRVI_3',
                            'ms_ind_quartile_NegExR_1','ms_ind_quartile_NegExR_3'])
            featClass.extend(['ms_ind']*16)
            featSizeInvar.extend([True]*16)
            featHeightInvar.extend([True]*16)
            featScale.extend([0]*16)
            if featStack.size==0:
                featStack = self.ms_ind_quartiles.flatten('F')
            else:
                featStack = np.concatenate((featStack,
                                            self.ms_ind_quartiles.flatten('F')))
        if hasattr(self,'ms_ind_iqr'):
            featList.extend(['ms_ind_iqr_NDVI','ms_ind_iqr_NDVIg',
                            'ms_ind_iqr_NDVIre','ms_ind_iqr_CIG',
                            'ms_ind_iqr_CVI','ms_ind_iqr_GRVI',
                            'ms_ind_iqr_mGRVI','ms_ind_iqr_NegExR'])
            featClass.extend(['ms_ind']*8)
            featSizeInvar.extend([True]*8)
            featHeightInvar.extend([True]*8)
            featScale.extend([0]*8)
            if featStack.size==0:
                featStack = self.ms_ind_iqr
            else:
                featStack = np.concatenate((featStack,self.ms_ind_iqr))
        if hasattr(self,'ms_ind_iqrsig'):
            featList.extend(['ms_ind_iqrsig_NDVI','ms_ind_iqrsig_NDVIg',
                            'ms_ind_iqrsig_NDVIre','ms_ind_iqrsig_CIG',
                            'ms_ind_iqrsig_CVI','ms_ind_iqrsig_GRVI',
                            'ms_ind_iqrsig_mGRVI','ms_ind_iqrsig_NegExR'])
            featClass.extend(['ms_ind']*8)
            featSizeInvar.extend([True]*8)
            featHeightInvar.extend([True]*8)
            featScale.extend([0]*8)
            if featStack.size==0:
                featStack = self.ms_ind_iqrsig
            else:
                featStack = np.concatenate((featStack,self.ms_ind_iqrsig))
        if hasattr(self,'ms_ind_iqrmean'):
            featList.extend(['ms_ind_iqrmean_NDVI','ms_ind_iqrmean_NDVIg',
                            'ms_ind_iqrmean_NDVIre','ms_ind_iqrmean_CIG',
                            'ms_ind_iqrmean_CVI','ms_ind_iqrmean_GRVI',
                            'ms_ind_iqrmean_mGRVI','ms_ind_iqrmean_NegExR'])
            featClass.extend(['ms_ind']*8)
            featSizeInvar.extend([True]*8)
            featHeightInvar.extend([True]*8)
            featScale.extend([0]*8)
            if featStack.size==0:
                featStack = self.ms_ind_iqrmean
            else:
                featStack = np.concatenate((featStack,self.ms_ind_iqrmean))
        if hasattr(self,'glcm_ms_vals'):
            for ms_glcm_d in self.glcm_ms_dist:
                for band in ['G','R','RE','NIR','mean']:
                    glcm_list = ['glcm_ms_asm_' + str(ms_glcm_d) + '_' + band,
                                'glcm_ms_con_' + str(ms_glcm_d) + '_' + band,
                                'glcm_ms_cor_' + str(ms_glcm_d) + '_' + band,
                                'glcm_ms_var_' + str(ms_glcm_d) + '_' + band,
                                'glcm_ms_idm_' + str(ms_glcm_d) + '_' + band,
                                'glcm_ms_sumav_' + str(ms_glcm_d) + '_' + band,
                                'glcm_ms_sumvar_' + str(ms_glcm_d) + '_' + band,
                                'glcm_ms_sument_' + str(ms_glcm_d) + '_' + band,
                                'glcm_ms_ent_' + str(ms_glcm_d) + '_' + band,
                                'glcm_ms_difvar_' + str(ms_glcm_d) + '_' + band,
                                'glcm_ms_difent_' + str(ms_glcm_d) + '_' + band,
                                'glcm_ms_infcor1_' + str(ms_glcm_d) + '_' + band,
                                'glcm_ms_infcor2_' + str(ms_glcm_d) + '_' + band,
                                'glcm_ms_asm_rng_' + str(ms_glcm_d) + '_' + band,
                                'glcm_ms_con_rng_' + str(ms_glcm_d) + '_' + band,
                                'glcm_ms_cor_rng_' + str(ms_glcm_d) + '_' + band,
                                'glcm_ms_var_rng_' + str(ms_glcm_d) + '_' + band,
                                'glcm_ms_idm_rng_' + str(ms_glcm_d) + '_' + band,
                                'glcm_ms_sumav_rng_' + str(ms_glcm_d) + '_' + band,
                                'glcm_ms_sumvar_rng_' + str(ms_glcm_d) + '_' + band,
                                'glcm_ms_sument_rng_' + str(ms_glcm_d) + '_' + band,
                                'glcm_ms_ent_rng_' + str(ms_glcm_d) + '_' + band,
                                'glcm_ms_difvar_rng_' + str(ms_glcm_d) + '_' + band,
                                'glcm_ms_difent_rng_' + str(ms_glcm_d) + '_' + band,
                                'glcm_ms_infcor1_rng_' + str(ms_glcm_d) + '_' + band,
                                'glcm_ms_infcor2_rng_' + str(ms_glcm_d) + '_' + band]
                    featList.extend(glcm_list)
                    featClass.extend(['ms_glcm']*26)
                    featSizeInvar.extend([True]*26)
                    featHeightInvar.extend([True]*26)
                    featScale.extend([ms_glcm_d]*26)
            if featStack.size==0:
                featStack = self.glcm_ms_vals
            else:
                featStack = np.concatenate((featStack,self.glcm_ms_vals))
        if hasattr(self,'acor_ms_vals'):
            for acor_ms_d in self.acor_ms_dist:
                for band in ['G','R','RE','NIR','mean']:
                    acor_list = ['acor_ms_mean_' + str(acor_ms_d) + '_' + band,
                                'acor_ms_rng_' + str(acor_ms_d) + '_' + band]
                    featList.extend(acor_list)
                    featClass.extend(['ms_acor']*2)
                    featSizeInvar.extend([True]*2)
                    featHeightInvar.extend([True]*2)
                    featScale.extend([acor_ms_d]*2)
            if featStack.size==0:
                featStack = self.acor_ms_vals
            else:
                featStack = np.concatenate((featStack,self.acor_ms_vals))
        if hasattr(self,'lbp_ms_vals'):
            for ms_lbp_d in self.lbp_ms_dist:
                for band in ['G','R','RE','NIR','mean']:
                    for ft_i in range(2+8*ms_lbp_d):
                        featList.extend(
                            ['lbp_ms_d_' + str(ms_lbp_d) + '_' + band +  '_feat_' + str(ft_i)]
                        )
                        featClass.extend(['ms_lbp'])
                        featSizeInvar.extend([True])
                        featHeightInvar.extend([True])
                        featScale.extend([ms_lbp_d])
            if featStack.size==0:
                featStack = self.lbp_ms_vals
            else:
                featStack = np.concatenate((featStack,self.lbp_ms_vals))
        if hasattr(self,'laws_ms_feats'):
            laws_list = []
            filtbank = ['L5','E5','S5','R5','W5']
            for band in ['G','R','RE','NIR','mean']:
                for stat in ['mean','std']:
                    for i in range(5):
                        for j in range(5):
                                if j < i or (i==0 and j ==0):
                                    continue
                                else:
                                    featList.append('laws_' + filtbank[i] + filtbank[j] +'_'+band+'_' + stat)
                                    featClass.extend(['ms_laws'])
                                    featSizeInvar.extend([True])
                                    featHeightInvar.extend([True])
                                    featScale.extend([0])
            if featStack.size==0:
                featStack = self.laws_ms_feats.flatten('F')
            else:
                featStack = np.concatenate((featStack,self.laws_ms_feats.flatten('F')))
        if hasattr(self,'hsv_max'):
            featList.extend(['hsv_max_H','hsv_max_S','hsv_max_V'])
            featClass.extend(['rgb_hsv']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.hsv_max
            else:
                featStack = np.concatenate((featStack,self.hsv_max))
        if hasattr(self,'hsv_min'):
            featList.extend(['hsv_min_H','hsv_min_S','hsv_min_V'])
            featClass.extend(['rgb_hsv']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.hsv_min
            else:
                featStack = np.concatenate((featStack,self.hsv_min))
        if hasattr(self,'hsv_mean'):
            featList.extend(['hsv_mean_H','hsv_mean_S','hsv_mean_V'])
            featClass.extend(['rgb_hsv']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.hsv_mean
            else:
                featStack = np.concatenate((featStack,self.hsv_mean))
        if hasattr(self,'hsv_std'):
            featList.extend(['hsv_std_H','hsv_std_S','hsv_std_V'])
            featClass.extend(['rgb_hsv']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.hsv_std
            else:
                featStack = np.concatenate((featStack,self.hsv_std))
        if hasattr(self,'hsv_median'):
            featList.extend(['hsv_median_H','hsv_median_S','hsv_median_V'])
            featClass.extend(['rgb_hsv']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.hsv_median
            else:
                featStack = np.concatenate((featStack,self.hsv_median))
        if hasattr(self,'hsv_cov'):
            featList.extend(['hsv_cov_H','hsv_cov_S','hsv_cov_V'])
            featClass.extend(['rgb_hsv']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.hsv_cov
            else:
                featStack = np.concatenate((featStack,self.hsv_cov))
        if hasattr(self,'hsv_skew'):
            featList.extend(['hsv_skew_H','hsv_skew_S','hsv_skew_V'])
            featClass.extend(['rgb_hsv']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.hsv_skew
            else:
                featStack = np.concatenate((featStack,self.hsv_skew))
        if hasattr(self,'hsv_kurt'):
            featList.extend(['hsv_kurt_H','hsv_kurt_S','hsv_kurt_V'])
            featClass.extend(['rgb_hsv']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.hsv_kurt
            else:
                featStack = np.concatenate((featStack,self.hsv_kurt))
        if hasattr(self,'hsv_sum'):
            featList.extend(['hsv_sum_H','hsv_sum_S','hsv_sum_V'])
            featClass.extend(['rgb_hsv']*3)
            featSizeInvar.extend([False]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.hsv_sum
            else:
                featStack = np.concatenate((featStack,self.hsv_sum))
        if hasattr(self,'hsv_rng'):
            featList.extend(['hsv_rng_H','hsv_rng_S','hsv_rng_V'])
            featClass.extend(['rgb_hsv']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.band_rng
            else:
                featStack = np.concatenate((featStack,self.hsv_rng))
        if hasattr(self,'hsv_rngsig'):
            featList.extend(['hsv_rngsig_H','hsv_rngsig_S','hsv_rngsig_V'])
            featClass.extend(['rgb_hsv']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.hsv_rngsig
            else:
                featStack = np.concatenate((featStack,self.hsv_rngsig))
        if hasattr(self,'hsv_rngmean'):
            featList.extend(['hsv_rngmean_H','hsv_rngmean_S','hsv_rngmean_V'])
            featClass.extend(['rgb_hsv']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.hsv_rngmean
            else:
                featStack = np.concatenate((featStack,self.hsv_rngmean))
        if hasattr(self,'hsv_mode'):
            featList.extend(['hsv_mode_H','hsv_mode_S','hsv_mode_V'])
            featClass.extend(['rgb_hsv']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.hsv_mode
            else:
                featStack = np.concatenate((featStack,self.hsv_mode))
        if hasattr(self,'hsv_deciles'):
            featList.extend(['hsv_decile_H_1','hsv_decile_H_2',
                            'hsv_decile_H_3','hsv_decile_H_4',
                            'hsv_decile_H_5','hsv_decile_H_6',
                            'hsv_decile_H_7','hsv_decile_H_8',
                            'hsv_decile_H_9','hsv_decile_S_1',
                            'hsv_decile_S_2','hsv_decile_S_3',
                            'hsv_decile_S_4','hsv_decile_S_5',
                            'hsv_decile_S_6','hsv_decile_S_7',
                            'hsv_decile_S_8','hsv_decile_S_9',
                            'hsv_decile_V_1','hsv_decile_V_2',
                            'hsv_decile_V_3','hsv_decile_V_4',
                            'hsv_decile_V_5','hsv_decile_V_6',
                            'hsv_decile_V_7','hsv_decile_V_8',
                            'hsv_decile_V_9'])
            featClass.extend(['rgb_hsv']*27)
            featSizeInvar.extend([True]*27)
            featHeightInvar.extend([True]*27)
            featScale.extend([0]*27)
            if featStack.size==0:
                featStack = self.hsv_deciles
            else:
                featStack = np.concatenate((featStack,
                                            self.hsv_deciles.flatten('F')))
        if hasattr(self,'hsv_quartiles'):
            featList.extend(['hsv_quartile_H_1','hsv_quartile_H_3',
                            'hsv_quartile_S_1','hsv_quartile_S_3',
                            'hsv_quartile_V_1','hsv_quartile_V_3'])
            featClass.extend(['rgb_hsv']*6)
            featSizeInvar.extend([True]*6)
            featHeightInvar.extend([True]*6)
            featScale.extend([0]*6)
            if featStack.size==0:
                featStack = self.hsv_quartiles
            else:
                featStack = np.concatenate((featStack,
                                            self.hsv_quartiles.flatten('F')))
        if hasattr(self,'hsv_iqr'):
            featList.extend(['hsv_iqr_H','hsv_iqr_S','hsv_iqr_V'])
            featClass.extend(['rgb_hsv']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.hsv_iqr
            else:
                featStack = np.concatenate((featStack,self.hsv_iqr))
        if hasattr(self,'hsv_iqrsig'):
            featList.extend(['hsv_iqrsig_H','hsv_iqrsig_S','hsv_iqrsig_V'])
            featClass.extend(['rgb_hsv']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.hsv_iqrsig
            else:
                featStack = np.concatenate((featStack,self.hsv_iqrsig))
        if hasattr(self,'hsv_iqrmean'):
            featList.extend(['hsv_iqrmean_H','hsv_iqrmean_S','hsv_iqrmean_V'])
            featClass.extend(['rgb_hsv']*3)
            featSizeInvar.extend([True]*3)
            featHeightInvar.extend([True]*3)
            featScale.extend([0]*3)
            if featStack.size==0:
                featStack = self.hsv_iqrmean
            else:
                featStack = np.concatenate((featStack,self.hsv_iqrmean))
        if hasattr(self,'dsm_raw_max'):
            featList.extend(['dsm_raw_max_H'])
            featClass.extend(['dsm_raw'])
            featSizeInvar.extend([True])
            featHeightInvar.extend([False])
            featScale.extend([0])
            if featStack.size==0:
                featStack = self.dsm_raw_max
            else:
                featStack = np.concatenate((featStack,self.dsm_raw_max))
        if hasattr(self,'dsm_raw_min'):
            featList.extend(['dsm_raw_min_H'])
            featClass.extend(['dsm_raw'])
            featSizeInvar.extend([True])
            featHeightInvar.extend([False])
            featScale.extend([0])
            if featStack.size==0:
                featStack = self.dsm_raw_min
            else:
                featStack = np.concatenate((featStack,self.dsm_raw_min))
        if hasattr(self,'dsm_raw_mean'):
            featList.extend(['dsm_raw_mean_H'])
            featClass.extend(['dsm_raw'])
            featSizeInvar.extend([True])
            featHeightInvar.extend([False])
            featScale.extend([0])
            if featStack.size==0:
                featStack = self.dsm_raw_mean
            else:
                featStack = np.concatenate((featStack,self.dsm_raw_mean))
        if hasattr(self,'dsm_raw_std'):
            featList.extend(['dsm_raw_std_H'])
            featClass.extend(['dsm_raw'])
            featSizeInvar.extend([True])
            featHeightInvar.extend([True])
            featScale.extend([0])
            if featStack.size==0:
                featStack = self.dsm_raw_std
            else:
                featStack = np.concatenate((featStack,self.dsm_raw_std))
        if hasattr(self,'dsm_raw_median'):
            featList.extend(['dsm_raw_median_H'])
            featClass.extend(['dsm_raw'])
            featSizeInvar.extend([True])
            featHeightInvar.extend([False])
            featScale.extend([0])
            if featStack.size==0:
                featStack = self.dsm_raw_median
            else:
                featStack = np.concatenate((featStack,self.dsm_raw_median))
        if hasattr(self,'dsm_raw_cov'):
            featList.extend(['dsm_raw_cov_H'])
            featClass.extend(['dsm_raw'])
            featSizeInvar.extend([True])
            featHeightInvar.extend([False])
            featScale.extend([0])
            if featStack.size==0:
                featStack = self.dsm_raw_cov
            else:
                featStack = np.concatenate((featStack,self.dsm_raw_cov))
        if hasattr(self,'dsm_raw_skew'):
            featList.extend(['dsm_raw_skew_H'])
            featClass.extend(['dsm_raw'])
            featSizeInvar.extend([True])
            featHeightInvar.extend([True])
            featScale.extend([0])
            if featStack.size==0:
                featStack = self.dsm_raw_skew
            else:
                featStack = np.concatenate((featStack,self.dsm_raw_skew))
        if hasattr(self,'dsm_raw_kurt'):
            featList.extend(['dsm_raw_kurt_H'])
            featClass.extend(['dsm_raw'])
            featSizeInvar.extend([True])
            featHeightInvar.extend([True])
            featScale.extend([0])
            if featStack.size==0:
                featStack = self.dsm_raw_kurt
            else:
                featStack = np.concatenate((featStack,self.dsm_raw_kurt))
        if hasattr(self,'dsm_raw_sum'):
            featList.extend(['dsm_raw_sum_H'])
            featClass.extend(['dsm_raw'])
            featSizeInvar.extend([False])
            featHeightInvar.extend([False])
            featScale.extend([0])
            if featStack.size==0:
                featStack = self.dsm_raw_sum
            else:
                featStack = np.concatenate((featStack,self.dsm_raw_sum))
        if hasattr(self,'dsm_raw_rng'):
            featList.extend(['dsm_raw_rng_H'])
            featClass.extend(['dsm_raw'])
            featSizeInvar.extend([True])
            featHeightInvar.extend([True])
            featScale.extend([0])
            if featStack.size==0:
                featStack = self.dsm_raw_rng
            else:
                featStack = np.concatenate((featStack,self.dsm_raw_rng))
        if hasattr(self,'dsm_raw_rngsig'):
            featList.extend(['dsm_raw_rngsig_H'])
            featClass.extend(['dsm_raw'])
            featSizeInvar.extend([True])
            featHeightInvar.extend([True])
            featScale.extend([0])
            if featStack.size==0:
                featStack = self.dsm_raw_rngsig
            else:
                featStack = np.concatenate((featStack,self.dsm_raw_rngsig))
        if hasattr(self,'dsm_raw_rngmean'):
            featList.extend(['dsm_raw_rngmean_H'])
            featClass.extend(['dsm_raw'])
            featSizeInvar.extend([True])
            featHeightInvar.extend([False])
            featScale.extend([0])
            if featStack.size==0:
                featStack = self.dsm_raw_rngmean
            else:
                featStack = np.concatenate((featStack,self.dsm_raw_rngmean))
        if hasattr(self,'dsm_raw_mode'):
            featList.extend(['dsm_raw_mode_H'])
            featClass.extend(['dsm_raw'])
            featSizeInvar.extend([True])
            featHeightInvar.extend([False])
            featScale.extend([0])
            if featStack.size==0:
                featStack = self.dsm_raw_mode
            else:
                featStack = np.concatenate((featStack,self.dsm_raw_mode))
        if hasattr(self,'dsm_raw_deciles'):
            featList.extend(['dsm_raw_decile_H_1','dsm_raw_decile_H_2',
                            'dsm_raw_decile_H_3','dsm_raw_decile_H_4',
                            'dsm_raw_decile_H_5','dsm_raw_decile_H_6',
                            'dsm_raw_decile_H_7','dsm_raw_decile_H_8',
                            'dsm_raw_decile_H_9'])
            featClass.extend(['dsm_raw']*9)
            featSizeInvar.extend([True]*9)
            featHeightInvar.extend([False]*9)
            featScale.extend([0]*9)
            if featStack.size==0:
                featStack = self.dsm_raw_deciles.flatten('F')
            else:
                featStack = np.concatenate((featStack,
                                            self.dsm_raw_deciles.flatten('F')))
        if hasattr(self,'dsm_raw_quartiles'):
            featList.extend(['dsm_raw_quartile_H_1','dsm_raw_quartile_H_3'])
            featClass.extend(['dsm_raw']*2)
            featSizeInvar.extend([True]*2)
            featHeightInvar.extend([False]*2)
            featScale.extend([0]*2)
            if featStack.size==0:
                featStack = self.dsm_raw_quartiles.flatten('F')
            else:
                featStack = np.concatenate((featStack,
                                            self.dsm_raw_quartiles.flatten('F')))
        if hasattr(self,'dsm_raw_iqr'):
            featList.extend(['dsm_raw_iqr_H'])
            featClass.extend(['dsm_raw'])
            featSizeInvar.extend([True])
            featHeightInvar.extend([True])
            featScale.extend([0])
            if featStack.size==0:
                featStack = self.dsm_raw_iqr
            else:
                featStack = np.concatenate((featStack,self.dsm_raw_iqr))
        if hasattr(self,'dsm_raw_iqrsig'):
            featList.extend(['dsm_raw_iqrsig_H'])
            featClass.extend(['dsm_raw'])
            featSizeInvar.extend([True])
            featHeightInvar.extend([True])
            featScale.extend([0])
            if featStack.size==0:
                featStack = self.dsm_raw_iqrsig
            else:
                featStack = np.concatenate((featStack,self.dsm_raw_iqrsig))
        if hasattr(self,'dsm_raw_iqrmean'):
            featList.extend(['dsm_raw_iqrmean_H'])
            featClass.extend(['dsm_raw'])
            featSizeInvar.extend([True])
            featHeightInvar.extend([True])
            featScale.extend([0])
            if featStack.size==0:
                featStack = self.dsm_raw_iqrmean
            else:
                featStack = np.concatenate((featStack,self.dsm_raw_iqrmean))
        if hasattr(self,'dsm_raw_mad'):
            featList.extend(['dsm_raw_mad_H'])
            featClass.extend(['dsm_raw'])
            featSizeInvar.extend([True])
            featHeightInvar.extend([True])
            featScale.extend([0])
            if featStack.size==0:
                featStack = self.dsm_raw_mad
            else:
                featStack = np.concatenate((featStack,self.dsm_raw_mad))
        if hasattr(self,'dsm_raw_maxmed'):
            featList.extend(['dsm_raw_maxmed_H'])
            featClass.extend(['dsm_raw'])
            featSizeInvar.extend([True])
            featHeightInvar.extend([True])
            featScale.extend([0])
            if featStack.size==0:
                featStack = self.dsm_raw_maxmed
            else:
                featStack = np.concatenate((featStack,self.dsm_raw_maxmed))
        if hasattr(self,'dsm_raw_minmed'):
            featList.extend(['dsm_raw_minmed_H'])
            featClass.extend(['dsm_raw'])
            featSizeInvar.extend([True])
            featHeightInvar.extend([True])
            featScale.extend([0])
            if featStack.size==0:
                featStack = self.dsm_raw_minmed
            else:
                featStack = np.concatenate((featStack,self.dsm_raw_minmed))
        if hasattr(self,'dsm_raw_summed'):
            featList.extend(['dsm_raw_summed_H'])
            featClass.extend(['dsm_raw'])
            featSizeInvar.extend([True])
            featHeightInvar.extend([True])
            featScale.extend([0])
            if featStack.size==0:
                featStack = self.dsm_raw_summed
            else:
                featStack = np.concatenate((featStack,self.dsm_raw_summed))
        if hasattr(self,'dsm_raw_decilesmed'):
            featList.extend(['dsm_raw_decilemed_H_1','dsm_raw_decilemed_H_2',
                            'dsm_raw_decilemed_H_3','dsm_raw_decilemed_H_4',
                            'dsm_raw_decilemed_H_5','dsm_raw_decilemed_H_6',
                            'dsm_raw_decilemed_H_7','dsm_raw_decilemed_H_8',
                            'dsm_raw_decilemed_H_9'])
            featClass.extend(['dsm_raw']*9)
            featSizeInvar.extend([True]*9)
            featHeightInvar.extend([True]*9)
            featScale.extend([0]*9)
            if featStack.size==0:
                featStack = self.dsm_raw_decilesmed.flatten('F')
            else:
                featStack = np.concatenate((featStack,
                                            self.dsm_raw_decilesmed.flatten('F')))
        if hasattr(self,'dsm_raw_quartilesmed'):
            featList.extend(['dsm_raw_quartilemed_H_1','dsm_raw_quartilemed_H_3'])
            featClass.extend(['dsm_raw']*2)
            featSizeInvar.extend([True]*2)
            featHeightInvar.extend([True]*2)
            featScale.extend([0]*2)
            if featStack.size==0:
                featStack = self.dsm_raw_quartilesmed.flatten('F')
            else:
                featStack = np.concatenate((featStack,
                                            self.dsm_raw_quartilesmed.flatten('F')))
        if hasattr(self,'glcm_dsm_vals'):
            for dsm_glcm_d in self.glcm_dsm_dist:
                glcm_list = ['glcm_dsm_asm_' + str(dsm_glcm_d),
                            'glcm_dsm_con_' + str(dsm_glcm_d),
                            'glcm_dsm_cor_' + str(dsm_glcm_d),
                            'glcm_dsm_var_' + str(dsm_glcm_d),
                            'glcm_dsm_idm_' + str(dsm_glcm_d),
                            'glcm_dsm_sumav_' + str(dsm_glcm_d),
                            'glcm_dsm_sumvar_' + str(dsm_glcm_d),
                            'glcm_dsm_sument_' + str(dsm_glcm_d),
                            'glcm_dsm_ent_' + str(dsm_glcm_d),
                            'glcm_dsm_difvar_' + str(dsm_glcm_d),
                            'glcm_dsm_difent_' + str(dsm_glcm_d),
                            'glcm_dsm_infcor1_' + str(dsm_glcm_d),
                            'glcm_dsm_infcor2_' + str(dsm_glcm_d),
                            'glcm_dsm_asm_rng_' + str(dsm_glcm_d),
                            'glcm_dsm_con_rng_' + str(dsm_glcm_d),
                            'glcm_dsm_cor_rng_' + str(dsm_glcm_d),
                            'glcm_dsm_var_rng_' + str(dsm_glcm_d),
                            'glcm_dsm_idm_rng_' + str(dsm_glcm_d),
                            'glcm_dsm_sumav_rng_' + str(dsm_glcm_d),
                            'glcm_dsm_sumvar_rng_' + str(dsm_glcm_d),
                            'glcm_dsm_sument_rng_' + str(dsm_glcm_d),
                            'glcm_dsm_ent_rng_' + str(dsm_glcm_d),
                            'glcm_dsm_difvar_rng_' + str(dsm_glcm_d),
                            'glcm_dsm_difent_rng_' + str(dsm_glcm_d),
                            'glcm_dsm_infcor1_rng_' + str(dsm_glcm_d),
                            'glcm_dsm_infcor2_rng_' + str(dsm_glcm_d)]
                featList.extend(glcm_list)
                featClass.extend(['dsm_glcm']*26)
                featSizeInvar.extend([True]*26)
                featHeightInvar.extend([True]*26)
                featScale.extend([dsm_glcm_d]*26)
            if featStack.size==0:
                featStack = self.glcm_dsm_vals
            else:
                featStack = np.concatenate((featStack,self.glcm_dsm_vals))
        if hasattr(self,'acor_dsm_vals'):
            for dsm_acor_d in self.acor_dsm_dist:
                acor_list = ['acor_dsm_mean_' + str(dsm_acor_d),
                            'acor_dsm_rng_' + str(dsm_acor_d)]
                featList.extend(acor_list)
                featClass.extend(['dsm_acor']*2)
                featSizeInvar.extend([True]*2)
                featHeightInvar.extend([True]*2)
                featScale.extend([dsm_acor_d]*2)
            if featStack.size==0:
                featStack = self.acor_dsm_vals
            else:
                featStack = np.concatenate((featStack,self.acor_dsm_vals))
        if hasattr(self,'lbp_dsm_vals'):
            for dsm_lbp_d in self.lbp_dsm_dist:
                for ft_i in range(2+8*dsm_lbp_d):
                    featList.extend(
                        ['lbp_dsm_d_' + str(dsm_lbp_d) + '_feat_' + str(ft_i)]
                    )
                    featClass.extend(['dsm_lbp'])
                    featSizeInvar.extend([True])
                    featHeightInvar.extend([True])
                    featScale.extend([dsm_lbp_d])
            if featStack.size==0:
                featStack = self.lbp_dsm_vals
            else:
                featStack = np.concatenate((featStack,self.lbp_dsm_vals))
        if hasattr(self,'laws_dsm_feats'):
            laws_list = []
            filtbank = ['L5','E5','S5','R5','W5']
            for stat in ['mean','std']:
                for i in range(5):
                    for j in range(5):
                            if j < i or (i==0 and j ==0):
                                continue
                            else:
                                featList.append('laws_' + filtbank[i] + filtbank[j] +'_DSM_' + stat)
                                featClass.extend(['dsm_laws'])
                                featSizeInvar.extend([True])
                                featHeightInvar.extend([True])
                                featScale.extend([0])
            if featStack.size==0:
                featStack = self.laws_dsm_feats.flatten('F')
            else:
                featStack = np.concatenate((featStack,self.laws_dsm_feats.flatten('F')))
        self.featList = featList
        self.featStack = featStack
        self.featClass = featClass
        self.featSizeInvar = featSizeInvar
        self.featHeightInvar = featHeightInvar
        self.featScale = featScale
    
    def runFeaturePipeline(self,thresh=0.5,glcm_steps=5,acor_steps=5,mode=False,HSV=False):
        '''
        Run the full pipeline for creating features from imagery
        thresh - sets threshold for brightest pixels
        glcm_steps - sets the highest offset for GLCM imagery (works from 1 to this)
        acor_steps - same as glcm_steps but for autocorrelation
        mode - decide if mode is used in certain features
        HSV - decide whether to compute HSV features
        '''
        self.createRGBBandFeats(mode)
        self.createRGBThreshFeats(0.5,mode)
        self.createRGBLawsFeats()
        self.createSpecIndices()
        self.createRGBIndFeats(mode)
        if HSV:
            self.createHSVFeats(mode)
        if hasattr(self,'dsm_pixels'):
            self.createDSMRawFeats(mode)
            self.createDSMLawsFeats()
        if hasattr(self,'ms_pixels'):
            self.createMSLawsFeats()
            self.createMSIndFeats(mode)
            self.createMSBandFeats(mode)
        for i in range(glcm_steps):
            self.createRGBGLCMfeats(i+1)
            if hasattr(self,'ms_pixels'):
                self.createMSGLCMfeats(i+1)
            if hasattr(self,'dsm_pixels'):
                self.createDSMGLCMfeats(i+1)
        for i in range(acor_steps):
            self.createRGBautoCorFeats(i+1)
            if hasattr(self,'ms_pixels'):
                self.createMSautoCorFeats(i+1)
            if hasattr(self,'dsm_pixels'):
                self.createDSMautoCorFeats(i+1)
        for i in range(3):
            self.createRGBLBPFeats(i+1)
            if hasattr(self,'ms_pixels'):
                self.createMSLBPFeats(i+1)
            if hasattr(self,'dsm_pixels'):
                self.createDSMLBPFeats(i+1)
        self.stackFeats()
