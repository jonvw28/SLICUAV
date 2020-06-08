# -*- coding: utf-8 -*-
"""

@author: Jonathan Williams
jonvw28@gmail.com

structure to enable SLIC-UAV mapping of trees. This is a data structure which
handles imagery, superpixel generation and clipping of imagery by reference
shapefile data

@author: Jonathan Williams
jonvw28@gmail.com
21/05/2020
"""


# modules
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import rasterio as rio
import geopandas as gpd
from osgeo import gdal, ogr, osr

# methods
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from rasterio.mask import mask
from shapely.geometry import mapping

# Create class definition
class TreeMap:
    """
    TreeMap
        .imagePath - path to image, including filename
        .__fileType - what kind of file is being used - JPEG or TIFF only
        .image - numpy array of image
        .clipped_image - numpy array of image when clipped by a shapefile
        .imgDims - array dimensions of .image
        .__segMethod - string of segmentation approach to use
        .__segParams - tuple of parameters for a given segmenatation approach
        .segmentation - mask of segments for image
        .__crs - crs of tiff
        .__transformation - affine transformation of tiff
        
    methods
        setPath - set image path
        getPath - get image path
        loadImage - load image into a numpy array
        getImage - get a loaded array
        subsampleImage - subsample image by a specified factor
        getDims - get dimensions of the array
        plotImage - plot the array
        getBand - get a single band of the array
        plotBand - plot a single band of the array
        setSegOpt - set segmentation method
        getSegOpt - check segmentation method
        setSegParam - set parameters for segmentation
        getSegParam - get parameters for segmentation
        applySeg - apply the segmentation
        getSeg - get the segmentation mask
        plotSeg - get a plot of image overlaid with segmentation
        plotSegBand - get plot of specific band of image with segmentation overlay
        clipImage - clip tiff image by a shapefile
        clipImageByFeature - clip tiff image by a shapefile bfiltered based on feature

    @author: Jonathan Williams
    jonvw28@gmail.com
    21/05/2020
    """
    
    # just initialise the path to the image and methods to alter it
    def __init__(self,path='.'):
        self.imagePath = path
        self.__segMethod = 'SLIC'
        self.__segParams = (250,10,1)
    
    def setPath(self,path):
        self.imagePath = path
        if path[-3:] == 'jpg' or path[-3:] == 'JPG':
            self.__fileType = 'JPEG'
        elif path[-3:] == 'tif' or path[-3:] == 'TIF':
            self.__fileType = 'TIFF'
        else:
            raise ValueError('imagePath must be for a JPEG or TIFF file')
    
    def getPath(self):
        return self.imagePath
    
    # read and load the image with options to show and to access
    def loadImage(self):
        '''
        Load an image, either JPEG of TIFF
        '''
        if self.__fileType == 'JPEG':
            self.image = img_as_float(io.imread(self.imagePath,
                                                plugin='matplotlib'))
            self.imgDims = self.image.shape
        elif self.__fileType == 'TIFF':
            # Connect and extract data
            gtif = rio.open(self.imagePath)
            gimg = gtif.read()
            
            # Reorder correctly as first dimension is bands
            gimg = np.swapaxes(gimg,0,2)
            gimg = np.swapaxes(gimg,0,1)
            
            # now add to struct
            self.__crs = gtif.crs
            self.__transform = gtif.transform
            self.image = gimg
            self.imgDims = self.image.shape
            gtif.close()
        else:
            raise ValueError('__fileType has been manually edited to an \
                             invalid value, it is recommended not to alter \
                             this')
    
    def getImage(self):
        '''
        Access the image as an array
        '''
        if hasattr(self,'image'):
            return self.image
        else:
            raise AttributeError('image not yet loaded, use .loadImage()')
    
    def subsampleImage(self,factor=5):
        '''
        Subsample image by a factor as set
        '''
        if hasattr(self,'image'):
            if(int(factor)==factor and factor>0):
                self.image = self.image[::factor,::factor,:]
            else:
                raise ValueError('factor must be a postive integer')
        else:
            raise AttributeError('image not yet loaded, use .loadImage()')
    
    def plotImage(self,band1=0,band2=1,band3=2):
        '''
        Print image to console
        '''
        if hasattr(self,'image'):
            plt.figure()
            plt.imshow(self.image[:,:,[band1,band2,band3]])
        else:
            raise AttributeError('image not yet loaded, use .loadImage()')
    
    def getBand(self,band=0):
        '''
        Return array of single band of image, by index 'band'
        '''
        if hasattr(self,'image'):
            return self.image[:,:,band]
        else:
            raise AttributeError('image not yet loaded, use .loadImage()')
    
    def plotBand(self,band=0):
        '''
        Print monochrome image to terminal of band, indexed by 'band'
        '''
        if hasattr(self,'image'):
            plt.figure()
            plt.imshow(self.image[:,:,band]) 
        else:
            raise AttributeError('image not yet loaded, use .loadImage()')
    
    # Access dimensions
    def getDims(self):
        '''
        Find dimensions of image array
        '''
        if hasattr(self,'image'):
            return(self.imgDims)
        else:
            raise AttributeError('image not yet loaded, use .loadImage()')
    
    # Set and get Segmentation type
    def setSegOpt(self,option):
        '''
        Set segmentation option from:
        SLIC, felzenszwalb, quickshift and watershed from scikit image
        '''
        if(option in ['felzenszwalb','SLIC','quickshift','watershed']):
            self.__segMethod = option
            if(option == 'felzenszwalb'):
                self.__segParams = (200,0.5,50)
            elif(option == 'SLIC'):
                self.__segParams = (250,10,1)
            elif(option == 'quickshift'):
                self.__segParams = (3,6,0.5)
            else:
                self.__segParams = (250,0.001)
        else:
             raise ValueError('Only valid segmentation methods are\
                              felzenszwalb, SLIC, quickshift and \
                              watershed')
    
    def getSegOpt(self):
        '''
        Report the Segmentation option set
        '''
        return(self.__segMethod)
    
    def setSegParam(self,param):
        '''
        Set the segmentation option paramters (check scikit-image for each method)
        '''
        self.__segParams = param
    
    def getSegParam(self):
        '''
        Report the Segmentation parameter
        '''
        return(self.__segParams)
    
    def applySeg(self):
        '''
        Apply the set segmentation method to the loaded image
        '''
        if hasattr(self,'image'):
            if(self.__segMethod == 'felzenszwalb'):
                self.segmentation = felzenszwalb(
                        self.image[:,:,0:3], scale=self.__segParams[0],
                        sigma=self.__segParams[1],
                        min_size=self.__segParams[2])
            elif(self.__segMethod == 'SLIC'):
                self.segmentation = slic(
                        self.image[:,:,0:3],
                        n_segments=self.__segParams[0],
                        compactness=self.__segParams[1], 
                        sigma=self.__segParams[2])
            elif(self.__segMethod == 'mSLIC'):
                tmp_mslic = mSLIC.MSLICProcessor(self.image[:,:,0:3],
                        k = self.__segParams[0],
                        m = self.__segParams[1],
                        max_iter = self.__segParams[2])
                self.segmentation = tmp_mslic.mainWorkFlow()
            elif(self.__segMethod == 'quickshift'):
                self.segmentation = quickshift(
                        self.image[:,:,0:3],
                        kernel_size=self.__segParams[0],
                        max_dist=self.__segParams[1], 
                        ratio=self.__segParams[2])
            elif(self.__segMethod == 'watershed'):
                gradient = sobel(rgb2gray(self.image[:,:,0:3]))
                self.segmentation = watershed(
                        gradient, markers=self.__segParams[0], 
                        compactness=self.__segParams[1])
            else:
                raise ValueError('__segMethod has been set to an illegal \
                                 value, only use setSegOpt to choose this')
            # index from 1 not 0 so clipping doesn't cause issues with zeros later
            self.segmentation +=1
        else:
            raise AttributeError('image not yet loaded, use .loadImage()')
    
    def getSeg(self):
        '''
        Access the array of segemtn labels
        '''
        if hasattr(self,'segmentation'):
            return self.segmentation
        else:
            raise AttributeError('segmention not yet created, use .applySeg()')
    
    def plotSeg(self,band1=0,band2=1,band3=2):
        '''
        plot the segmentation, with options to choose bands for RGB
        '''
        if hasattr(self,'segmentation'):
            plt.figure()
            plt.imshow(mark_boundaries(
                    self.image[:,:,[band1,band2,band3]],
                    self.segmentation))
        else:
            raise AttributeError('segmention not yet created, use .applySeg()')
    
    def plotSegBand(self,band=0):
        '''
        Plot the segmentation over a single band of the imagery
        '''
        if hasattr(self,'segmentation'):
            plt.figure()
            plt.imshow(mark_boundaries(
                    self.image[:,:,band],
                    self.segmentation))
        else:
            raise AttributeError('segmention not yet created, use .applySeg()')
    
    def clipImage(self,shapeFile,save=False):
        '''
        clip the image by the shapefile gicen (given the path to the filename)
        save causes the clipped image to be saved
        '''
        if shapeFile[-3:] == 'shp' or shapeFile[-3:] == 'SHP':
            if not hasattr(self,'image'):
                self.loadImage()
            shp_clip = gpd.read_file(shapeFile)
            shps_clip = []
            for i in range(shp_clip.shape[0]):
                tmp_gjson = mapping(shp_clip['geometry'][i])
                shps_clip.append(tmp_gjson)
                del tmp_gjson
            with rio.open(self.imagePath) as gtif:
                clip_gtif, clip_affine = mask(gtif,shps_clip,crop=False,all_touched=True)
            clip_img = np.swapaxes(clip_gtif,0,2)
            clip_img = np.swapaxes(clip_img,0,1)
            self.clipped_image = clip_img
            if save == True:
                image_name = self.imagePath[:-4]
                shp_name = shapeFile[:-4]
                tiff_out = image_name + '_clipped_by_' + shp_name + '.tif'
                new_tiff = rio.open(tiff_out,'w',driver='GTiff',
                        height = clip_img.shape[0], 
                        width = clip_img.shape[1],
                        count = clip_img.shape[2], 
                        dtype = clip_img.dtype,
                        crs = self.__crs,
                        transform = self.__transform)
                for i in range(clip_img.shape[2]):
                    new_tiff.write(clip_img[:,:,i],i+1)
                new_tiff.close()
        else:
            raise ValueError('shapeFile must be valid path to a shapefile')
    
    def clipImageByFeature(self,shapeFile,feature,value,save=False):
        '''
        clip the image by the shapefile given (given the path to the filename)
        clipping will only be by the polygon slected by attribute 'feature'
        having value 'value'
        save causes the clipped image to be saved
        '''
        if shapeFile[-3:] == 'shp' or shapeFile[-3:] == 'SHP':
            if not hasattr(self,'image'):
                self.loadImage()
            shp_clip = gpd.read_file(shapeFile)
            sub_clip = np.where(shp_clip[feature]==value)[0]
            shps_clip = []
            for i in sub_clip:
                tmp_gjson = mapping(shp_clip['geometry'][i])
                shps_clip.append(tmp_gjson)
                del tmp_gjson
            with rio.open(self.imagePath) as gtif:
                clip_gtif, clip_affine = mask(gtif,shps_clip,crop=False,all_touched=True)
            clip_img = np.swapaxes(clip_gtif,0,2)
            clip_img = np.swapaxes(clip_img,0,1)
            self.clipped_image = clip_img
            if save == True:
                image_name = self.imagePath[:-4]
                shp_name = shapeFile[:-4]
                tiff_out = image_name + '_clipped_by_' + shp_name +\
                        '_subset_by_' + feature + '_being_' + str(value) + '.tif'
                new_tiff = rio.open(tiff_out,'w',driver='GTiff',
                        height = clip_img.shape[0], 
                        width = clip_img.shape[1],
                        count = clip_img.shape[2], 
                        dtype = clip_img.dtype,
                        crs = self.__crs,
                        transform = self.__transform)
                for i in range(clip_img.shape[2]):
                    new_tiff.write(clip_img[:,:,i],i+1)
                new_tiff.close()
        else:
            raise ValueError('shapeFile must be valid path to a shapefile')
    
    def saveSeg(self,saveShape=True,filestem=None):
        '''
        Save segmentation once completed
        saveShape saves a shapefile version of the segmentation
        filestem sets a choice of fielname for output, but one is autogenerated
        if not given
        '''
        if self.__fileType == 'TIFF':
            if hasattr(self,'segmentation'):
                if filestem is None:
                    filestem = self.imagePath[:-4] + '_segmented_by_' +\
                        self.__segMethod
                trans = self.__transform
                outtrans = (trans[2],trans[0],trans[1],trans[5],trans[3],
                            trans[4])
                out_rst = gdal.GetDriverByName('GTiff').Create(
                            filestem + '.tif',self.imgDims[1],self.imgDims[0],
                            1,gdal.GDT_Int32)
                out_rst.SetGeoTransform(outtrans)
                srs = osr.SpatialReference()
                srs.ImportFromEPSG(self.__crs.to_epsg())
                out_rst.GetRasterBand(1).WriteArray(self.segmentation)
                out_rst.FlushCache()
                if saveShape == True:
                    drv = ogr.GetDriverByName("ESRI Shapefile")
                    dst_ds = drv.CreateDataSource(filestem + ".shp" )
                    dst_layer = dst_ds.CreateLayer(filestem, srs = None )
                    newField = ogr.FieldDefn('ID', ogr.OFTInteger)
                    dst_layer.CreateField(newField)
                    gdal.Polygonize(out_rst.GetRasterBand(1), None, dst_layer,
                                    0, [], callback=None )
                    dst_ds.Destroy()
                out_rst = None
                self.__segFileStem = filestem
            else:
                raise AttributeError('segmention not yet created,\
                                     use .applySeg()')
        else:
            raise ValueError('saveSeg only implemented for TIFF inputs')
    
    def saveLabelledSeg(self,shapeFile,feature,value,minProp=0.5):
        '''
        Save segmentation, after each feature is labelled by overlap with a
        polygon in the fiel 'shapeFile' (filename). This si specifed by the
        polygon with attribute 'feautre' having value 'value'.
        minProp sets the minimum proportion of each superpixel that must lie
        within the polygon. These superpixels will be labelled 1, with others 0
        '''
        if self.__fileType == 'TIFF':
            if shapeFile[-3:] == 'shp' or shapeFile[-3:] == 'SHP':
                if hasattr(self,'segmentation'):
                    if hasattr(self,'_TreeMap__segFileStem'):
                        shp_name = shapeFile[:-4]
                        filestem = self.__segFileStem +'_labelled'
                        shp_clip = gpd.read_file(shapeFile)
                        sub_clip = np.where(shp_clip[feature]==value)[0]
                        ths_shp = []
                        for i in sub_clip:
                            tmp_gjson = mapping(shp_clip['geometry'][i])
                            ths_shp.append(tmp_gjson)
                            del tmp_gjson
                        del shp_clip, sub_clip
                        with rio.open(self.__segFileStem + '.tif') as gtif:
                            seg_clip, clip_affine = mask(gtif,ths_shp,crop=False,all_touched=True)
                        del clip_affine, ths_shp
                        # get dictionary of value and counts for clipped data
                        clip_uni, clip_count = np.unique(seg_clip,return_counts=True)
                        clip_dict = dict(zip(clip_uni, clip_count))
                        del clip_uni, clip_count, seg_clip
                        # and for input of segments
                        seg_uni, seg_count = np.unique(self.segmentation,return_counts=True)
                        seg_dict = dict(zip(seg_uni, seg_count))
                        del seg_uni, seg_count
                        # find which are valid and track best option in case none are
                        valid_segs = []
                        curBest = []
                        curIn = 0 
                        curSz = 0
                        for seg_i in clip_dict:
                            if seg_i == 0:
                                continue
                            in_pol = clip_dict[seg_i]
                            tot = seg_dict[seg_i]
                            if in_pol/tot >= minProp:
                                valid_segs.append(seg_i)
                            # track the best fall back option
                            if in_pol > curIn:
                                curBest = [seg_i]
                                curIn = in_pol
                                curSz = tot
                            elif in_pol == curIn:
                                if tot < curSz:
                                    curBest = [seg_i]
                                    curIn = in_pol
                                    curSz = tot
                                elif tot == curSz:
                                    curBest.append(seg_i)
                            del in_pol, tot
                        del seg_i, clip_dict, seg_dict
                        # Add column for inclusion to shapefile and save it
                        seg_shps = gpd.read_file(self.__segFileStem + '.shp')
                        segs_temp = np.zeros(seg_shps.shape[0]).astype(int)
                        seg_shps['label'] = segs_temp
                        if not valid_segs:
                            for lab_i in curBest:
                                seg_shps.loc[seg_shps['ID']==lab_i,'label'] = 1
                            del lab_i
                        else:
                            for lab_i in valid_segs:
                                seg_shps.loc[seg_shps['ID']==lab_i,'label'] = 1
                            del lab_i
                        del segs_temp, valid_segs, curBest, curIn, curSz
                        seg_shps.to_file(filename = filestem, driver = "ESRI Shapefile")
                        del filestem, seg_shps
                    else:
                        raise AttributeError('Segmentation must first be saved by using saveSeg')
                else:
                    raise AttributeError('segmention not yet created,\
                                         use .applySeg()')
            else:
                raise ValueError('shapeFile must be valid path to a shapefile')
        else:
            raise ValueError('saveLabelledSeg only implemented for TIFF inputs')