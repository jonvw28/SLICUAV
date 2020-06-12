# SLIC-UAV

This is the python library SLIC-UAV which we have written as part of our accompanying manuscript (to which a link will be added in due course). This library is used for mapping tropical tree species from UAV imagery. Our pipeline centres around segmenting UAv-captures orthmosaic imagery into superpixels for which we are then able to compute a combination of features based on imagery bands and texture. We then use these features as input to modelling approaches to predict the species of each superpixel based upon a set of manually labelled crowns. This then allows extension of labels to all regions of the imagery. Full details on the method are outlined in our manuscript, but we include brief details on the steps here to enable you to run our pipeline.

## Loading Library

We recommend that you download a copy of this repository and make sure you know where it is saved. To load our library in python is then as simple as calling:

```
import slicuav
```

where you may need to be careful to ensure that you either in the parent directory for slicuav, or that you use appropriate paths to connect to the library (it is not designed to load as part of your standard python libraries, though you can move it to your libraries location to be able to always load if you know how to do this).

This will then load the definitions of two classes used for our pipeline as now detailed

### TreeMap

This is an object used to load in UAV imagery and apply segmentation to create superpixels. It supports JPEG and TIFF imagery, but to get a shapefile of the resulting superpixels you will need to supply a TIFF imagery which has been georeferenced (as can be extracted from UAV orthmosaics created by most common structure from motion appraoches). In our work we have focussed on using RGB UAV orthomosaic imagery for segmentation using the SLIC algorithm, though we include alternative options. This extract from our example pipeline script shows the steps this object normally uses:

```
spixel_obj = slicuav.TreeMap()
spixel_obj.setPath('example/data/Example_RGB.tif')
spixel_obj.loadImage()
spixel_obj.setSegOpt('SLIC')
spixel_obj.setSegParam((5000,10,1))
spixel_obj.applySeg()
spixel_obj.saveSeg(saveShape=True,filestem='example/output/SLIC_superpixels')
```

### SuperpixelFeatures

This is an object used to load in upto three types of imagery and create features based upon these (though now these must be TIFFs). You must load RGB imagery, and can additionally add DSM (from SfM) and multispectral imagery, where these exist. Only RGB data are essential. You will also need to supply a shapefile polygon to define the area for which the features are computed. The snippet below gives an example on how to do this, and the second snippet shows how simple running the pipeline for a given superpixel is.

```
import geopandas as gpd
from shapely.geometry import mapping
shps = gpd.read_file('shapefile.shp')
shape = []
shape.append(mapping(shps['geometry'][i]))
```

```
feat_obj = slicuav.SuperpixelFeatures(id,'Crown - Pulai','example/data/Example_RGB.tif',shape,dsm_path='example/data/Example_DSM.tif',ms_path='example/data/Example_MS.tif')
feat_obj.runFeaturePipeline(thresh=0.5,glcm_steps=3,acor_steps=3,mode=False,HSV=True)
features = feat_obj.featStack
```

NB: currently this approach is built Parrot Sequoia camera data and is designed for 4-band multispectral imagery. Importantly, for the mutlispectral indices to be valid you need the order of the bands to be Green, Red, Red Edge then Near Infrared. We would recommend including vegetation indices for multispectral data only if multispectral data is included and includes these four bands. Future work should make this part of the pipeline more flexible for other cameras allowing tailoer selection of indices as well as variable band number and order.


## Example Pipeline

The directory exmaple pipeline contains some example data and code to run the pipeline for a 0.25 ha patch of imagery, including a crown shapefile to allow you to see how to use this to label superpixels where they have overlap. See the script there for the steps with comments to explain further each step. At the end of the two processes above you can stack features for all superpixels and save this to a CSV. We then used this in various modelling appraoches in our work, as detailed in our manuscript.

## manuscript_code

This directory includes all the code we used to undertake the analysis and generate figures for our manuscript. The data needed to do this is far too large to place on github. At present we are getting these data archived formally - including all processing steps. In the meantime we have uploaded the input data that are needed for the pipelinem requiring running all steps to generate intermediate products to a less formal datastore. The link below will allow you to access our orthomosaic imagery and shapefiles for the crowns we mapped.

## Manuscript

The manuscript for this work is submitted and going through the process of review. In the meantime you can access our [preprint of the submission before review on arXiv](https://arxiv.org/abs/2006.06624)

## Data

Data for this project are available currently from [this link](https://drive.google.com/drive/folders/1RBAQb14kThDA_3TZ51-GJO0wnFdbZAj7?usp=sharing). Once they are fully archived with NERC we will include the DOI for the archived data. This delay is owing to the volume of imagery data combined with restrictions on internet speed from working from home.

## Versions

This library was built under Python 3.7.1. If you have any issues please do get in touch so I can investigate. We intend to further develop this pipeline and will archive versions of this library, or else link from here to any further developed approaches.

## Author

Jonathan Williams
jonvw28@gmail.com