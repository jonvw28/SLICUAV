Included in this directory are all the scripts and steps necessary to recreate
the main results figures for the manuscript for SLIC-UAV. This doesn't include
some figures as they are simply images of the data or schematics.

Below is an outline of what is within each folder.

1_Assessing_models contains the work to build and assess models on both crowns
  and superpixels for these crowns. See the README within the folder to see
  the steps you need to complete in which order to get to the final figures from
  our manuscript.

2_Testing_imagery_and_features contains the work needed to recreate the analysis
  looking at the contribution of imagery and feature choice on the accuracy of
  our SLIC-UAV approach combined with SVM modelling, again see the README there
  for the steps to follow

3_Landscape_mapping contains the work necessary to build the model we use for
  mapping species occurrences across our study site, using SLIC-UAV. This
  produces the labelled superpixels, which are then saved as a shapefile as 
  detailed in README for the data folder (which also explains how to load
  the data for plotting in a GIS viewer). See the README in this folder for more

Miscellaneous includes the data and/or code parts needed to generate the
illustrative figures for SLIC superpixels and features in our manuscript. These
generally involve artificial or simplified data to illustrate concepts, but are
included for completeness.

NB: in some folders the (now renamed) trees python library includes references
to using mSLIC for clustering. This was a custom variant of SLIC provided
kindly by Philip Sellars. We chose not to use this approacha and have since
removed this from SLIC-UAV. The library has had a single line removed to ensure
that the library loads now we have removed this code (it not being ours to share)
Within the TreeMap method you can still try to set this option but it won't work,
this is because we have removed it, using the smallest change possible without
affected how our pipeline works. We removed the line loading this library within
TreeMap, but did not change the code within the definition of TreeMap.

Author: Jonathan Williams
email: jonvw28@gamil.com

