This folder contains the scripts to build the model we used for landscape
mapping. This goes through generating superpixels, applying a model based upon
the labelled crowns and predicting the species or label of each superpixel.

First you will need to make sure the clipped OM imagery for all of the grid
cells is ready in the subfolders of data folder 3_Clipped_OMs (this should be
hosted already, but if the folders all_clips aren't full then follow the
instructions for the data folder to recreate the clips.

Next you will need to generate the superpixels for each grid cell. This can be 
done by navigating to 2019_09_03_1_generate_grid_superpixels and simply running
all of the python scripts in this folder. This will then generate and save all
of the superpixels into data folder 5_Landscape_superpixels. These are computed
based on a direct clip of the imagery to the grid cells.

Next you will need to compute features for each superpixel across the landscape.
To do this, navigate to 2019_10_23_1_compute_superpixel_features and run the
python scripts. This uses imagery clipped with a buffer for each grid cell to
ensure there are no edge effects in computing features (since grid cells are not
forced to align with pixel boundaries). This also uses a shapefile for the
footprint of where we have imagery for both surveys as stored in data folder 7

Then you need to construct the model based on the labelled data from our crowns
dataset. To do this navigate to 0202_04_23_2_train_model and run the R scripts.
These will fit several models, but the one we use in our manuscript is the SVM
model with all species label, which is created by 2020_04_23_2_glm_segs_all_fit.
The other scripts will recreate models with differing labels and model approach,
so it is up to you if you wish to use them. We include the pre-built models but
again you can edit and retrain as you see fit. This uses the features computed
in the code folder 1_Assessing_models, accessing the relevant data for the
superpixels for the crowns we have dara for. This step could be replaced in an
operational pipeline to extract superpixels within the gridding step for each
crown and building models on these. We only did this our way for convenience,
since we already processed superpixels for the crowns.

Next you will need to forward predict from the model onto all superpixels in the
grid framework. This is in the folder 2020_04_28_1_forward_prediction. First,
run 2019_11_29_get_id_list just to create a csv list of the grid cell ids to be
loaded and save loading the full shapefile.

Then you will need to run all r scripts starting with STEP_1 first, which
produces predictions for superpixels for each grid cell, and saves these in data
folder 5, in the subfolder predictions.

Then run the files starting STEP_2a and STEP_2b to generate CSV files for all
superpixels, indicating if they are within Harapan and outside of the forest
loss between years

Then you need to run the file STEP_2c to merge the predictions to the shapefiles
as well as the indicators for superpixel locations and areas of each and saves
all the output shapefiles for each grid cell in the subfolder grid_cells of
data folder 6

Next you should navigate to the subfolder STEP_3_2020_05_01_merge_sfiles which
will enable you stitch each grid cell into blocks of superpixels (in 13 parts
over 50 of fewer cells as R does not efficiently deal with RAm for this process)
Run the 13 files listed as merge_each_shp, which saves the merged shapefiles in
the subfolder merged_chunks in data folder 6.

Next you will want to run the script 2020_05_01_area_analysis to compute the
occurrence rates in terms of area and percentage of each species across the
landscape (as we report in our manuscript) which saves these data as CSVs in
this folder.

Finally, within this folder navigate to the subfolder beginning STEP_4 which
contains the code used to compute the confusion matrix and related statistics we
report in our manuscript. Simply run the R script in this folder to generate the
relevant reports.

At this point you will need to merge the chunked files in whichever way works
best for you. We merged them in a GIS programme (QGIS) since R used a lot of
RAM to do this. Once we merged the chunks we saved the output as a shapefile in
data folder 6 called all_preds_2020_04_23.

Finally to produce our heatmaps of species of interest you should move back to 
code folder 3 and then to 2020_04_04_raster_prevelance_map. Running the R script
will then produce CSVs for area and percentage of cover for each species for the
three sets of labels for each grid cell. You can then merge these results to the
grid shapefile (stored in this folder for convenience) in a GIS tool of your
choice to finally build the Grid shapefiles we include. The one suffixed
masked_by_def was created by filtering all grid cells to only include those
which overlap the area of Harapan for which there was no forest loss between
the surveys. We then plotted these heat maps as respectively: a) the combined
percentage of classes A. scholaris, E. malaccense and Other vegetation, b)
B. pentamera, c) M. gigantea.

Following this full pipeline will enable you to produce all of the figures we
include in our manuscript.

Author: Jonathan Williams
email: jonvw28@gamil.com
