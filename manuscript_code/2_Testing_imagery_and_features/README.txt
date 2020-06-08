The pipeline below will allow you to recreate the analysis and included figures
for the choices of imagery and features.

First you will need to multiplex the various combinations of features and
imagery included. This is split, owing to this work spinning off from a subtly
different comparsion. Therefore the models using all imagery and all features
are from a different folder than the rest. We originally looked at the effect
of correction of MS data and its impact on model accuracies, but as noted in our
work, we felt this analysis was less meaningful given issues with this process.
We then focused on the choice of imagery and features, owing to the different
costs of sensors and processing effort, but given the number of models to be
fitted, we used the existing models for the one case we had already completed -
as much for convenience that the remaining combinations totalled 20 which
matched perfectly to available cores on a workstation in the group at the time
of running this analysis. However, the approach is directly comparable.

In this analysis we used an existing training and test split (rather than much
more computationally costly 10-fold cross-validation). This file used a random
split created in a previous model, for which we include the necessary code to
recreate this split. Simply run the Rscript in 0a_create_split

Next you need to filter the tree crowns to ensure they sit outside of the areas
for which the multispectral data had issues. For this you will need to used
a GIS programme. Load the crowns from the data folder 2_Crowns into this. Then
load the file of the MS issues with 50m buffer from going to 1_Orthomosaics /
2019_03_07_Harapan_basecamp_sequoia_2017 and then loading
2020_02_27_MS_data_good_masked_by_correction_issues_UTM_50m_buffer_dissolved
for which you can then select the crowns polygons which are within this polygon,
being the area mapped in both surveys which was outside of the buffer around
MS data issues. Saving these crowns in a shapefile will give you the file in
2_Crowns/Outide_MS_issues as well as saving the data from this to CSV. Running 
the script in 0b_get_tree_ids will generate the csv of tree tagstrings for the
trees outside of the issues

Next you will need to run the model using all imagery and features (again on its
own since it doesn't require the same filtering step). Navigate to
2020_03_06_3_all_included_model and run the R script which will fit models for
the 3 sets of labels for all imagery and feautres, for the crowns without MS
processing issues, with the split as generated.

Next you can navigate to 2020_05_01_2_multiplexing_imagery_features where you
can then run each of the 20 R scripts to run the SVM models for all remaining
combinations of features and imagery.

Once you've run all 20 scripts there, and the previous folder, you will want to
mve to 2020_05_04_2_create_plots, the folder with scripts to recreate our main
figure as well as supplemental figure.

First you need to run 2020_05_04_compile_stats to collate the outputs from all
of the various models into a single dataframe which is then saved as a csv. Then
run 2020_05_04_plot to generate and save the plots as used in our manuscript.

This should all combine to give you the figures we use in our work.

Author: Jonathan Williams
email: jonvw28@gamil.com