To recreate the models and accuracies for cross-validation for crowns:

1) navigate to 2019_09_12_5_generate_crown_features and run
   2019_09_12_stack_spec_text_dsm_feats_all_classes.py which will compile the
   features for each crown mannually defined
        Uses the orthomosaics in data folder 1_Orthomosaics and crowns from
        data folder 2_Crowns. NB the orthomosaics are pre-clipped to save time
        in processing, done by a buffe around each crown. These clips are 
        already included in data folder 3_Clipped_OMs, but the code in there
        will allow you to reproduce them if you wish from the complete OMs in
        folder 1
        
2) Next navigate to 2020_04_21_1_crown_cv_tests, which has 3 folders for the 3
    methods (GLMNET is the package name for lasso regression). Within each
    each folder you can run the r script to run the 10-folds of cross-validation
    for that method, and you will get an output of the accuracies and 
    predictions (combining the results on the held out data for all folds)
    
    NB run GLMNET first, as it generates the fold list used in all work. This is
    in the other folders as hosted here, but has been copied directly from the
    GLMNET crown cross-validation (to be the same for all models)


To recreate the models and accuracies for cross-validation on superpixels for
our crowns:

1) navigate to 2019_09_12_6_generate_crown_superpixels, running each of the 3
   python sripts here will then prepare and save SLIC superpixels for each
   crown in our dataset, labelling the resulting shapefile polygons with 1
   if more than half overlaps the crown and 0 otherwise.
        Uses the same clipped orthomosaics as the crown processing above, and
        saves the superpixels as both a raster and shapefile in the data folder
        4_Crown_superpixels
        
2) next move to 2019_09_19_1_generate_superpixel_features. This folder contains
    the scripts to generate the features for all of the superpixels linked to a
    crown generated in step 1.You simply need to run the 3 python scripts here.
    Then after this run the R script, which will stitch the chunks together
    (allowing you to run the feature computations in chunks, though you can
    modify the scripts if you wish, since everything is done sequentially)
    
3) next you should navigate to 2020_04_21_1_superpixel_cv_tests which has 3 
    folders for the three methods we consider (as for cross-validation for
    crowns above). Each folder contains 10 R scripts which can be run, 1 for
    each fold. These need the fold_list to be created for step 2 of the crown
    modelling, as well as step two for the superpixel (in order to have data to
    work with). The scripts will also produce csvs of the timing of steps, but
    we don't include these as it is highly dependant on the machine you use.

Next you will want to produce the analysis we completed and present in the
manuscript, showing the accuracies of the models across cross-validation.

To do this navigate to 2020_04_23_1_analysing_cv_tests, and simply run the R
script in each sub-folder for the two problems.

Finally, after running these (which output csv files for the mean and standard
deviation of accuracies), run the R script at the top folder
1_Assessing_models and this will output the figures as we include in our
manuscript


Author: Jonathan Williams
email: jonvw28@gamil.com

