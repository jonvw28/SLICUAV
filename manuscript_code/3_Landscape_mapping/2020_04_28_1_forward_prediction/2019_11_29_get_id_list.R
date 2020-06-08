# subtract a list of valid grid plot ids

setwd("E:/SLICUAV_manuscript_code/3_Landscape_mapping/2020_04_28_1_forward_prediction")


files <- list.files('E:/SLICUAV_manuscript_data/5_Landscape_superpixels/features/')
feat_files <- grep('2019_10_14_segment_features_id*',files)
files <- files[feat_files]
files <- substring(files,first =32,last=nchar(files)-4)

write.csv(files,'2019_11_29_grid_ids.csv')