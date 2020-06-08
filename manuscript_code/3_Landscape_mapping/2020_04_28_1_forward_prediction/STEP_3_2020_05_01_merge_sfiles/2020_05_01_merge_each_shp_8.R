setwd("E:/SLICUAV_manuscript_code/3_Landscape_mapping/2020_04_28_1_forward_prediction/STEP_3_2020_05_01_merge_sfiles")

library(raster)

id_list <- read.csv('../2019_11_29_grid_ids.csv')

id <- id_list[351,2]
final_s <- raster::shapefile(paste('E:/SLICUAV_manuscript_data/6_Landscape_predictions/grid_cells/',id,'_preds_2020_04_23.shp',sep=''))

ticker <- 1

for(i in 352:400){
        id <- id_list[i,2]
        sfile <- raster::shapefile(paste('E:/SLICUAV_manuscript_data/6_Landscape_predictions/grid_cells/',id,'_preds_2020_04_23.shp',sep=''))
		final_s <- raster::bind(final_s,sfile)
	ticker <- ticker + 1
	print(ticker)
}

shapefile(final_s,'E:/SLICUAV_manuscript_data/6_Landscape_predictions/merged_chunks/full_preds_2020_04_23_8')
