setwd("E:/SLICUAV_manuscript_code/3_Landscape_mapping/2020_04_28_1_forward_prediction")

library(raster)
library(dplyr)

id_list <- read.csv("2019_11_29_grid_ids.csv")
ticker <- 1

for(id in id_list[,2]){
        sfile <- raster::shapefile(paste('E:/SLICUAV_manuscript_data/5_Landscape_superpixels/',id,'_SLIC_5000.shp',sep=''))
        # add areas
        sfile$area <- area(sfile)
        
        # Load predictions
        data <- read.csv(paste('E:/SLICUAV_manuscript_data/5_Landscape_superpixels/predictions/2020_04_28_preds_id_',id,'.csv',sep=''))
        data <- data[,-1] %>%
                dplyr::mutate(block = as.character(id))
        names(data)[1] <- 'ID'
        
        harap_data <- read.csv(paste('E:/SLICUAV_manuscript_data/5_Landscape_superpixels/Harapan_status/2020_01_27_harapan_status_',id,'.csv',sep=''),header=F)
        harap_data <- harap_data[,-1]
        names(harap_data) <- c('ID','Harapan')
        
        defor_data <- read.csv(paste('E:/SLICUAV_manuscript_data/5_Landscape_superpixels/Deforest_status/2020_04_17_harapan_non_deforest_status_',id,'.csv',sep=''),header=F)
        defor_data <- defor_data[,-1]
        names(defor_data) <- c('ID','HF_no_def')
        
        s_data <- merge(data,harap_data,by='ID') %>%
                merge(defor_data,by='ID')%>%
                dplyr::mutate(no_defor = ifelse(HF_no_def == Harapan,1,0))
        
        newshp <- merge(sfile,s_data,by='ID')

        shapefile(newshp,paste('E:/SLICUAV_manuscript_data/6_Landscape_predictions/grid_cells/',id,'_preds_2020_04_23',sep=''))   
        rm(sfile,data,harap_data,defor_data,newshp,s_data)
        print(ticker)
        ticker <- ticker + 1
        }
