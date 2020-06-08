# compute area of each species across the landscape
setwd("E:/SLICUAV_manuscript_code/3_Landscape_mapping/2020_04_28_1_forward_prediction/STEP_3_2020_05_01_merge_sfiles")

library(raster)


# set up table
res_table <- as.data.frame(matrix(nrow = 7, ncol = 3,data=0))
names(res_table) <- c('all','sen','ran')
res_table <- cbind(class=c('BEL','MAC','MIS','PAL','RAN','PUL','SEN'),res_table)

tot_segs <- 0

for(i in 1:13){
        cur_shp <- raster::shapefile(paste('E:/SLICUAV_manuscript_data/6_Landscape_predictions/merged_chunks/full_preds_2020_04_23_',i,sep=''))
        
        # Filter to only include shps within harapan
        har_mask <- cur_shp$HF_no_def == 1
        har_mask[is.na(har_mask)] <- FALSE
        
        # count the number of segs for average seg size
        tot_segs <- tot_segs + sum(is.na(cur_shp$all)==FALSE)
        
        
        # pull out full model results
        all_mask <- cur_shp$all == 'BEL'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask & har_mask])
        res_table[1,'all'] <- res_table[1,'all'] + area
        
        all_mask <- cur_shp$all == 'MAC'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask & har_mask])
        res_table[2,'all'] <- res_table[2,'all'] + area
        
        all_mask <- cur_shp$all == 'MIS'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask & har_mask])
        res_table[3,'all'] <- res_table[3,'all'] + area
        
        all_mask <- cur_shp$all == 'PAL'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask & har_mask])
        res_table[4,'all'] <- res_table[4,'all'] + area
        
        all_mask <- cur_shp$all == 'RAN'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask & har_mask])
        res_table[5,'all'] <- res_table[5,'all'] + area
        
        all_mask <- cur_shp$all == 'PUL'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask & har_mask])
        res_table[6,'all'] <- res_table[6,'all'] + area
 
        all_mask <- cur_shp$all == 'SEN'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask & har_mask])
        res_table[7,'all'] <- res_table[7,'all'] + area
        
        # pull out sen model results
        all_mask <- cur_shp$sen == 'BEL'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask & har_mask])
        res_table[1,'sen'] <- res_table[1,'sen'] + area
        
        all_mask <- cur_shp$sen == 'MAC'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask & har_mask])
        res_table[2,'sen'] <- res_table[2,'sen'] + area
        
        all_mask <- cur_shp$sen == 'MIS'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask & har_mask])
        res_table[3,'sen'] <- res_table[3,'sen'] + area
        
        all_mask <- cur_shp$sen == 'PAL'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask & har_mask])
        res_table[4,'sen'] <- res_table[4,'sen'] + area
        
        all_mask <- cur_shp$sen == 'RAN'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask & har_mask])
        res_table[5,'sen'] <- res_table[5,'sen'] + area
        
        all_mask <- cur_shp$sen == 'PUL'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask & har_mask])
        res_table[6,'sen'] <- res_table[6,'sen'] + area
        
        
        # pull out ran model results
        all_mask <- cur_shp$ran == 'BEL'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask & har_mask])
        res_table[1,'ran'] <- res_table[1,'ran'] + area
        
        all_mask <- cur_shp$ran == 'MAC'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask & har_mask])
        res_table[2,'ran'] <- res_table[2,'ran'] + area
        
        all_mask <- cur_shp$ran == 'MIS'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask & har_mask])
        res_table[3,'ran'] <- res_table[3,'ran'] + area
        
        all_mask <- cur_shp$ran == 'PAL'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask & har_mask])
        res_table[4,'ran'] <- res_table[4,'ran'] + area
        
        all_mask <- cur_shp$ran == 'RAN'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask & har_mask])
        res_table[5,'ran'] <- res_table[5,'ran'] + area
        
        print(i)
        
}

write.csv(res_table,'2020_05_01_area_by_class.csv')

# get percentages
perc_table <- res_table
perc_table$all <- 100*res_table$all/sum(res_table$all)
perc_table$sen <- 100*res_table$sen/sum(res_table$sen)
perc_table$ran <- 100*res_table$ran/sum(res_table$ran)

write.csv(perc_table,'2020_05_01_area_percentage_by_class.csv')

average_area <- sum(res_table$all)/tot_segs
write.csv(average_area,'2020_05_01_av_superpixel_size.csv')
