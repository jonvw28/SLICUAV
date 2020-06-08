setwd("E:/SLICUAV_manuscript_code/3_Landscape_mapping/2020_05_04_raster_prevelance_map")

library(raster)

# set up table
all_area <- as.data.frame(matrix(nrow = 617, ncol = 8,data=0))
names(all_area) <- c('ID','BEL','MAC','MIS','PAL','RAN','PUL','SEN')
sen_area <- as.data.frame(matrix(nrow = 617, ncol = 7,data=0))
names(sen_area) <- c('ID','BEL','MAC','MIS','PAL','RAN','PUL')
ran_area <- as.data.frame(matrix(nrow = 617, ncol = 6,data=0))
names(ran_area) <- c('ID','BEL','MAC','MIS','PAL','RAN')

id_list <- read.csv('../2019_11_29_3_forward_prediction_SVM/2019_11_29_grid_ids.csv')

# cycle over blocks and get areas for each class
for(i in 1:617){
        id <- id_list[i,2]
        all_area[i,1] <- id
        sen_area[i,1] <- id
        ran_area[i,1] <- id
        
        cur_shp <- raster::shapefile(paste('C:/Users/jonny/Documents/grid_segments_labelled/2020_04_30/',id,'_preds_2020_04_23.shp',sep=''))
        
        # pull out full model results
        all_mask <- cur_shp$all == 'BEL'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask])
        all_area[i,2] <- area
        
        all_mask <- cur_shp$all == 'MAC'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask])
        all_area[i,3] <- area
        
        all_mask <- cur_shp$all == 'MIS'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask])
        all_area[i,4] <- area
        
        all_mask <- cur_shp$all == 'PAL'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask])
        all_area[i,5] <- area
        
        all_mask <- cur_shp$all == 'RAN'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask])
        all_area[i,6] <- area
        
        all_mask <- cur_shp$all == 'PUL'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask])
        all_area[i,7] <- area
        
        all_mask <- cur_shp$all == 'SEN'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask])
        all_area[i,8] <- area
        
        # pull out sen model results
        all_mask <- cur_shp$sen == 'BEL'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask])
        sen_area[i,2] <- area
        
        all_mask <- cur_shp$sen == 'MAC'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask])
        sen_area[i,3] <- area
        
        all_mask <- cur_shp$sen == 'MIS'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask])
        sen_area[i,4] <- area
        
        all_mask <- cur_shp$sen == 'PAL'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask])
        sen_area[i,5] <- area
        
        all_mask <- cur_shp$sen == 'RAN'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask])
        sen_area[i,6] <- area
        
        all_mask <- cur_shp$sen == 'PUL'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask])
        sen_area[i,7] <- area
        
        
        # pull out ran model results
        all_mask <- cur_shp$ran == 'BEL'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask])
        ran_area[i,2] <- area
        
        all_mask <- cur_shp$ran == 'MAC'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask])
        ran_area[i,3] <- area
        
        all_mask <- cur_shp$ran == 'MIS'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask])
        ran_area[i,4] <- area
        
        all_mask <- cur_shp$ran == 'PAL'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask])
        ran_area[i,5] <- area
        
        all_mask <- cur_shp$ran == 'RAN'
        all_mask[is.na(all_mask)] <- FALSE
        area <- sum(cur_shp$area[all_mask])
        ran_area[i,6] <- area
        
        print(i)
        

}
write.csv(all_area,'2020_05_04_all_grid_area.csv')
write.csv(sen_area,'2020_05_04_sen_grid_area.csv')
write.csv(ran_area,'2020_05_04_ran_grid_area.csv')

# get percentages
all_perc <- apply(all_area[,c('BEL','MAC','MIS','PAL','RAN','PUL','SEN')],1,function(x){x/sum(x)})
all_perc <- cbind(all_area[,'ID'],t(all_perc))

sen_perc <- apply(sen_area[,c('BEL','MAC','MIS','PAL','RAN','PUL')],1,function(x){x/sum(x)})
sen_perc <- cbind(sen_area[,'ID'],t(sen_perc))

ran_perc <- apply(ran_area[,c('BEL','MAC','MIS','PAL','RAN')],1,function(x){x/sum(x)})
ran_perc <- cbind(ran_area[,'ID'],t(ran_perc))

write.csv(all_perc,'2020_05_04_all_grid_perc.csv')
write.csv(sen_perc,'2020_05_04_sen_grid_perc.csv')
write.csv(ran_perc,'2020_05_04_ran_grid_perc.csv')


