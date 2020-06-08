# Merge results from cluster runs on 09/08/2019

# They were separate to make processing easier and to allow multiple sessions


setwd("E:/SLICUAV_manusript_code/1_assessing_models/2019_09_19_1_generate_superpixel_features")

for (i in 1:3){
        # Read in features
        features <- read.csv(paste('2019_09_19_segment_features_',i,'.csv',sep = ''),header=F)
        # set NAs to 0 (eg where an index is uniformly zero so normalisation fails)
        features[is.na(features)] <-0 
        
        # Read in labels
        labels <- read.csv(paste('2019_09_19_cluster_labels_',i,'.csv',sep=''),header = F)
        
        if(i==1){
                X <- features
                Y <- labels
        } else {
                X <- rbind(X,features)
                Y <- rbind(Y,labels)
        }
        rm(features,labels)
}

write.table(X,'2019_09_19_compiled_segment_features.csv',sep=',',col.names = F, row.names = F)
write.table(Y,'2019_09_19_compiled_segment_labels.csv',sep=',',col.names = F, row.names = F)