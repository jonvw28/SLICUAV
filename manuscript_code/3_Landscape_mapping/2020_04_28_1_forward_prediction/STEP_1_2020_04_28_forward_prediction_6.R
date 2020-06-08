# Attemoting to forward predict on a complete image using pre-built models

library(caret)
library(e1071)
library(tictoc)


setwd("E:/SLICUAV_manuscript_code/3_Landscape_mapping/2020_04_28_1_forward_prediction")
set.seed(42)

N_max <- 25

# Load the machinery
load('../2020_04_23_2_train_model/2020_04_23_2_preproc_time_SVM_all.RData') # NB though this is saved for the all models, it is identical to that for the other two models (both by design and by explicit checking)
load(''../2020_04_23_2_train_model/2020_04_23_2_all_classes_svm_model.RData')
load(''../2020_04_23_2_train_model/2020_04_23_2_no_sen_classes_svm_model.RData')
load(''../2020_04_23_2_train_model/2020_04_23_2_no_sen_pul_classes_svm_model.RData')

# load id list
id_list <- read.csv('2019_11_29_grid_ids.csv',header=TRUE)

# keep only invariant features
Hinv <- read.csv('../../1_Assessing_models/2019_09_19_1_generate_superpixel_features/2019_09_19_variable_Hinv.csv',header=F,stringsAsFactors = F)
Sinv <- read.csv('../../1_Assessing_models/2019_09_19_1_generate_superpixel_features/2019_09_19_variable_sizeInv.csv',header=F,stringsAsFactors = F)
mask <- Hinv[,1]=='True' & Sinv[,1]=='True'
rm(Hinv,Sinv)

variable_names <- read.csv('../../1_Assessing_models/2019_09_19_1_generate_superpixel_features/2019_09_19_variable_names.csv',header=F,stringsAsFactors = F)
kept_var <- variable_names[mask,1]

tic()
for(i in 251:300){
        id <- id_list[i,2]
        
        # read in the data
        X <- as.matrix(read.csv(paste('E:/SLICUAV_manuscript_data/5_Landscape_superpixels/features/2019_10_14_segment_features_id_',id,'.csv',sep=''),header = FALSE))
        X[is.na(X)] <- 0
        X[is.nan(X)] <- 0
        X[is.infinite(X)] <- 0
        
        

        # filter feats now
        X <- X[,mask]
        colnames(X) <- kept_var
        
        # predict data
        X <- predict(preProcValues, X)
        labels <- read.csv(paste('E:/SLICUAV_manuscript_data/5_Landscape_superpixels/features/2019_10_14_cluster_labels_id_',id,'.csv',sep=''),stringsAsFactors = FALSE,header = FALSE)
        
        # SVM
        Y_svm_all <- predict(m_all,X)
        Y_svm_mods <- predict(m_sen,X)
        Y_svm_modsp <- predict(m_ran,X)
        
        results <- data.frame(grid = labels[,1], cluster = labels[,2],
                              all_svm = Y_svm_all, sen = Y_svm_mods, ran = Y_svm_modsp)
        names(results) <- c('grid','cluster','all','sen','ran')
        write.table(results, paste('E:/SLICUAV_manuscript_data/5_Landscape_superpixels/predictions/2020_04_28_preds_id_',id,'.csv',sep=''),sep=',',row.names=F)
        print(i)
}
time_50 <- toc()
write.csv(time_50$toc-time_50$tic,'2020_04_28_1_forward_pred_time_6.csv')
