# Compute CV statitics for the crown models


library(dplyr)
library(caret)

setwd("E:/SLICUAV_manusript_code/1_assessing_models/2020_04_23_1_analysing_cv_tests/SEGS")

temp_vec <- character(11996)
temp_num <- numeric(11996)
preds_table <- data.frame(fold = temp_num,
                          label_all = temp_vec,
                          label_sen = temp_vec,
                          label_ran = temp_vec,
                          glm_all = temp_vec,
                          glm_sen = temp_vec,
                          glm_ran = temp_vec,
                          svm_all = temp_vec,
                          svm_sen = temp_vec,
                          svm_ran = temp_vec,
                          rf_all = temp_vec,
                          rf_sen = temp_vec,
                          rf_ran = temp_vec,
                          stringsAsFactors = F)
rm(temp_vec,temp_num)









################################################################################
#                                                                              #
#                            GLMNET                                            #
#                                                                              #
################################################################################

temp_vec <- numeric(60)
temp_char <- character(60)
GLM_acc_tbl <- data.frame(model = temp_char, rep = temp_vec, set = temp_char,
                          accuracy = temp_vec, stringsAsFactors = F
)
rm(temp_char,temp_vec)

for(i in 1:10){
        data <- read.csv(paste('../../2020_04_21_2_superpixels_cv_tests/GLMNET/2020_04_21_2_fold_',i,'_lasso_predictions.csv',sep=''))
        
        train_data <- data %>%
                dplyr::filter(split == 'train')
        test_data <- data %>%
                dplyr::filter(split == 'test')
        
        if(i==1){
                preds_table[,'label_all'] <- data$label
                preds_table[,'label_sen'] <- data$label_nosen
                preds_table[,'label_ran'] <- data$label_ran
                
        }
        
        preds_table[data$split=='test','fold'] <- i
        
        # All classes
        GLM_acc_tbl[i,1] <- 'all'
        GLM_acc_tbl[i,2] <- i
        GLM_acc_tbl[i,3] <- 'train'
        GLM_acc_tbl[i,4] <- sum(train_data[,paste('fold_',i,'_all',sep='')] == train_data[,'label'])/nrow(train_data)
        GLM_acc_tbl[i+10,1] <- 'all'
        GLM_acc_tbl[i+10,2] <- i
        GLM_acc_tbl[i+10,3] <- 'test'
        GLM_acc_tbl[i+10,4] <- sum(test_data[,paste('fold_',i,'_all',sep='')] == test_data[,'label'])/nrow(test_data)
        
        preds_table[data$split=='test','glm_all'] <- as.character(data[data$split=='test',paste('fold_',i,'_all',sep='')])
        
        # No SEN
        GLM_acc_tbl[i+20,1] <- 'sen'
        GLM_acc_tbl[i+20,2] <- i
        GLM_acc_tbl[i+20,3] <- 'train'
        GLM_acc_tbl[i+20,4] <- sum(train_data[,paste('fold_',i,'_sen',sep='')] == train_data[,'label_nosen'])/nrow(train_data)
        GLM_acc_tbl[i+30,1] <- 'sen'
        GLM_acc_tbl[i+30,2] <- i
        GLM_acc_tbl[i+30,3] <- 'test'
        GLM_acc_tbl[i+30,4] <- sum(test_data[,paste('fold_',i,'_sen',sep='')] == test_data[,'label_nosen'])/nrow(test_data)
        
        preds_table[data$split=='test','glm_sen'] <- as.character(data[data$split=='test',paste('fold_',i,'_sen',sep='')])
        
        # No SEN PUL
        GLM_acc_tbl[i+40,1] <- 'ran'
        GLM_acc_tbl[i+40,2] <- i
        GLM_acc_tbl[i+40,3] <- 'train'
        GLM_acc_tbl[i+40,4] <- sum(train_data[,paste('fold_',i,'_ran',sep='')] == train_data[,'label_ran'])/nrow(train_data)
        GLM_acc_tbl[i+50,1] <- 'ran'
        GLM_acc_tbl[i+50,2] <- i
        GLM_acc_tbl[i+50,3] <- 'test'
        GLM_acc_tbl[i+50,4] <- sum(test_data[,paste('fold_',i,'_ran',sep='')] == test_data[,'label_ran'])/nrow(test_data)
        
        preds_table[data$split=='test','glm_ran'] <- as.character(data[data$split=='test',paste('fold_',i,'_ran',sep='')])
}

glm_acc <- GLM_acc_tbl %>%
        dplyr::group_by(model,set) %>%
        dplyr::summarise(mean = mean(accuracy),SD = sd(accuracy))

rm(GLM_acc_tbl)
rm(data,train_data,test_data)
################################################################################
#                                                                              #
#                            SVM                                               #
#                                                                              #
################################################################################

temp_vec <- numeric(60)
temp_char <- character(60)
SVM_acc_tbl <- data.frame(model = temp_char, rep = temp_vec, set = temp_char,
                          accuracy = temp_vec, stringsAsFactors = F
)
rm(temp_char,temp_vec)

for(i in 1:10){
        data <- read.csv(paste('../../2020_04_21_2_superpixels_cv_tests/SVM/2020_04_21_1_fold_',i,'_svm_predictions.csv',sep=''))
        
        train_data <- data %>%
                dplyr::filter(split == 'train')
        test_data <- data %>%
                dplyr::filter(split == 'test')
        
        SVM_acc_tbl[i,1] <- 'all'
        SVM_acc_tbl[i,2] <- i
        SVM_acc_tbl[i,3] <- 'train'
        SVM_acc_tbl[i,4] <- sum(train_data[,paste('fold_',i,'_all',sep='')] == train_data[,'label'])/nrow(train_data)
        SVM_acc_tbl[i+10,1] <- 'all'
        SVM_acc_tbl[i+10,2] <- i
        SVM_acc_tbl[i+10,3] <- 'test'
        SVM_acc_tbl[i+10,4] <- sum(test_data[,paste('fold_',i,'_all',sep='')] == test_data[,'label'])/nrow(test_data)
        
        preds_table[data$split=='test','svm_all'] <- as.character(data[data$split=='test',paste('fold_',i,'_all',sep='')])
        
        # No SEN
        SVM_acc_tbl[i+20,1] <- 'sen'
        SVM_acc_tbl[i+20,2] <- i
        SVM_acc_tbl[i+20,3] <- 'train'
        SVM_acc_tbl[i+20,4] <- sum(train_data[,paste('fold_',i,'_sen',sep='')] == train_data[,'label_nosen'])/nrow(train_data)
        SVM_acc_tbl[i+30,1] <- 'sen'
        SVM_acc_tbl[i+30,2] <- i
        SVM_acc_tbl[i+30,3] <- 'test'
        SVM_acc_tbl[i+30,4] <- sum(test_data[,paste('fold_',i,'_sen',sep='')] == test_data[,'label_nosen'])/nrow(test_data)
        
        preds_table[data$split=='test','svm_sen'] <- as.character(data[data$split=='test',paste('fold_',i,'_sen',sep='')])
        
        # No SEN PUL
        SVM_acc_tbl[i+40,1] <- 'ran'
        SVM_acc_tbl[i+40,2] <- i
        SVM_acc_tbl[i+40,3] <- 'train'
        SVM_acc_tbl[i+40,4] <- sum(train_data[,paste('fold_',i,'_ran',sep='')] == train_data[,'label_ran'])/nrow(train_data)
        SVM_acc_tbl[i+50,1] <- 'ran'
        SVM_acc_tbl[i+50,2] <- i
        SVM_acc_tbl[i+50,3] <- 'test'
        SVM_acc_tbl[i+50,4] <- sum(test_data[,paste('fold_',i,'_ran',sep='')] == test_data[,'label_ran'])/nrow(test_data)

        preds_table[data$split=='test','svm_ran'] <- as.character(data[data$split=='test',paste('fold_',i,'_ran',sep='')])
        
}

svm_acc <- SVM_acc_tbl %>%
        dplyr::group_by(model,set) %>%
        dplyr::summarise(mean = mean(accuracy),SD = sd(accuracy))

rm(SVM_acc_tbl)
rm(data,train_data,test_data)

################################################################################
#                                                                              #
#                            RF                                                #
#                                                                              #
################################################################################

temp_vec <- numeric(60)
temp_char <- character(60)
RF_acc_tbl <- data.frame(model = temp_char, rep = temp_vec, set = temp_char,
                          accuracy = temp_vec, stringsAsFactors = F
)
rm(temp_char,temp_vec)

for(i in 1:10){
        data <- read.csv(paste('../../2020_04_21_2_superpixels_cv_tests/RF/2020_04_21_2_fold_',i,'_RF_predictions.csv',sep=''))
        
        train_data <- data %>%
                dplyr::filter(split == 'train')
        test_data <- data %>%
                dplyr::filter(split == 'test')
        
        
        # All classes
        RF_acc_tbl[i,1] <- 'all'
        RF_acc_tbl[i,2] <- i
        RF_acc_tbl[i,3] <- 'train'
        RF_acc_tbl[i,4] <- sum(train_data[,paste('fold_',i,'_all',sep='')] == train_data[,'label'])/nrow(train_data)
        RF_acc_tbl[i+10,1] <- 'all'
        RF_acc_tbl[i+10,2] <- i
        RF_acc_tbl[i+10,3] <- 'test'
        RF_acc_tbl[i+10,4] <- sum(test_data[,paste('fold_',i,'_all',sep='')] == test_data[,'label'])/nrow(test_data)
        
        preds_table[data$split=='test','rf_all'] <- as.character(data[data$split=='test',paste('fold_',i,'_all',sep='')])
        
        # No SEN
        RF_acc_tbl[i+20,1] <- 'sen'
        RF_acc_tbl[i+20,2] <- i
        RF_acc_tbl[i+20,3] <- 'train'
        RF_acc_tbl[i+20,4] <- sum(train_data[,paste('fold_',i,'_sen',sep='')] == train_data[,'label_nosen'])/nrow(train_data)
        RF_acc_tbl[i+30,1] <- 'sen'
        RF_acc_tbl[i+30,2] <- i
        RF_acc_tbl[i+30,3] <- 'test'
        RF_acc_tbl[i+30,4] <- sum(test_data[,paste('fold_',i,'_sen',sep='')] == test_data[,'label_nosen'])/nrow(test_data)
        
        preds_table[data$split=='test','rf_sen'] <- as.character(data[data$split=='test',paste('fold_',i,'_sen',sep='')])
        
        # No SEN PUL
        RF_acc_tbl[i+40,1] <- 'ran'
        RF_acc_tbl[i+40,2] <- i
        RF_acc_tbl[i+40,3] <- 'train'
        RF_acc_tbl[i+40,4] <- sum(train_data[,paste('fold_',i,'_ran',sep='')] == train_data[,'label_ran'])/nrow(train_data)
        RF_acc_tbl[i+50,1] <- 'ran'
        RF_acc_tbl[i+50,2] <- i
        RF_acc_tbl[i+50,3] <- 'test'
        RF_acc_tbl[i+50,4] <- sum(test_data[,paste('fold_',i,'_ran',sep='')] == test_data[,'label_ran'])/nrow(test_data)
        
        preds_table[data$split=='test','rf_ran'] <- as.character(data[data$split=='test',paste('fold_',i,'_ran',sep='')])
}

rf_acc <- RF_acc_tbl %>%
        dplyr::group_by(model,set) %>%
        dplyr::summarise(mean = mean(accuracy),SD = sd(accuracy))  
rm(RF_acc_tbl)
rm(data,train_data,test_data,i)

################################################################################

#                          collate and save

################################################################################

glm_acc$method <- 'glm'
svm_acc$method <- 'svm'
rf_acc$method <- 'rf'

acc_tbl <- rbind(glm_acc,svm_acc,rf_acc)
rm(glm_acc,svm_acc,rf_acc)

write.csv(acc_tbl,'2020_04_23_1_seg_cv_accuracy_tables.csv')