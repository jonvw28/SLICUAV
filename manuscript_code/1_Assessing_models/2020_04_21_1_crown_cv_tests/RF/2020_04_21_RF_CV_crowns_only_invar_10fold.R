# Fit SVM to various forms of the data
# run 10 times with 10-fold CV


# Do:
#  all   -    Data as collected
#  sen   -    Merge Sendok and Random
#  ran   -    Merge Pulai, Sendok and Random
#

library(randomForest)
library(caret)


setwd("E:/SLICUAV_manuscript_code/1_Assessingmodels/2020_04_21_1_crown_cv_tests/RF")

################################################################################
#                                                                              #
#                            PRE-PROCESS AND SPLIT DATA                        #                      
#                                                                              #
################################################################################

# set up output table
temp_vec <- numeric(60)
temp_char <- character(60)
res_table <- data.frame(model = temp_char, rep = temp_vec, set = temp_char,
                          accuracy = temp_vec, stringsAsFactors = F
)
rm(temp_char,temp_vec)


pred_table_all <- as.data.frame(matrix(nrow=633,ncol=10,data='NA'),stringsAsFactors = F)
pred_table_sen <- as.data.frame(matrix(nrow=633,ncol=10,data='NA'),stringsAsFactors = F)
pred_table_ran <- as.data.frame(matrix(nrow=633,ncol=10,data='NA'),stringsAsFactors = F)
colnames(pred_table_all) <- paste(rep('fold',10),1:10,sep='_')
colnames(pred_table_sen) <- paste(rep('fold',10),1:10,sep='_')
colnames(pred_table_ran) <- paste(rep('fold',10),1:10,sep='_')

# Load data and create 10 folds (90:10 split)

# read features and labels
X <- as.matrix(read.csv('../../2019_09_12_5_crown_feats_generate_with_all_overlap/2019_09_12_spectral_texture_dsm_features.csv',header=F))
Y <- read.csv('../../2019_09_12_5_crown_feats_generate_with_all_overlap/2019_09_12_cluster_labels.csv',header=F)

# keep only invariant features
Hinv <- read.csv('../../2019_09_12_5_crown_feats_generate_with_all_overlap/2019_09_12_variable_Hinv.csv',header=F,stringsAsFactors = F)
Sinv <- read.csv('../../2019_09_12_5_crown_feats_generate_with_all_overlap/2019_09_12_variable_sizeInv.csv',header=F,stringsAsFactors = F)
mask <- Hinv[,1]=='True' & Sinv[,1]=='True'
rm(Hinv,Sinv)

variable_names <- read.csv('../../2019_09_12_5_crown_feats_generate_with_all_overlap/2019_09_12_variable_names.csv',header=F,stringsAsFactors = F)
kept_var <- variable_names[mask,1]
write.csv(kept_var,'2019_09_12_invar_feat_names.csv')
rm(variable_names)

# filter feats now
X <- X[,mask]
colnames(X) <- kept_var
rm(mask)

fold_list <- read.csv('2020_04_21_fold_list.csv')

for(i in 1:10){
        # generate split indices
        train_idx <- logical(nrow(X))
        for(j in 1:nrow(fold_list)){
            if(fold_list[j,3]!=i){
                train_idx[Y[,2]==fold_list[j,2]] <- TRUE
            }
        }
        
        train_X <- X[train_idx,]
        test_X <- X[!train_idx,]
        train_Y <- Y[train_idx,1]
        test_Y <- Y[!train_idx,1]
        
        # center and scale based only on training data
        preProcValues <- caret::preProcess(train_X, method = c('center','scale'))
        train_X_trans <- predict(preProcValues, train_X)
        test_X_trans <- predict(preProcValues, test_X)
        rm(train_X,test_X,preProcValues)
        
################################################################################
#                                                                              #
#                               DATA AS COLLECTED                              #                      
#                                                                              #
################################################################################
        
        
        #Use weights to get fairer learning
        
        train_N <- length(train_Y)
        N_bel <- sum(train_Y=='BEL')
        N_mac <- sum(train_Y=='MAC')
        N_mis <- sum(train_Y=='MIS')
        N_pal <- sum(train_Y=='PAL')
        N_pul <- sum(train_Y=='PUL')
        N_ran <- sum(train_Y=='RAN')
        N_sen <- sum(train_Y=='SEN')
        
        weights <- numeric(7)
        weights[1] <- train_N/(N_bel*7)
        weights[2] <- train_N/(N_mac*7)
        weights[3] <- train_N/(N_mis*7)
        weights[4] <- train_N/(N_pal*7)
        weights[5] <- train_N/(N_pul*7)
        weights[6] <- train_N/(N_ran*7)
        weights[7] <- train_N/(N_sen*7)
        
        #normalise to sum to 1
        weights <- weights/sum(weights)
        
        rm(train_N,N_bel,N_mac,N_mis,N_pal,N_ran,N_pul,N_sen)
        
        
        ########## Fit the glmnet path for alpha = 1 (lasso)
        
        set.seed(42)
m_all <- randomForest::randomForest(
                    x=train_X_trans,y=train_Y,importance=TRUE,
                    classwt = weights)
        
        
        # predict based on this
        train_pred <- predict(m_all,train_X_trans)
        test_pred <- predict(m_all,test_X_trans)
        
        pred_table_all[train_idx,i] <- as.character(train_pred)
        pred_table_all[!train_idx,i] <- as.character(test_pred)
        
        res_table[i,1] <- 'all'
        res_table[i,2] <- i
        res_table[i,3] <- 'train'
        res_table[i,4] <- sum(train_pred==train_Y)/length(train_Y)

        res_table[i+10,1] <- 'all'
        res_table[i+10,2] <- i
        res_table[i+10,3] <- 'test'
        res_table[i+10,4] <- sum(test_pred==test_Y)/length(test_Y)
        
        # tidy
        rm(m_all,weights)
        
        
################################################################################
#                                                                              #
#                       MERGE SEN AND RAN ON INPUT                             #
#                                                                              #
################################################################################
        
        # Merge manual Sendok label into Random
        drop_idx_tr <- train_Y == "SEN" 
        train_Y[drop_idx_tr] <- 'RAN'
        train_Y <- droplevels(train_Y)
        drop_idx_te <- test_Y == "SEN" 
        test_Y[drop_idx_te] <- 'RAN'
        test_Y <- droplevels(test_Y)
        rm(drop_idx_te,drop_idx_tr)
        
        
        # weights
        
        train_N <- length(train_Y)
        N_bel <- sum(train_Y=='BEL')
        N_mac <- sum(train_Y=='MAC')
        N_mis <- sum(train_Y=='MIS')
        N_pal <- sum(train_Y=='PAL')
        N_pul <- sum(train_Y=='PUL')
        N_ran <- sum(train_Y=='RAN')
        
        weights <- numeric(6)
        weights[1] <- train_N/(N_bel*6)
        weights[2] <- train_N/(N_mac*6)
        weights[3] <- train_N/(N_mis*6)
        weights[4] <- train_N/(N_pal*6)
        weights[5] <- train_N/(N_pul*6)
        weights[6] <- train_N/(N_ran*6)
        
        weights <- weights / sum(weights)
        
        rm(train_N,N_bel,N_mac,N_mis,N_pal,N_pul,N_ran)
        
        ########## Fit the glmnet path for alpha = 1 (lasso)
        
        set.seed(42)
		set.seed(42)
m_sen <- randomForest::randomForest(
        x=train_X_trans,y=train_Y,importance=TRUE,
        classwt = weights)

		train_pred<-predict(m_sen,train_X_trans)
		test_pred<-predict(m_sen,test_X_trans)
		pred_table_sen[train_idx,i] <- as.character(train_pred)
		pred_table_sen[!train_idx,i] <- as.character(test_pred)


        
        res_table[i+20,1] <- 'sen_dropped'
        res_table[i+20,2] <- i
        res_table[i+20,3] <- 'train'
        res_table[i+20,4] <- sum(train_pred==train_Y)/length(train_Y)
        
        res_table[i+30,1] <- 'sen_dropped'
        res_table[i+30,2] <- i
        res_table[i+30,3] <- 'test'
        res_table[i+30,4] <- sum(test_pred==test_Y)/length(test_Y)
        
        
        # tidy
    rm(m_sen,train_pred,test_pred,weights)

        
################################################################################
#                                                                              #
#               MERGE PUL, SEN AND RAN ON OUTPUT                               #                      
#                                                                              #
################################################################################
        
        # Merge manual pulai label into Random
        drop_idx_tr <- train_Y == "PUL" 
        train_Y[drop_idx_tr] <- 'RAN'
        train_Y <- droplevels(train_Y)
        drop_idx_te <- test_Y == "PUL" 
        test_Y[drop_idx_te] <- 'RAN'
        test_Y <- droplevels(test_Y)
        rm(drop_idx_te,drop_idx_tr)
        
        # weights
        
        train_N <- length(train_Y)
        N_bel <- sum(train_Y=='BEL')
        N_mac <- sum(train_Y=='MAC')
        N_mis <- sum(train_Y=='MIS')
        N_pal <- sum(train_Y=='PAL')
        N_ran <- sum(train_Y=='RAN')
        
        weights <- numeric(5)
        weights[1] <- train_N/(N_bel*5)
        weights[2] <- train_N/(N_mac*5)
        weights[3] <- train_N/(N_mis*5)
        weights[4] <- train_N/(N_pal*5)
        weights[5] <- train_N/(N_ran*5)
        
        weights <- weights / sum(weights)
        
        rm(train_N,N_bel,N_mac,N_mis,N_pal,N_ran)
        
        ########## Fit the glmnet path for alpha = 1 (lasso)
        
        set.seed(42)
m_ran <- randomForest::randomForest(
        x=train_X_trans,y=train_Y,importance=TRUE,
        classwt = weights)

		train_pred<-predict(m_ran,train_X_trans)
		test_pred<-predict(m_ran,test_X_trans)
		pred_table_ran[train_idx,i] <- as.character(train_pred)
		pred_table_ran[!train_idx,i] <- as.character(test_pred)


        
        res_table[i+40,1] <- 'sen_ran_dropped'
        res_table[i+40,2] <- i
        res_table[i+40,3] <- 'train'
        res_table[i+40,4] <- sum(train_pred==train_Y)/length(train_Y)
        
        res_table[i+50,1] <- 'sen_ran_dropped'
        res_table[i+50,2] <- i
        res_table[i+50,3] <- 'test'
        res_table[i+50,4] <- sum(test_pred==test_Y)/length(test_Y)
        
        
        # tidy
    rm(m_ran,train_pred,test_pred,weights)
        
        print(i)
}

write.csv(res_table,'2020_04_21_RF_accuracies_in_10fold_CV.csv')
write.csv(pred_table_all,'2020_04_21_RF_predictions_all_model_10fold.csv')
write.csv(pred_table_sen,'2020_04_21_RF_predictions_drop_sen_10fold.csv')
write.csv(pred_table_ran,'2020_04_21_RF_predictions_drop_sen_pul_10fold.csv')