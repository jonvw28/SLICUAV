# Fit lasso regression to various forms of the data
# run 10 times with 10-fold CV


# Do:
#  all   -    Data as collected
#  sen   -    Merge Sendok and Random
#  ran   -    Merge Pulai, Sendok and Random
#

library(glmnet)
library(caret)

# Fit various glmnet models, where at most N preidctors are allowed across all classes

N_max <- 25

setwd("E:/SLICUAV_manuscript_code/1_Assessingmodels/2020_04_21_1_crown_cv_tests/GLMNET")

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

set.seed(42)
folds <- caret::createFolds(Y[,1],k=10)
fold_list <- numeric(length=633)
for(fld_i in 1:10){
        fold_list[folds[[fld_i]]]<-fld_i
}
rm(fld_i)
fold_list <- data.frame(tagstring = as.character(Y[,2]),fold = fold_list)
write.csv(fold_list,'2020_04_21_fold_list.csv')

for(i in 1:10){
        train_X <- X[-folds[[i]],]
        test_X <- X[folds[[i]],]
        train_Y <- Y[-folds[[i]],1]
        test_Y <- Y[folds[[i]],1]
        
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
        
        weights <- numeric(train_N)
        weights[train_Y == 'BEL'] <- train_N/(N_bel*7)
        weights[train_Y == 'MAC'] <- train_N/(N_mac*7)
        weights[train_Y == 'MIS'] <- train_N/(N_mis*7)
        weights[train_Y == 'PAL'] <- train_N/(N_pal*7)
        weights[train_Y == 'PUL'] <- train_N/(N_pul*7)
        weights[train_Y == 'RAN'] <- train_N/(N_ran*7)
        weights[train_Y == 'SEN'] <- train_N/(N_sen*7)
        
        rm(train_N,N_bel,N_mac,N_mis,N_pal,N_ran,N_pul,N_sen)
        
        
        ########## Fit the glmnet path for alpha = 1 (lasso)
        
        set.seed(42)
        all_a_1<- glmnet::glmnet(train_X_trans, train_Y, weights = weights, alpha = 1, family="multinomial", 
                                 type.multinomial = "grouped")
        
        # Pick lambda that keeps at most 20 predictors
        mdl_idx <- max(which(all_a_1$df<=N_max))
        mdl_lam <- all_a_1$lambda[mdl_idx]
        
        # predict based on this
        train_pred <- predict(all_a_1, newx = train_X_trans, s = mdl_lam, type = "class")
        test_pred <- predict(all_a_1, newx = test_X_trans, s = mdl_lam, type = "class")
        
        pred_table_all[-folds[[i]],i] <- train_pred
        pred_table_all[folds[[i]],i] <- test_pred
        
        res_table[i,1] <- 'all'
        res_table[i,2] <- i
        res_table[i,3] <- 'train'
        res_table[i,4] <- sum(train_pred==train_Y)/length(train_Y)

        res_table[i+10,1] <- 'all'
        res_table[i+10,2] <- i
        res_table[i+10,3] <- 'test'
        res_table[i+10,4] <- sum(test_pred==test_Y)/length(test_Y)
        
        
        
        # tidy
        rm(all_a_1,mdl_idx,mdl_lam,weights)
        
        
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
        
        weights <- numeric(train_N)
        weights[train_Y == 'BEL'] <- train_N/(N_bel*6)
        weights[train_Y == 'MAC'] <- train_N/(N_mac*6)
        weights[train_Y == 'MIS'] <- train_N/(N_mis*6)
        weights[train_Y == 'PAL'] <- train_N/(N_pal*6)
        weights[train_Y == 'PUL'] <- train_N/(N_pul*6)
        weights[train_Y == 'RAN'] <- train_N/(N_ran*6)
        
        rm(train_N,N_bel,N_mac,N_mis,N_pal,N_pul,N_ran)
        
        ########## Fit the glmnet path for alpha = 1 (lasso)
        
        set.seed(42)
        mods_a_1<- glmnet::glmnet(train_X_trans, train_Y, weights = weights, alpha = 1, family="multinomial", 
                                  type.multinomial = "grouped")
        
        # Pick lambda that keeps at most 20 predictors
        mdl_idx <- max(which(mods_a_1$df<=N_max))
        mdl_lam <- mods_a_1$lambda[mdl_idx]
        
        # predict based on this
        train_pred <- predict(mods_a_1, newx = train_X_trans, s = mdl_lam, type = "class")
        test_pred <- predict(mods_a_1, newx = test_X_trans, s = mdl_lam, type = "class")
        
        pred_table_sen[-folds[[i]],i] <- train_pred
        pred_table_sen[folds[[i]],i] <- test_pred

        
        res_table[i+20,1] <- 'sen_dropped'
        res_table[i+20,2] <- i
        res_table[i+20,3] <- 'train'
        res_table[i+20,4] <- sum(train_pred==train_Y)/length(train_Y)
        
        res_table[i+30,1] <- 'sen_dropped'
        res_table[i+30,2] <- i
        res_table[i+30,3] <- 'test'
        res_table[i+30,4] <- sum(test_pred==test_Y)/length(test_Y)
        
        
        # tidy
        rm(mods_a_1,mdl_idx,mdl_lam,weights)
        rm(train_pred,test_pred)
        
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
        
        weights <- numeric(train_N)
        weights[train_Y == 'BEL'] <- train_N/(N_bel*5)
        weights[train_Y == 'MAC'] <- train_N/(N_mac*5)
        weights[train_Y == 'MIS'] <- train_N/(N_mis*5)
        weights[train_Y == 'PAL'] <- train_N/(N_pal*5)
        weights[train_Y == 'RAN'] <- train_N/(N_ran*5)
        
        rm(train_N,N_bel,N_mac,N_mis,N_pal,N_ran)
        
        ########## Fit the glmnet path for alpha = 1 (lasso)
        
        set.seed(42)
        modsp_a_1<- glmnet::glmnet(train_X_trans, train_Y, weights = weights, alpha = 1, family="multinomial", 
                                   type.multinomial = "grouped")
        
        # Pick lambda that keeps at most 20 predictors
        mdl_idx <- max(which(modsp_a_1$df<=N_max))
        mdl_lam <- modsp_a_1$lambda[mdl_idx]
        
        # predict based on this
        train_pred <- predict(modsp_a_1, newx = train_X_trans, s = mdl_lam, type = "class")
        test_pred <- predict(modsp_a_1, newx = test_X_trans, s = mdl_lam, type = "class")
        
        pred_table_ran[-folds[[i]],i] <- train_pred
        pred_table_ran[folds[[i]],i] <- test_pred
        
        res_table[i+40,1] <- 'sen_ran_dropped'
        res_table[i+40,2] <- i
        res_table[i+40,3] <- 'train'
        res_table[i+40,4] <- sum(train_pred==train_Y)/length(train_Y)
        
        res_table[i+50,1] <- 'sen_ran_dropped'
        res_table[i+50,2] <- i
        res_table[i+50,3] <- 'test'
        res_table[i+50,4] <- sum(test_pred==test_Y)/length(test_Y)
        
        
        # tidy
        rm(modsp_a_1,mdl_idx,mdl_lam,weights)
        rm(train_pred,test_pred)
        
        print(i)
}

write.csv(res_table,'2020_04_21_GLMNET_accuracies_in_10fold_CV.csv')
write.csv(pred_table_all,'2020_04_21_GLMNET_predictions_all_model_10fold.csv')
write.csv(pred_table_sen,'2020_04_21_GLMNET_predictions_drop_sen_10fold.csv')
write.csv(pred_table_ran,'2020_04_21_GLMNET_predictions_drop_sen_pul_10fold.csv')