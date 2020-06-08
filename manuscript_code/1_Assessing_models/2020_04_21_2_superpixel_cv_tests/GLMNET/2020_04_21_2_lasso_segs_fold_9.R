# Fit various forms of lasso models to the crown segment data using uncorrected MS data

#################### All features, only DSM

# here use all variables that are invariant

# Build a total of 7 basic combinations of data streams

#  all -     use all features
#  noDSM -   use RGB and MS feats
#  noMS -    use RGB and DSM feats
#  noRGB -   use RGB and DSM feats
#  RGB -     only RGB feats
#  MS -     only MS feats
#  DSM -     only DSM feats

# For each stream do:
#  all   -    Data as collected
#  mods  -    Post-process to combine Sendok and Random
#  sen   -    Merge Sendok and Random
#  modsp -    Post-process to combine Pulai, Sendok and Random
#  ran   -    Merge Pulai, Sendok and Random



library(glmnet)
library(tictoc)
library(caret)

setwd("E:/SLICUAV_manusript_code/1_assessing_models/2020_04_21_2_superpixel_cv_tests/GLMNET")

fold <- 9

N_max <- 25

################################################################################
#                                                                              #
#                            PRE-PROCESS AND SPLIT DATA                        #
#                                                                              #
################################################################################

# read features and labels (for using uncorrected MS data)
X <- as.matrix(read.csv('../../2019_09_19_1_generate_superpixel_features/2019_09_19_compiled_segment_features.csv',header=F))
Y <- read.csv('../../2019_09_19_1_generate_superpixel_features/2019_09_19_compiled_segment_labels.csv',header=F,stringsAsFactors = F)

# keep only invariant features
Hinv <- read.csv('../../2019_09_19_1_generate_superpixel_features/2019_09_19_variable_Hinv.csv',header=F,stringsAsFactors = F)
Sinv <- read.csv('../../2019_09_19_1_generate_superpixel_features/2019_09_19_variable_sizeInv.csv',header=F,stringsAsFactors = F)
mask <- Hinv[,1]=='True' & Sinv[,1]=='True'
rm(Hinv,Sinv)

variable_names <- read.csv('../../2019_09_19_1_generate_superpixel_features/2019_09_19_variable_names.csv',header=F,stringsAsFactors = F)
kept_var <- variable_names[mask,1]

rm(variable_names)

# filter feats now to be only invariant ones
X <- X[,mask]
colnames(X) <- kept_var


fold_list <- read.csv('../../2020_04_21_1_crown_cv_tests/GLMNET/2020_04_21_fold_list.csv')



train_idx <- logical(nrow(X))
for(j in 1:nrow(fold_list)){
        if(fold_list[j,3]!=fold){
                train_idx[Y[,2]==fold_list[j,2]] <- TRUE
        }
}

# set up output table
temp_vec = character(nrow(X))
preds_table <- data.frame(tagstring = Y[,2],cluster = Y[,3],label = Y[,1],
                          label_nosen = temp_vec,
                          label_ran = temp_vec,
                          split = temp_vec,
                          fold_9_all = temp_vec,
                          fold_9_sen = temp_vec,
                          fold_9_ran = temp_vec,
                          stringsAsFactors = F
                          )
preds_table[train_idx,'split'] <- 'train'
preds_table[!train_idx,'split'] <- 'test'

rm(temp_vec)


tic()
train_X <- X[train_idx,]
train_Y <- Y[train_idx,1]
test_X <- X[!train_idx,]
test_Y <- Y[!train_idx,1]

# center and scale based only on training data
preProcValues <- caret::preProcess(train_X, method = c('center','scale'))
train_X_trans <- predict(preProcValues, train_X)
test_X_trans <- predict(preProcValues, test_X)
pptime <- toc()
write.csv(pptime$toc-pptime$tic,paste('2020_04_21_2_preproc_time_lasso_fold_',fold,'.csv',sep=''))

# Save preprocessing machinery and also the variables which were used
save(preProcValues,file = paste('2020_04_21_2_preproc_lasso_fold_',fold,'.RData',sep=''))


# tidy
rm(preProcValues,test_X,train_X)


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

###################### Fit lasso model on full data ############################
tic()
set.seed(42)
m_all <- glmnet::glmnet(train_X_trans, train_Y, weights = weights, alpha = 1, family="multinomial", 
                          type.multinomial = "grouped")
save(m_all,file = paste('2020_04_21_2_lasso_fold_',fold,'_all_classes_lasso_cluster_model.RData',sep=''))

# Pick lambda that keeps at most 20 predictors
mdl_idx <- max(which(m_all$df<=N_max))
mdl_lam <- m_all$lambda[mdl_idx]

train_pred <- predict(m_all, newx = train_X_trans, s = mdl_lam, type = "class")
test_pred <- predict(m_all, newx = test_X_trans, s = mdl_lam, type = "class")

preds_table[train_idx,paste('fold_',fold,'_all',sep='')] <- as.character(train_pred)
preds_table[!train_idx,paste('fold_',fold,'_all',sep='')] <- as.character(test_pred)
lasso_all_time <- toc()
write.csv(lasso_all_time$toc-lasso_all_time$tic,paste('2020_04_21_2_all_classes_lasso_time_fold_',fold,'.csv',sep=''))

# tidy
rm(m_all,train_pred,test_pred,weights,mdl_idx,mdl_lam)

write.csv(preds_table,paste('2020_04_21_2_fold_',fold,'_lasso_predictions_pre_sen.csv',sep=''))

################################################################################
#                                                                              #
#                       MERGE SEN AND RAN ON OUTPUT                            #
#                                                                              #
################################################################################
# Merge manual Sendok label into Random

drop_idx <- preds_table$label == "SEN" 
tmp_Y <- preds_table$label
tmp_Y[drop_idx] <- 'RAN'
preds_table$label_nosen <- tmp_Y


################################################################################
#                                                                              #
#                       MERGE SEN AND RAN ON INPUT                             #
#                                                                              #
################################################################################

train_Y_mods <- preds_table$label_nosen
train_Y_mods <- train_Y_mods[preds_table$split=='train']

# weights

train_N <- length(train_Y_mods)
N_bel <- sum(train_Y_mods=='BEL')
N_mac <- sum(train_Y_mods=='MAC')
N_mis <- sum(train_Y_mods=='MIS')
N_pal <- sum(train_Y_mods=='PAL')
N_pul <- sum(train_Y_mods=='PUL')
N_ran <- sum(train_Y_mods=='RAN')


weights <- numeric(train_N)
weights[train_Y_mods == 'BEL'] <- train_N/(N_bel*6)
weights[train_Y_mods == 'MAC'] <- train_N/(N_mac*6)
weights[train_Y_mods == 'MIS'] <- train_N/(N_mis*6)
weights[train_Y_mods == 'PAL'] <- train_N/(N_pal*6)
weights[train_Y_mods == 'PUL'] <- train_N/(N_pul*6)
weights[train_Y_mods == 'RAN'] <- train_N/(N_ran*6)


rm(train_N,N_bel,N_mac,N_mis,N_pal,N_pul,N_ran)

###################### Fit lasso model on no sen data ############################
tic()
set.seed(42)
m_sen <- glmnet::glmnet(train_X_trans, train_Y_mods, weights = weights, alpha = 1, family="multinomial", 
                         type.multinomial = "grouped")
save(m_sen,file = paste('2020_04_21_2_lasso_fold_1_',fold,'_drop_sen_lasso_cluster_model.RData',sep=''))

# Pick lambda that keeps at most 20 predictors
mdl_idx <- max(which(m_sen$df<=N_max))
mdl_lam <- m_sen$lambda[mdl_idx]

# predict based on this
train_pred <- predict(m_sen, newx = train_X_trans, s = mdl_lam, type = "class")
test_pred <- predict(m_sen, newx = test_X_trans, s = mdl_lam, type = "class")
preds_table[train_idx,paste('fold_',fold,'_sen',sep='')] <- as.character(train_pred)
preds_table[!train_idx,paste('fold_',fold,'_sen',sep='')] <- as.character(test_pred)
lasso_sen_time <- toc()
write.csv(lasso_sen_time$toc-lasso_sen_time$tic,paste('2020_04_21_2_drop_sen_lasso_time_fold_',fold,'.csv',sep=''))

# tidy
rm(m_sen,train_pred,test_pred,weights,mdl_idx,mdl_lam)

write.csv(preds_table,paste('2020_04_21_2_fold_',fold,'_lasso_predictions_pre_ran.csv',sep=''))

################################################################################
#                                                                              #
#               MERGE PUL, SEN AND RAN ON OUTPUT                               #
#                                                                              #
################################################################################

# Merge manual Sendok label into Random

drop_idx <- preds_table$label == "SEN" | preds_table$label =='PUL'
tmp_Y <- preds_table$label
tmp_Y[drop_idx] <- 'RAN'
preds_table$label_ran <- tmp_Y
rm(tmp_Y,drop_idx)



################################################################################
#                                                                              #
#               MERGE PUL, SEN AND RAN ON InPUT                                #
#                                                                              #
################################################################################

train_Y_modsp <- preds_table$label_ran
train_Y_modsp <- train_Y_modsp[preds_table$split=='train']

# weights

train_N <- length(train_Y_modsp)
N_bel <- sum(train_Y_modsp=='BEL')
N_mac <- sum(train_Y_modsp=='MAC')
N_mis <- sum(train_Y_modsp=='MIS')
N_pal <- sum(train_Y_modsp=='PAL')
N_ran <- sum(train_Y_modsp=='RAN')

weights <- numeric(train_N)
weights[train_Y_modsp == 'BEL'] <- train_N/(N_bel*5)
weights[train_Y_modsp == 'MAC'] <- train_N/(N_mac*5)
weights[train_Y_modsp == 'MIS'] <- train_N/(N_mis*5)
weights[train_Y_modsp == 'PAL'] <- train_N/(N_pal*5)
weights[train_Y_modsp == 'RAN'] <- train_N/(N_ran*5)

rm(train_N,N_bel,N_mac,N_mis,N_pal,N_ran)

###################### Fit lasso model on no sen or ran data #####################
tic()
set.seed(42)
m_ran <- glmnet::glmnet(train_X_trans, train_Y_modsp, weights = weights, alpha = 1, family="multinomial", 
                         type.multinomial = "grouped")
save(m_ran,file = paste('2020_04_21_2_lasso_fold_1_',fold,'_drop_sen_pul_lasso_cluster_model.RData',sep=''))

# Pick lambda that keeps at most 20 predictors
mdl_idx <- max(which(m_ran$df<=N_max))
mdl_lam <- m_ran$lambda[mdl_idx]

# predict based on this
train_pred <- predict(m_ran, newx = train_X_trans, s = mdl_lam, type = "class")
test_pred <- predict(m_ran, newx = test_X_trans, s = mdl_lam, type = "class")
preds_table[train_idx,paste('fold_',fold,'_ran',sep='')] <- as.character(train_pred)
preds_table[!train_idx,paste('fold_',fold,'_ran',sep='')] <- as.character(test_pred)
lasso_ran_time <- toc()
write.csv(lasso_ran_time$toc-lasso_ran_time$tic,paste('2020_04_21_2_drop_pul_sen_lasso_time_fold_',fold,'.csv',sep=''))

# tidy
rm(m_ran,train_pred,test_pred,weights)


############################# SAVE OUTPUT ######################################

write.csv(preds_table,paste('2020_04_21_2_fold_',fold,'_lasso_predictions.csv',sep=''))