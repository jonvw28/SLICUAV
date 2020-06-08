# Fit various forms of SVM models to the crown segment data using uncorrected MS data

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




library(e1071)
library(tictoc)
library(caret)

setwd("E:/SLICUAV_manuscript_code/2_testing_imagery_and_features/2020_05_01_2_multiplexing_imagery_features")

################################################################################
#                                                                              #
#                            PRE-PROCESS AND SPLIT DATA                        #
#                                                                              #
################################################################################

# read features and labels (for using uncorrected MS data)
# read features and labels
X <- as.matrix(read.csv('../../1_Assessing_models/2019_09_19_1_generate_superpixel_features/2019_09_19_compiled_segment_features.csv',header=F))
Y <- read.csv('../../1_Assessing_models/2019_09_19_1_generate_superpixel_features/2019_09_19_compiled_segment_labels.csv',header=F,stringsAsFactors = F)

# keep only invariant features
Hinv <- read.csv('../../1_Assessing_models/2019_09_19_1_generate_superpixel_features/2019_09_19_variable_Hinv.csv',header=F,stringsAsFactors = F)
Sinv <- read.csv('../../1_Assessing_models/2019_09_19_1_generate_superpixel_features/2019_09_19_variable_sizeInv.csv',header=F,stringsAsFactors = F)
mask <- Hinv[,1]=='True' & Sinv[,1]=='True'
rm(Hinv,Sinv)

variable_names <- read.csv('../../1_Assessing_models/2019_09_19_1_generate_superpixel_features/2019_09_19_variable_names.csv',header=F,stringsAsFactors = F)
kept_var <- variable_names[mask,1]

rm(variable_names)

# filter feats now to be only invariant ones
X <- X[,mask]
colnames(X) <- kept_var

# Deal with variable classes

var_classes <- read.csv('../../1_Assessing_models/2019_09_19_1_generate_superpixel_features/2019_09_19_variable_class.csv',header=F,stringsAsFactors = F)
var_classes <- var_classes[mask,]

rm(mask)

rgb_vars <- grep('rgb_*',var_classes,ignore.case = T)
ms_vars <- grep('ms_*',var_classes,ignore.case = T)
dsm_vars <- grep('dsm_*',var_classes,ignore.case = T)
band_vars <- grep('*band*|*top*|*hsv*|*ind*|*raw*',var_classes,ignore.case = T)
text_vars <- grep('*acor*|*lbp*|*laws*|*glcm*',var_classes,ignore.case = T)

rm(var_classes)

####### SUBSET to only good MS data
good_MS <- read.csv('../0b_get_tree_ids/2020_03_03_unaffected_tree_ids.csv',stringsAsFactors = FALSE)
good_MS_mask <- Y[,2] %in% good_MS[,2]

# Load split from previous work
split_data <- read.csv('../0a_create_split/2019_09_12_train_idx.csv',stringsAsFactors = F)
split_info <- cbind(split_data$tagstring,split_data$split)
rm(split_data)

# generate split indices
train_idx <- logical(nrow(X))
for(i in 1:nrow(split_info)){
        if(split_info[i,2]=='train'){
                train_idx[Y[,2]==split_info[i,1]] <- TRUE
        }
}
rm(split_info)

# set up output table
temp_vec = character(sum(good_MS_mask))
preds_table <- data.frame(ttagstring = Y[good_MS_mask,2],cluster = Y[good_MS_mask,3],label = Y[good_MS_mask,1],
                          label_nosen = temp_vec,
                          label_ran = temp_vec,
                          split = temp_vec,
                          all_feats_noRGB_imagery_all = temp_vec,
                          all_feats_noRGB_imagery_mods = temp_vec,
                          all_feats_noRGB_imagery_sen = temp_vec,
                          all_feats_noRGB_imagery_modsp = temp_vec,
                          all_feats_noRGB_imagery_ran = temp_vec,
                          stringsAsFactors = F
                          )
preds_table[train_idx[good_MS_mask],'split'] <- 'train'
preds_table[!train_idx[good_MS_mask],'split'] <- 'test'

rm(temp_vec)

# no RGB
var_mask <-union(dsm_vars,ms_vars)
mod_string <- 'all_feats_noRGB_imagery'

tic()
train_X <- X[train_idx & good_MS_mask,var_mask]
train_Y <- Y[train_idx & good_MS_mask,1]
test_X <- X[!train_idx & good_MS_mask,var_mask]
test_Y <- Y[!train_idx & good_MS_mask,1]

# center and scale based only on training data
preProcValues <- caret::preProcess(train_X, method = c('center','scale'))
train_X_trans <- predict(preProcValues, train_X)
test_X_trans <- predict(preProcValues, test_X)
pptime <- toc()
write.csv(pptime$toc-pptime$tic,paste('2020_05_01_preproc_time_',mod_string,'.csv',sep=''))

# Save preprocessing machinery and also the variables which were used
save(preProcValues,file = paste('2020_05_01_preproc_',mod_string,'.RData',sep=''))
write.csv(kept_var[var_mask],paste('2020_05_01_variables_',mod_string,'.csv',sep=''))


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

###################### Fit svm model on full data ############################
tic()
set.seed(42)
m_all <- e1071::best.svm(
                    x=train_X_trans,y=factor(train_Y),
                    class.weights=c(BEL = weights[1],
                                    MAC = weights[2],
                                    MIS = weights[3],
                                    PAL = weights[4],
                                    PUL = weights[5],
                                    RAN = weights[6],
                                    SEN = weights[7]),
                    type = 'C'
                    )
save(m_all,file = paste('2020_05_01_',mod_string,'_all_classes_svm_cluster_model.RData',sep=''))

train_pred<-predict(m_all,train_X_trans)
test_pred<-predict(m_all,test_X_trans)
preds_table[train_idx[good_MS_mask],paste(mod_string,'_all',sep='')] <- as.character(train_pred)
preds_table[!train_idx[good_MS_mask],paste(mod_string,'_all',sep='')] <- as.character(test_pred)
svm_all_time <- toc()
write.csv(svm_all_time$toc-svm_all_time$tic,paste('2020_05_01_all_classes_svm_time_',mod_string,'.csv',sep=''))

# tidy
rm(m_all,train_pred,test_pred,weights)



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


# Do same merge for outputs
pred_tmp_1 <- preds_table[,paste(mod_string,'_all',sep='')]
mrg_idx_1 <- pred_tmp_1 == "SEN"
pred_tmp_1[mrg_idx_1] <- 'RAN'
preds_table[,paste(mod_string,'_mods',sep='')] <- pred_tmp_1
rm(pred_tmp_1, mrg_idx_1)



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

weights <- numeric(6)
weights[1] <- train_N/(N_bel*6)
weights[2] <- train_N/(N_mac*6)
weights[3] <- train_N/(N_mis*6)
weights[4] <- train_N/(N_pal*6)
weights[5] <- train_N/(N_pul*6)
weights[6] <- train_N/(N_ran*6)

weights <- weights / sum(weights)

rm(train_N,N_bel,N_mac,N_mis,N_pal,N_pul,N_ran)

###################### Fit svm model on no sen data ############################
tic()
set.seed(42)
m_sen <- e1071::best.svm(
        x=train_X_trans,y=factor(train_Y_mods),
        class.weights=c(BEL = weights[1],
                        MAC = weights[2],
                        MIS = weights[3],
                        PAL = weights[4],
                        PUL = weights[5],
                        RAN = weights[6]),
        type = 'C'
)
save(m_sen,file = paste('2020_05_01_',mod_string,'_drop_sen_svm_cluster_model.RData',sep=''))

train_pred<-predict(m_sen,train_X_trans)
test_pred<-predict(m_sen,test_X_trans)
preds_table[train_idx[good_MS_mask],paste(mod_string,'_sen',sep='')] <- as.character(train_pred)
preds_table[!train_idx[good_MS_mask],paste(mod_string,'_sen',sep='')] <- as.character(test_pred)
svm_sen_time <- toc()
write.csv(svm_sen_time$toc-svm_sen_time$tic,paste('2020_05_01_drop_sen_svm_time_',mod_string,'.csv',sep=''))

# tidy
rm(m_sen,train_pred,test_pred,weights)


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


# Do same merge for outputs

pred_tmp_1 <- preds_table[,paste(mod_string,'_all',sep='')]
mrg_idx_1 <- pred_tmp_1 == "SEN" | pred_tmp_1 == "PUL"
pred_tmp_1[mrg_idx_1] <- 'RAN'
preds_table[,paste(mod_string,'_modsp',sep='')] <- pred_tmp_1
rm(pred_tmp_1, mrg_idx_1)


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

weights <- numeric(5)
weights[1] <- train_N/(N_bel*5)
weights[2] <- train_N/(N_mac*5)
weights[3] <- train_N/(N_mis*5)
weights[4] <- train_N/(N_pal*5)
weights[5] <- train_N/(N_ran*5)

weights <- weights / sum(weights)

rm(train_N,N_bel,N_mac,N_mis,N_pal,N_ran)

###################### Fit svm model on no sen or ran data #####################
tic()
set.seed(42)
m_ran <- e1071::best.svm(
        x=train_X_trans,y=factor(train_Y_modsp),
        class.weights=c(BEL = weights[1],
                        MAC = weights[2],
                        MIS = weights[3],
                        PAL = weights[4],
                        RAN = weights[5])
)
save(m_ran,file = paste('2020_05_01_',mod_string,'_drop_sen_pul_svm_cluster_model.RData',sep=''))

train_pred<-predict(m_ran,train_X_trans)
test_pred<-predict(m_ran,test_X_trans)
preds_table[train_idx[good_MS_mask],paste(mod_string,'_ran',sep='')] <- as.character(train_pred)
preds_table[!train_idx[good_MS_mask],paste(mod_string,'_ran',sep='')] <- as.character(test_pred)
svm_ran_time <- toc()
write.csv(svm_ran_time$toc-svm_ran_time$tic,paste('2020_05_01_drop_pul_sen_svm_time_',mod_string,'.csv',sep=''))

# tidy
rm(m_ran,train_pred,test_pred,weights)


############################# SAVE OUTPUT ######################################

write.csv(preds_table,paste('2020_05_01_',mod_string,'_svm_predictions.csv',sep=''))
