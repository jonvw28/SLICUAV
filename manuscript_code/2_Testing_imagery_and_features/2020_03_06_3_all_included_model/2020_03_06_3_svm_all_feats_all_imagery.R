# Fit 3 models  to various forms of the data

# Do:
#  all   -    Data as collected
#  mods  -    Post-process to combine Sendok and Random
#  sen   -    Merge Sendok and Random
#  modsp -    Post-process to combine Pulai, Sendok and Random
#  ran   -    Merge Pulai, Sendok and Random
#

library(e1071)
library(caret)



setwd("E:/SLICUAV_manuscript_code/2_Testing_imagery_and_features/2020_03_06_3_all_included_model")

################################################################################
#                                                                              #
#                            PRE-PROCESS AND SPLIT DATA                        #                      
#                                                                              #
################################################################################

# Includes 3/4 trianing, 1/4 test split

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
write.csv(kept_var,'2019_09_19_invar_feat_names.csv')
rm(variable_names)

# filter feats now
X <- X[,mask]
colnames(X) <- kept_var
rm(mask)

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

train_X <- X[train_idx & good_MS_mask,]
train_Y <- Y[train_idx & good_MS_mask,1]
test_X <- X[!train_idx & good_MS_mask,]
test_Y <- Y[!train_idx & good_MS_mask,1]


# center and scale based only on training data
preProcValues <- caret::preProcess(train_X, method = c('center','scale'))
train_X_trans <- predict(preProcValues, train_X)
test_X_trans <- predict(preProcValues, test_X)

# set up output table
temp_vec = character(nrow(train_X) + nrow(test_X))
preds_table <- data.frame(tagstring = Y[good_MS_mask,2],cluster = Y[good_MS_mask,3],label = Y[good_MS_mask,1],
                          label_nosen = temp_vec,
                          label_ran = temp_vec,
                          split = temp_vec, 
                          all = temp_vec,
                          mods = temp_vec,
                          sen = temp_vec,
                          modsp = temp_vec,
                          ran = temp_vec,
                          stringsAsFactors = F
                          )
preds_table[train_idx[good_MS_mask],6] <- 'train'
preds_table[!train_idx[good_MS_mask],6] <- 'test'

# tidy
save(preProcValues,file = '2020_03_06_3_preproc_model.RData')
rm(X,Y,preProcValues,test_X,train_X, temp_vec)


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
save(m_all,file = '2020_03_06_3_all_classes_svm_cluster_model.RData')

train_pred<-predict(m_all,train_X_trans)
test_pred<-predict(m_all,test_X_trans)
preds_table[train_idx[good_MS_mask],'all'] <- as.character(train_pred)
preds_table[!train_idx[good_MS_mask],'all'] <- as.character(test_pred)


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
rm(tmp_Y,drop_idx)

# Do same merge for outputs
pred_tmp_1 <- preds_table$all
mrg_idx_1 <- pred_tmp_1 == "SEN"
pred_tmp_1[mrg_idx_1] <- 'RAN'
preds_table$mods <- pred_tmp_1
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
save(m_sen,file = '2020_03_06_3_drop_sen_svm_cluster_model.RData')

train_pred<-predict(m_sen,train_X_trans)
test_pred<-predict(m_sen,test_X_trans)
preds_table[train_idx[good_MS_mask],'sen'] <- as.character(train_pred)
preds_table[!train_idx[good_MS_mask],'sen'] <- as.character(test_pred)


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

pred_tmp_1 <- preds_table$all
mrg_idx_1 <- pred_tmp_1 == "SEN" | pred_tmp_1 == "PUL"
pred_tmp_1[mrg_idx_1] <- 'RAN'
preds_table$modsp <- pred_tmp_1
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

set.seed(42)
m_ran <- e1071::best.svm(
        x=train_X_trans,y=factor(train_Y_modsp),
        class.weights=c(BEL = weights[1],
                        MAC = weights[2],
                        MIS = weights[3],
                        PAL = weights[4],
                        RAN = weights[5])
)
save(m_ran,file = '2020_03_06_3_drop_sen_pul_svm_cluster_model.RData')

train_pred<-predict(m_ran,train_X_trans)
test_pred<-predict(m_ran,test_X_trans)
preds_table[train_idx[good_MS_mask],'ran'] <- as.character(train_pred)
preds_table[!train_idx[good_MS_mask],'ran'] <- as.character(test_pred)


# tidy
rm(m_ran,train_pred,test_pred,weights)

################################################################################

############################# SAVE OUTPUT ######################################

write.csv(preds_table,'2020_03_06_3_svm_predictions_weights.csv')


########################### COMPUTE ACCURACIES #################################

train_accuracies <- numeric(5)
test_accuracies <- numeric(5)

i <- 1
        train_accuracies[i] <- sum(preds_table[train_idx[good_MS_mask],3]==preds_table[train_idx[good_MS_mask],i+6])/sum(train_idx[good_MS_mask])
        test_accuracies[i] <-  sum(preds_table[!train_idx[good_MS_mask],3]==preds_table[!train_idx[good_MS_mask],i+6])/(nrow(preds_table) - sum(train_idx[good_MS_mask]))

for(i in 2:3){
        train_accuracies[i] <- sum(preds_table[train_idx[good_MS_mask],4]==preds_table[train_idx[good_MS_mask],i+6])/sum(train_idx[good_MS_mask])
        test_accuracies[i] <-  sum(preds_table[!train_idx[good_MS_mask],4]==preds_table[!train_idx[good_MS_mask],i+6])/(nrow(preds_table) - sum(train_idx[good_MS_mask]))
}

for(i in 4:5){
        train_accuracies[i] <- sum(preds_table[train_idx[good_MS_mask],5]==preds_table[train_idx[good_MS_mask],i+6])/sum(train_idx[good_MS_mask])
        test_accuracies[i] <-  sum(preds_table[!train_idx[good_MS_mask],5]==preds_table[!train_idx[good_MS_mask],i+6])/(nrow(preds_table) - sum(train_idx[good_MS_mask]))
}


accuracy_data <- data.frame(data = c('all','drps','mods','drpsp','modsp'),
                            training = train_accuracies, test = test_accuracies)
write.csv(accuracy_data,'2020_03_06_3_overall_accuracies_weights.csv')
