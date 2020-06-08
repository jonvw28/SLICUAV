# Fit rf model to the crown segment data using all classesall

library(randomForest)
library(tictoc)
library(caret)

setwd('E:/SLICUAV_manuscript_code/3_Landscape_mapping/2020_04_23_2_train_model')


################################################################################
#                                                                              #
#                            PRE-PROCESS DATA                                  #
#                                                                              #
################################################################################

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

# set up output table
temp_vec = character(nrow(X))
preds_table <- data.frame(tagstring = Y[,2],cluster = Y[,3],label_ran = Y[,1],
						  rf_ran = temp_vec,
                          stringsAsFactors = F
                          )
rm(temp_vec)


tic()

# center and scale
preProcValues <- caret::preProcess(X, method = c('center','scale'))
X_trans <- predict(preProcValues, X)
pptime <- toc()
write.csv(pptime$toc-pptime$tic,'2020_04_23_2_preproc_time_rf_ran.csv')

# Save preprocessing machinery and also the variables which were used
save(preProcValues,file = '2020_04_23_2_preproc_time_rf_ran.RData')

# tidy
rm(preProcValues,X)

################################################################################
#                                                                              #
#                       MERGE SEN AND RAN ON OUTPUT                            #
#                                                                              #
################################################################################
# Merge manual Sendok label into Random

drop_idx <- preds_table$label_ran == "SEN" | preds_table$label_ran =='PUL'
tmp_Y <- preds_table$label_ran
tmp_Y[drop_idx] <- 'RAN'
preds_table$label_ran <- tmp_Y
rm(tmp_Y)

################################################################################
#                                                                              #
#                               FIT MODEL                                      #
#                                                                              #
################################################################################

#Use weights to get fairer learning

N <- nrow(X_trans)
N_bel <- sum(preds_table$label_ran=='BEL')
N_mac <- sum(preds_table$label_ran=='MAC')
N_mis <- sum(preds_table$label_ran=='MIS')
N_pal <- sum(preds_table$label_ran=='PAL')
N_ran <- sum(preds_table$label_ran=='RAN')

weights <- numeric(5)
weights[1] <- N/(N_bel*5)
weights[2] <- N/(N_mac*5)
weights[3] <- N/(N_mis*5)
weights[4] <- N/(N_pal*5)
weights[5] <- N/(N_ran*5)

#normalise to sum to 1
weights <- weights/sum(weights)

rm(N,N_bel,N_mac,N_mis,N_pal,N_ran)

###################### Fit rf model on full data ############################
tic()
set.seed(42)
m_ran <- randomForest::randomForest(
        x=X_trans,y=factor(preds_table$label_ran),importance=TRUE,
        classwt = weights)
save(m_ran,file='2020_04_23_2_no_sen_pul_classes_rf_model.RData')

preds<-predict(m_ran,X_trans)
preds_table$rf_ran <- as.character(preds)
rf_ran_time <- toc()
write.csv(rf_ran_time$toc-rf_ran_time$tic,'2020_04_23_2_no_sen_pul_classes_rf_time.csv')

write.csv(preds_table,'2020_04_23_2_rf_predictions_ran.csv')

