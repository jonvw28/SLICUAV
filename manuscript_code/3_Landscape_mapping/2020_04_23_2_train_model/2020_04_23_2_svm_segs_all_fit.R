# Fit SVM model to the crown segment data using all classesall

library(e1071)
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
preds_table <- data.frame(tagstring = Y[,2],cluster = Y[,3],label = Y[,1],
						  svm_all = temp_vec,
                          stringsAsFactors = F
                          )
rm(temp_vec)


tic()

# center and scale
preProcValues <- caret::preProcess(X, method = c('center','scale'))
X_trans <- predict(preProcValues, X)
pptime <- toc()
write.csv(pptime$toc-pptime$tic,'2020_04_23_2_preproc_time_SVM_all.csv')

# Save preprocessing machinery and also the variables which were used
save(preProcValues,file = '2020_04_23_2_preproc_time_SVM_all.RData')

# tidy
rm(preProcValues,X)


################################################################################
#                                                                              #
#                               FIT MODEL                                      #
#                                                                              #
################################################################################

#Use weights to get fairer learning

N <- nrow(X_trans)
N_bel <- sum(Y[,1]=='BEL')
N_mac <- sum(Y[,1]=='MAC')
N_mis <- sum(Y[,1]=='MIS')
N_pal <- sum(Y[,1]=='PAL')
N_pul <- sum(Y[,1]=='PUL')
N_ran <- sum(Y[,1]=='RAN')
N_sen <- sum(Y[,1]=='SEN')

weights <- numeric(7)
weights[1] <- N/(N_bel*7)
weights[2] <- N/(N_mac*7)
weights[3] <- N/(N_mis*7)
weights[4] <- N/(N_pal*7)
weights[5] <- N/(N_pul*7)
weights[6] <- N/(N_ran*7)
weights[7] <- N/(N_sen*7)

#normalise to sum to 1
weights <- weights/sum(weights)

rm(N,N_bel,N_mac,N_mis,N_pal,N_ran,N_pul,N_sen)

###################### Fit svm model on full data ############################
tic()
set.seed(42)
m_all <- e1071::best.svm(
                    x=X_trans,y=factor(Y[,1]),
                    class.weights=c(BEL = weights[1],
                                    MAC = weights[2],
                                    MIS = weights[3],
                                    PAL = weights[4],
                                    PUL = weights[5],
                                    RAN = weights[6],
                                    SEN = weights[7]),
                    type = 'C'
                    )
save(m_all,file='2020_04_23_2_all_classes_svm_model.RData')

preds<-predict(m_all,X_trans)
preds_table$svm_all <- as.character(preds)
svm_all_time <- toc()
write.csv(svm_all_time$toc-svm_all_time$tic,'2020_04_23_2_all_classes_svm_time.csv')

write.csv(preds_table,'2020_04_23_2_svm_predictions_all.csv')

