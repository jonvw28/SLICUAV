# Compute CV statitics for the crown models


library(dplyr)
library(caret)

setwd("E:/SLICUAV_manuscript_code/3_Landscape_mapping/2020_04_28_1_forward_prediction/STEP_4_2020_05_04_3_conf_mat")

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

temp_vec <- numeric(3)
temp_char <- character(3)
GLM_acc_tbl <- data.frame(model = temp_char,
                          accuracy = temp_vec, stringsAsFactors = F
)
rm(temp_char,temp_vec)

data <- read.csv('../../2020_04_23_2_train_model/2020_04_23_2_glm_predictions_all.csv')
preds_table[,'label_all'] <- data$label
preds_table[,'glm_all'] <- data$glm_all
GLM_acc_tbl[1,1] <- 'all'
GLM_acc_tbl[1,2] <- sum(data$glm_all == data$label)/nrow(data)

# GLM SEN HAD A TYPO IN THE SCRIPT BUT NOT USED ANYWAY


data <- read.csv('../../2020_04_23_2_train_model/2020_04_23_2_glm_predictions_ran.csv')
preds_table[,'label_ran'] <- data$label_ran
preds_table[,'glm_ran'] <- data$glm_ran
GLM_acc_tbl[3,1] <- 'ran'
GLM_acc_tbl[3,2] <- sum(data$glm_ran == data$label_ran)/nrow(data)

rm(data)
################################################################################
#                                                                              #
#                            SVM                                               #
#                                                                              #
################################################################################


temp_vec <- numeric(3)
temp_char <- character(3)
SVM_acc_tbl <- data.frame(model = temp_char,
                          accuracy = temp_vec, stringsAsFactors = F
)
rm(temp_char,temp_vec)

data <- read.csv('../../2020_04_23_2_train_model/2020_04_23_2_svm_predictions_all.csv')
preds_table[,'svm_all'] <- data$svm_all
SVM_acc_tbl[1,1] <- 'all'
SVM_acc_tbl[1,2] <- sum(data$svm_all == data$label)/nrow(data)

data <- read.csv('../../2020_04_23_2_train_model/2020_04_23_2_svm_predictions_sen.csv')
preds_table[,'label_sen'] <- data$label_sen
preds_table[,'svm_sen'] <- data$svm_sen
SVM_acc_tbl[2,1] <- 'sen'
SVM_acc_tbl[2,2] <- sum(data$svm_sen == data$label_sen)/nrow(data)

data <- read.csv('../../2020_04_23_2_train_model/2020_04_23_2_svm_predictions_ran.csv')
preds_table[,'svm_ran'] <- data$svm_ran
SVM_acc_tbl[3,1] <- 'ran'
SVM_acc_tbl[3,2] <- sum(data$svm_ran == data$label_ran)/nrow(data)

rm(data)

################################################################################
#                                                                              #
#                            RF                                                #
#                                                                              #
################################################################################


temp_vec <- numeric(3)
temp_char <- character(3)
RF_acc_tbl <- data.frame(model = temp_char,
                          accuracy = temp_vec, stringsAsFactors = F
)
rm(temp_char,temp_vec)

data <- read.csv('../../2020_04_23_2_train_model/2020_04_23_2_rf_predictions_all.csv')
preds_table[,'rf_all'] <- data$rf_all
RF_acc_tbl[1,1] <- 'all'
RF_acc_tbl[1,2] <- sum(data$rf_all == data$label)/nrow(data)

data <- read.csv('../../2020_04_23_2_train_model/2020_04_23_2_rf_predictions_sen.csv')
preds_table[,'rf_sen'] <- data$rf_sen
RF_acc_tbl[2,1] <- 'sen'
RF_acc_tbl[2,2] <- sum(data$rf_sen == data$label_sen)/nrow(data)

data <- read.csv('../../2020_04_23_2_train_model/2020_04_23_2_rf_predictions_ran.csv')
preds_table[,'rf_ran'] <- data$rf_ran
RF_acc_tbl[3,1] <- 'ran'
RF_acc_tbl[3,2] <- sum(data$rf_ran == data$label_ran)/nrow(data)

rm(data)

################################################################################

#                          collate and save

################################################################################

GLM_acc_tbl$method <- 'glm'
SVM_acc_tbl$method <- 'svm'
RF_acc_tbl$method <- 'rf'

acc_tbl <- rbind(GLM_acc_tbl,SVM_acc_tbl,RF_acc_tbl)
rm(GLM_acc_tbl,SVM_acc_tbl,RF_acc_tbl)

write.csv(acc_tbl,'2020_05_04_1_seg_accuracy_tables.csv')

################################################################################

#                       Confusion Matrices

################################################################################

svm_all_mat <- caret::confusionMatrix(factor(preds_table$svm_all),factor(preds_table$label_all),mode='prec_recall')
svm_ran_mat <- caret::confusionMatrix(factor(preds_table$svm_ran),factor(preds_table$label_ran),mode='prec_recall')

write.csv(svm_all_mat[[2]],'2020_05_04_SVM_all_classes_matrix.csv')
write.csv(svm_all_mat[[4]],'2020_05_04_SVM_all_classes_stats.csv')

write.csv(svm_ran_mat[[2]],'2020_05_04_SVM_no_sen_pul_matrix.csv')
write.csv(svm_ran_mat[[4]],'2020_05_04_SVM_no_sen_pul_stats.csv')