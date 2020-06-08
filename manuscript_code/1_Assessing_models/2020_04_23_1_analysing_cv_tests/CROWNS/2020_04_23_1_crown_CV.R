# Compute CV statitics for the crown models


library(dplyr)
library(caret)

setwd("E:/SLICUAV_manusript_code/1_assessing_models/2020_04_23_1_analysing_cv_tests/CROWNS")

################################################################################
#                                                                              #
#                            GLMNET                                            #
#                                                                              #
################################################################################

accuracies <- read.csv('../../2020_04_21_1_crown_cv_tests/GLMNET/2020_04_21_GLMNET_accuracies_in_10fold_CV.csv')

glm_acc <- accuracies %>%
        dplyr::group_by(model,set) %>%
        dplyr::summarise(mean = mean(accuracy),SD = sd(accuracy))


################################################################################
#                                                                              #
#                            SVM                                               #
#                                                                              #
################################################################################

accuracies <- read.csv('../../2020_04_21_1_crown_cv_tests/SVM/2020_04_21_SVM_accuracies_in_10fold_CV.csv')

svm_acc <- accuracies %>%
        dplyr::group_by(model,set) %>%
        dplyr::summarise(mean = mean(accuracy),SD = sd(accuracy))


################################################################################
#                                                                              #
#                            RF                                                #
#                                                                              #
################################################################################

accuracies <- read.csv('../../2020_04_21_1_crown_cv_tests/RF/2020_04_21_RF_accuracies_in_10fold_CV.csv')

rf_acc <- accuracies %>%
        dplyr::group_by(model,set) %>%
        dplyr::summarise(mean = mean(accuracy),SD = sd(accuracy))


################################################################################

#                          collate and save

################################################################################

glm_acc$method <- 'glm'
svm_acc$method <- 'svm'
rf_acc$method <- 'rf'

acc_tbl <- rbind(glm_acc,svm_acc,rf_acc)
rm(glm_acc,svm_acc,rf_acc)

write.csv(acc_tbl,'2020_04_23_1_crown_cv_accuracy_tables.csv')