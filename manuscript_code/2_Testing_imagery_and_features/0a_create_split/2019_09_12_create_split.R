# Adapted code, which now just splits crowns into a 75/25% training and test
# split

library(caret)


setwd("E:/SLICUAV_manuscript_code/2_Testing_imagery_and_features/0a_create_split")

################################################################################
#                                                                              #
#                            PRE-PROCESS AND SPLIT DATA                        #                      
#                                                                              #
################################################################################

# Includes 3/4 trianing, 1/4 test split

# read features and labels
X <- as.matrix(read.csv('../../1_Assessing_models/2019_09_12_5_generate_crown_features/2019_09_12_spectral_texture_dsm_features.csv',header=F))
Y <- read.csv('../../1_Assessing_models/2019_09_12_5_generate_crown_features/2019_09_12_cluster_labels.csv',header=F)

set.seed(42,sample.kind = 'Rounding') # since the original analysis was done in R 3.4 and so the behaviour of sample now needs to be specified (since R 3.6)
# train test split
train_idx <- caret::createDataPartition(Y[,1], p=0.75, list=F)


# set up output table
temp_vec = character(nrow(X))
preds_table <- data.frame(tagstring = Y[,2],label = Y[,1],
                          split = temp_vec, 
                          stringsAsFactors = F
                          )
preds_table[train_idx,3] <- 'train'
preds_table[-train_idx,3] <- 'test'

# tidy
rm(X,Y,temp_vec)

write.csv(preds_table,'2019_09_12_train_idx.csv')