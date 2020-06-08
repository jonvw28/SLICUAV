# Fit lasso model to the crown segment data using all classesall

library(glmnet)
library(tictoc)
library(caret)

setwd('E:/SLICUAV_manuscript_code/3_Landscape_mapping/2020_04_23_2_train_model')

N_max <- 25



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
						  glm_ran = temp_vec,
                          stringsAsFactors = F
                          )
rm(temp_vec)


tic()

# center and scale
preProcValues <- caret::preProcess(X, method = c('center','scale'))
X_trans <- predict(preProcValues, X)
pptime <- toc()
write.csv(pptime$toc-pptime$tic,'2020_04_23_2_preproc_time_GLM_ran.csv')

# Save preprocessing machinery and also the variables which were used
save(preProcValues,file = '2020_04_23_2_preproc_time_GLM_ran.RData')

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

weights <- numeric(N)
weights[preds_table$label_ran == 'BEL'] <- N/(N_bel*5)
weights[preds_table$label_ran == 'MAC'] <- N/(N_mac*5)
weights[preds_table$label_ran == 'MIS'] <- N/(N_mis*5)
weights[preds_table$label_ran == 'PAL'] <- N/(N_pal*5)
weights[preds_table$label_ran == 'RAN'] <- N/(N_ran*5)

#normalise to sum to 1
weights <- weights/sum(weights)

rm(N,N_bel,N_mac,N_mis,N_pal,N_ran)

###################### Fit glm model on full data ############################
tic()
set.seed(42)
m_ran <- glmnet::glmnet(X_trans, preds_table$label_ran, weights = weights, alpha = 1, family="multinomial", 
                          type.multinomial = "grouped")
save(m_ran,file='2020_04_23_2_no_sen_pul_classes_glm_model.RData')

# Pick lambda that keeps at most 20 predictors
mdl_idx <- max(which(m_ran$df<=N_max))
mdl_lam <- m_ran$lambda[mdl_idx]

preds <- predict(m_ran, newx = X_trans, s = mdl_lam, type = "class")
preds_table$glm_ran <- as.character(preds)
glm_ran_time <- toc()
write.csv(glm_ran_time$toc-glm_ran_time$tic,'2020_04_23_2_no_sen_pul_classes_glm_time.csv')

write.csv(preds_table,'2020_04_23_2_glm_predictions_ran.csv')

