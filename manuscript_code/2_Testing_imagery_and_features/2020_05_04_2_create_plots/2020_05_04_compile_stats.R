# 
library(dplyr)

setwd("E:/SLICUAV_manuscript_code/2_testing_imagery_and_features/2020_05_04_2_create_plots")


# get all features all imagery
all_all_preds <- read.csv('../2020_03_06_3_all_included_model/2020_03_06_3_svm_predictions_weights.csv',stringsAsFactors = F)
all_all_preds <- all_all_preds[,-1]
all_all_preds <- all_all_preds %>%
        dplyr::select(-c(mods,modsp)) %>%
        dplyr::rename(all_all_all = all, all_all_sen = sen, all_all_ran = ran)

# get band features all imagery
band_all_preds <- read.csv('../2020_05_01_2_multiplexing_imagery_features/2020_05_01_band_feats_all_imagery_svm_predictions.csv',stringsAsFactors = F)
band_all_preds <- band_all_preds[,-1]
band_all_preds <- band_all_preds %>%
        dplyr::select(-c(band_feats_all_imagery_mods,band_feats_all_imagery_modsp)) %>%
        dplyr::rename(band_all_all = band_feats_all_imagery_all, 
                      band_all_sen = band_feats_all_imagery_sen, 
                      band_all_ran = band_feats_all_imagery_ran,
                      tagstring = ttagstring)

# get text features all imagery
text_all_preds <- read.csv('../2020_05_01_2_multiplexing_imagery_features/2020_05_01_text_feats_all_imagery_svm_predictions.csv',stringsAsFactors = F)
text_all_preds <- text_all_preds[,-1]
text_all_preds <- text_all_preds %>%
        dplyr::select(-c(text_feats_all_imagery_mods,text_feats_all_imagery_modsp)) %>%
        dplyr::rename(text_all_all = text_feats_all_imagery_all, 
                      text_all_sen = text_feats_all_imagery_sen, 
                      text_all_ran = text_feats_all_imagery_ran,
                      tagstring = ttagstring)


# get all features noDSM imagery
all_noDSM_preds <- read.csv('../2020_05_01_2_multiplexing_imagery_features/2020_05_01_all_feats_noDSM_imagery_svm_predictions.csv',stringsAsFactors = F)
all_noDSM_preds <- all_noDSM_preds[,-1]
all_noDSM_preds <- all_noDSM_preds %>%
        dplyr::select(-c(all_feats_noDSM_imagery_mods,all_feats_noDSM_imagery_modsp)) %>%
        dplyr::rename(all_noDSM_all = all_feats_noDSM_imagery_all, 
                      all_noDSM_sen = all_feats_noDSM_imagery_sen, 
                      all_noDSM_ran = all_feats_noDSM_imagery_ran,
                      tagstring = ttagstring)

# get band features noDSM imagery
band_noDSM_preds <- read.csv('../2020_05_01_2_multiplexing_imagery_features/2020_05_01_band_feats_noDSM_imagery_svm_predictions.csv',stringsAsFactors = F)
band_noDSM_preds <- band_noDSM_preds[,-1]
band_noDSM_preds <- band_noDSM_preds %>%
        dplyr::select(-c(band_feats_noDSM_imagery_mods,band_feats_noDSM_imagery_modsp)) %>%
        dplyr::rename(band_noDSM_all = band_feats_noDSM_imagery_all, 
                      band_noDSM_sen = band_feats_noDSM_imagery_sen, 
                      band_noDSM_ran = band_feats_noDSM_imagery_ran,
                      tagstring = ttagstring)

# get band features noDSM imagery
text_noDSM_preds <- read.csv('../2020_05_01_2_multiplexing_imagery_features/2020_05_01_text_feats_noDSM_imagery_svm_predictions.csv',stringsAsFactors = F)
text_noDSM_preds <- text_noDSM_preds[,-1]
text_noDSM_preds <- text_noDSM_preds %>%
        dplyr::select(-c(text_feats_noDSM_imagery_mods,text_feats_noDSM_imagery_modsp)) %>%
        dplyr::rename(text_noDSM_all = text_feats_noDSM_imagery_all, 
                      text_noDSM_sen = text_feats_noDSM_imagery_sen, 
                      text_noDSM_ran = text_feats_noDSM_imagery_ran,
                      tagstring = ttagstring)

# get all features noMS imagery
all_noMS_preds <- read.csv('../2020_05_01_2_multiplexing_imagery_features/2020_05_01_all_feats_noMS_imagery_svm_predictions.csv',stringsAsFactors = F)
all_noMS_preds <- all_noMS_preds[,-1]
all_noMS_preds <- all_noMS_preds %>%
        dplyr::select(-c(all_feats_noMS_imagery_mods,all_feats_noMS_imagery_modsp)) %>%
        dplyr::rename(all_noMS_all = all_feats_noMS_imagery_all, 
                      all_noMS_sen = all_feats_noMS_imagery_sen, 
                      all_noMS_ran = all_feats_noMS_imagery_ran,
                      tagstring = ttagstring)

# get band features noMS imagery
band_noMS_preds <- read.csv('../2020_05_01_2_multiplexing_imagery_features/2020_05_01_band_feats_noMS_imagery_svm_predictions.csv',stringsAsFactors = F)
band_noMS_preds <- band_noMS_preds[,-1]
band_noMS_preds <- band_noMS_preds %>%
        dplyr::select(-c(band_feats_noMS_imagery_mods,band_feats_noMS_imagery_modsp)) %>%
        dplyr::rename(band_noMS_all = band_feats_noMS_imagery_all, 
                      band_noMS_sen = band_feats_noMS_imagery_sen, 
                      band_noMS_ran = band_feats_noMS_imagery_ran,
                      tagstring = ttagstring)

# get band features noMS imagery
text_noMS_preds <- read.csv('../2020_05_01_2_multiplexing_imagery_features/2020_05_01_text_feats_noMS_imagery_svm_predictions.csv',stringsAsFactors = F)
text_noMS_preds <- text_noMS_preds[,-1]
text_noMS_preds <- text_noMS_preds %>%
        dplyr::select(-c(text_feats_noMS_imagery_mods,text_feats_noMS_imagery_modsp)) %>%
        dplyr::rename(text_noMS_all = text_feats_noMS_imagery_all, 
                      text_noMS_sen = text_feats_noMS_imagery_sen, 
                      text_noMS_ran = text_feats_noMS_imagery_ran,
                      tagstring = ttagstring)

# get all features noRGB imagery
all_noRGB_preds <- read.csv('../2020_05_01_2_multiplexing_imagery_features/2020_05_01_all_feats_noRGB_imagery_svm_predictions.csv',stringsAsFactors = F)
all_noRGB_preds <- all_noRGB_preds[,-1]
all_noRGB_preds <- all_noRGB_preds %>%
        dplyr::select(-c(all_feats_noRGB_imagery_mods,all_feats_noRGB_imagery_modsp)) %>%
        dplyr::rename(all_noRGB_all = all_feats_noRGB_imagery_all, 
                      all_noRGB_sen = all_feats_noRGB_imagery_sen, 
                      all_noRGB_ran = all_feats_noRGB_imagery_ran,
                      tagstring = ttagstring)

# get band features noRGB imagery
band_noRGB_preds <- read.csv('../2020_05_01_2_multiplexing_imagery_features/2020_05_01_band_feats_noRGB_imagery_svm_predictions.csv',stringsAsFactors = F)
band_noRGB_preds <- band_noRGB_preds[,-1]
band_noRGB_preds <- band_noRGB_preds %>%
        dplyr::select(-c(band_feats_noRGB_imagery_mods,band_feats_noRGB_imagery_modsp)) %>%
        dplyr::rename(band_noRGB_all = band_feats_noRGB_imagery_all, 
                      band_noRGB_sen = band_feats_noRGB_imagery_sen, 
                      band_noRGB_ran = band_feats_noRGB_imagery_ran,
                      tagstring = ttagstring)

# get band features noRGB imagery
text_noRGB_preds <- read.csv('../2020_05_01_2_multiplexing_imagery_features/2020_05_01_text_feats_noRGB_imagery_svm_predictions.csv',stringsAsFactors = F)
text_noRGB_preds <- text_noRGB_preds[,-1]
text_noRGB_preds <- text_noRGB_preds %>%
        dplyr::select(-c(text_feats_noRGB_imagery_mods,text_feats_noRGB_imagery_modsp)) %>%
        dplyr::rename(text_noRGB_all = text_feats_noRGB_imagery_all, 
                      text_noRGB_sen = text_feats_noRGB_imagery_sen, 
                      text_noRGB_ran = text_feats_noRGB_imagery_ran,
                      tagstring = ttagstring)

# get all features noDSM imagery
all_DSM_preds <- read.csv('../2020_05_01_2_multiplexing_imagery_features/2020_05_01_all_feats_DSM_imagery_svm_predictions.csv',stringsAsFactors = F)
all_DSM_preds <- all_DSM_preds[,-1]
all_DSM_preds <- all_DSM_preds %>%
        dplyr::select(-c(all_feats_DSM_imagery_mods,all_feats_DSM_imagery_modsp)) %>%
        dplyr::rename(all_DSM_all = all_feats_DSM_imagery_all, 
                      all_DSM_sen = all_feats_DSM_imagery_sen, 
                      all_DSM_ran = all_feats_DSM_imagery_ran,
                      tagstring = ttagstring)

# get band features noDSM imagery
band_DSM_preds <- read.csv('../2020_05_01_2_multiplexing_imagery_features/2020_05_01_band_feats_DSM_imagery_svm_predictions.csv',stringsAsFactors = F)
band_DSM_preds <- band_DSM_preds[,-1]
band_DSM_preds <- band_DSM_preds %>%
        dplyr::select(-c(band_feats_DSM_imagery_mods,band_feats_DSM_imagery_modsp)) %>%
        dplyr::rename(band_DSM_all = band_feats_DSM_imagery_all, 
                      band_DSM_sen = band_feats_DSM_imagery_sen, 
                      band_DSM_ran = band_feats_DSM_imagery_ran,
                      tagstring = ttagstring)

# get band features noDSM imagery
text_DSM_preds <- read.csv('../2020_05_01_2_multiplexing_imagery_features/2020_05_01_text_feats_DSM_imagery_svm_predictions.csv',stringsAsFactors = F)
text_DSM_preds <- text_DSM_preds[,-1]
text_DSM_preds <- text_DSM_preds %>%
        dplyr::select(-c(text_feats_DSM_imagery_mods,text_feats_DSM_imagery_modsp)) %>%
        dplyr::rename(text_DSM_all = text_feats_DSM_imagery_all, 
                      text_DSM_sen = text_feats_DSM_imagery_sen, 
                      text_DSM_ran = text_feats_DSM_imagery_ran,
                      tagstring = ttagstring)

# get all features noMS imagery
all_MS_preds <- read.csv('../2020_05_01_2_multiplexing_imagery_features/2020_05_01_all_feats_MS_imagery_svm_predictions.csv',stringsAsFactors = F)
all_MS_preds <- all_MS_preds[,-1]
all_MS_preds <- all_MS_preds %>%
        dplyr::select(-c(all_feats_MS_imagery_mods,all_feats_MS_imagery_modsp)) %>%
        dplyr::rename(all_MS_all = all_feats_MS_imagery_all, 
                      all_MS_sen = all_feats_MS_imagery_sen, 
                      all_MS_ran = all_feats_MS_imagery_ran,
                      tagstring = ttagstring)

# get band features noMS imagery
band_MS_preds <- read.csv('../2020_05_01_2_multiplexing_imagery_features/2020_05_01_band_feats_MS_imagery_svm_predictions.csv',stringsAsFactors = F)
band_MS_preds <- band_MS_preds[,-1]
band_MS_preds <- band_MS_preds %>%
        dplyr::select(-c(band_feats_MS_imagery_mods,band_feats_MS_imagery_modsp)) %>%
        dplyr::rename(band_MS_all = band_feats_MS_imagery_all, 
                      band_MS_sen = band_feats_MS_imagery_sen, 
                      band_MS_ran = band_feats_MS_imagery_ran,
                      tagstring = ttagstring)

# get band features noMS imagery
text_MS_preds <- read.csv('../2020_05_01_2_multiplexing_imagery_features/2020_05_01_text_feats_MS_imagery_svm_predictions.csv',stringsAsFactors = F)
text_MS_preds <- text_MS_preds[,-1]
text_MS_preds <- text_MS_preds %>%
        dplyr::select(-c(text_feats_MS_imagery_mods,text_feats_MS_imagery_modsp)) %>%
        dplyr::rename(text_MS_all = text_feats_MS_imagery_all, 
                      text_MS_sen = text_feats_MS_imagery_sen, 
                      text_MS_ran = text_feats_MS_imagery_ran,
                      tagstring = ttagstring)

# get all features noRGB imagery
all_RGB_preds <- read.csv('../2020_05_01_2_multiplexing_imagery_features/2020_05_01_all_feats_RGB_imagery_svm_predictions.csv',stringsAsFactors = F)
all_RGB_preds <- all_RGB_preds[,-1]
all_RGB_preds <- all_RGB_preds %>%
        dplyr::select(-c(all_feats_RGB_imagery_mods,all_feats_RGB_imagery_modsp)) %>%
        dplyr::rename(all_RGB_all = all_feats_RGB_imagery_all, 
                      all_RGB_sen = all_feats_RGB_imagery_sen, 
                      all_RGB_ran = all_feats_RGB_imagery_ran,
                      tagstring = ttagstring)

# get band features noRGB imagery
band_RGB_preds <- read.csv('../2020_05_01_2_multiplexing_imagery_features/2020_05_01_band_feats_RGB_imagery_svm_predictions.csv',stringsAsFactors = F)
band_RGB_preds <- band_RGB_preds[,-1]
band_RGB_preds <- band_RGB_preds %>%
        dplyr::select(-c(band_feats_RGB_imagery_mods,band_feats_RGB_imagery_modsp)) %>%
        dplyr::rename(band_RGB_all = band_feats_RGB_imagery_all, 
                      band_RGB_sen = band_feats_RGB_imagery_sen, 
                      band_RGB_ran = band_feats_RGB_imagery_ran,
                      tagstring = ttagstring)

# get band features noRGB imagery
text_RGB_preds <- read.csv('../2020_05_01_2_multiplexing_imagery_features/2020_05_01_text_feats_RGB_imagery_svm_predictions.csv',stringsAsFactors = F)
text_RGB_preds <- text_RGB_preds[,-1]
text_RGB_preds <- text_RGB_preds %>%
        dplyr::select(-c(text_feats_RGB_imagery_mods,text_feats_RGB_imagery_modsp)) %>%
        dplyr::rename(text_RGB_all = text_feats_RGB_imagery_all, 
                      text_RGB_sen = text_feats_RGB_imagery_sen, 
                      text_RGB_ran = text_feats_RGB_imagery_ran,
                      tagstring = ttagstring)









preds <- merge(all_all_preds,all_DSM_preds) %>%
        merge(all_MS_preds) %>%
        merge(all_noDSM_preds) %>%
        merge(all_noMS_preds) %>%
        merge(all_noRGB_preds) %>%
        merge(all_RGB_preds) %>%
        merge(band_all_preds) %>%
        merge(band_DSM_preds) %>%
        merge(band_MS_preds) %>%
        merge(band_noDSM_preds) %>%
        merge(band_noMS_preds) %>%
        merge(band_noRGB_preds) %>%
        merge(band_RGB_preds) %>%
        merge(text_all_preds) %>%
        merge(text_DSM_preds) %>%
        merge(text_MS_preds) %>%
        merge(text_noDSM_preds) %>%
        merge(text_noMS_preds) %>%
        merge(text_noRGB_preds) %>%
        merge(text_RGB_preds)

# Compute the number of models
n_info <- 6
n_model <- ncol(preds) - n_info

# summary set up
temp_char <- character(length = n_model)
temp_num <- numeric(length = n_model)
temp_log <- logical(length = n_model)
acc_table <- data.frame(model = temp_char, form = temp_char,
                        imagery = temp_char, feats = temp_char,
                        RGB_data = temp_log, MS_data = temp_log,
                        DSM_data = temp_log, band_data = temp_log,
                        text_data = temp_log, train_acc = temp_num,
                        test_acc = temp_num, stringsAsFactors = F)
rm(temp_char,temp_log,temp_num)
ticker <- 1


train_preds <- preds %>%
        dplyr::filter(split == 'train')
test_preds <- preds %>%
        dplyr::filter(split == 'test')
for(i in (n_info+1):ncol(train_preds)){
        split_name <- strsplit(names(train_preds)[i],'_')
        form <- split_name[[1]][3]
        acc_table[ticker,'model'] <- names(train_preds)[i]
        acc_table[ticker,'form'] <- form
        imagery <- split_name[[1]][2]
        acc_table[ticker,'imagery'] <- imagery     
        if(imagery == 'noRGB'){
                acc_table[ticker,c('MS_data','DSM_data')] <- TRUE
        } else if(imagery == 'noMS'){
                acc_table[ticker,c('RGB_data','DSM_data')] <- TRUE
        } else if(imagery == 'noDSM'){
                acc_table[ticker,c('RGB_data','MS_data')] <- TRUE
        } else if(imagery == 'RGB'){
                acc_table[ticker,'RGB_data'] <- TRUE
        } else if(imagery == 'MS'){
                acc_table[ticker,'MS_data'] <- TRUE
        } else if(imagery == 'DSM'){
                acc_table[ticker,'DSM_data'] <- TRUE
        } else {
                acc_table[ticker,c('MS_data','RGB_data','DSM_data')]
        }
        
        
        feats <- split_name[[1]][1]
        acc_table[ticker,'feats'] <- feats
        if(feats == 'all'){
                acc_table[ticker,c('band_data','text_data')] <- TRUE
        } else if(feats == 'band'){
                acc_table[ticker,'band_data'] <- TRUE
        } else {
                acc_table[ticker,'text_data'] <- TRUE
        }
        
        
        
        if(form == 'all'){
                lbl_idx <- 'label'
        } else if(form == 'sen' || form == 'mods'){
                lbl_idx <- 'label_nosen'
        } else {
                lbl_idx <- 'label_ran'
        }
        train <- sum(train_preds[,i]==train_preds[,lbl_idx])/nrow(train_preds)
        test <- sum(test_preds[,i]==test_preds[,lbl_idx])/nrow(test_preds)
        acc_table[ticker,'train_acc'] <- train
        acc_table[ticker,'test_acc'] <- test

        ticker = ticker + 1
}
write.csv(acc_table,'2020_05_04_summary_of_accuracies.csv')
