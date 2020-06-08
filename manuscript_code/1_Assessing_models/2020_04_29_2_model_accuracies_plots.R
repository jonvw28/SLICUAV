# Load accuracy table and create some plots

# Do:
#  all   -    Data as collected
#  mods  -    Post-process to combine Sendok and Random
#  sen   -    Merge Sendok and Random
#  modsp -    Post-process to combine Pulai, Sendok and Random
#  ran   -    Merge Pulai, Sendok and Random
#

library(plyr)
library(dplyr)
library(ggplot2)
library(reshape)

setwd("E:/SLICUAV_manusript_code/1_assessing_models")

################################################################################
#                                                                              #
#                            CROWNS                                            #                      
#                                                                              #
################################################################################

crown_acc <- read.csv('../2020_04_23_1_analysing_cv_tests/CROWNS/2020_04_23_1_crown_cv_accuracy_tables.csv',stringsAsFactors = F,header=T)

# Plot crown results for using all variables

crown_acc <- crown_acc %>%
        dplyr::select(c('method','model','set','mean','SD')) %>%
        reshape::melt(id=c('model','set','method')) %>%
        dplyr::filter(variable == 'mean') %>%
        dplyr::select(c('method','model','set','value')) %>%
        dplyr::rename(Set = set, Version = model, Model = method)

crown_acc$value <- crown_acc$value*100


crown_acc$Set <- factor(plyr::revalue(crown_acc$Set, c('train'='Training','test'='Test')),levels=c('Training','Test'))
crown_acc$Version <- factor(plyr::revalue(crown_acc$Version, 
                                               c('all'='All Species',
                                                 'sen_dropped'='No E. malaccense',
                                                 'sen_ran_dropped'='No E. malaccense and A. scholaris')
                                        ),
                                 levels=c('All Species',
                                          'No E. malaccense',
                                          'No E. malaccense and A. scholaris')
)

crown_acc$Model<- factor(plyr::revalue(crown_acc$Model, 
                                               c('glm' = 'Lasso Regression',
                                                 'svm' = 'Support Vector Machine',
                                                 'rf' = 'Random Forest')
),
levels=c('Lasso Regression',
         'Support Vector Machine',
         'Random Forest')
)

# create facet labels
lab_data <- data.frame(x=rep(0.55,3),
                       y=rep(100,3),
                       Version = factor(c('All Species',
                                          'No E. malaccense',
                                          'No E. malaccense and A. scholaris'),
                                      levels=c('All Species',
                                               'No E. malaccense',
                                                'No E. malaccense and A. scholaris')),
                       labs = c('(a)','(b)','(c)'),
                       Model = 'Lasso Regression'
                                                                
)


# panel - version, x - train/test, bars - model
fig1 <- ggplot(crown_acc, aes(x=Set, y=value, fill=Model, label=sprintf('%0.1f',signif(value,3))))+
        geom_bar(position='dodge',stat='identity')+
        facet_grid(. ~ Version)+
        labs(y='Accuracy (%)')+
        geom_text(vjust=1.2,color="white", size=3.0, position = position_dodge(0.9)) +
        scale_fill_manual(values=c('#a6cee3','#1f78b4','#b2df8a')) +
        geom_text(aes(x,y,label=labs,group=NULL),data=lab_data)
ggsave('2020_04_29_crowns_accuracy_plot_10fold.pdf',fig1,width=9,height=4.5)

################################################################################
#                                                                              #
#                            SEGS                                              #                      
#                                                                              #
################################################################################

seg_acc <- read.csv('../2020_04_23_1_analysing_cv_tests/SEGS/2020_04_23_1_seg_cv_accuracy_tables.csv',stringsAsFactors = F,header=T)

# Plot crown results for using all variables

seg_acc <- seg_acc %>%
        dplyr::select(c('method','model','set','mean','SD')) %>%
        reshape::melt(id=c('model','set','method')) %>%
        dplyr::filter(variable == 'mean') %>%
        dplyr::select(c('method','model','set','value')) %>%
        dplyr::rename(Set = set, Version = model, Model = method)

seg_acc$value <- seg_acc$value*100


seg_acc$Set <- factor(plyr::revalue(seg_acc$Set, c('train'='Training','test'='Test')),levels=c('Training','Test'))
seg_acc$Version <- factor(plyr::revalue(seg_acc$Version, 
                                          c('all'='All Species',
                                            'sen'='No E. malaccense',
                                            'ran'='No E. malaccense and A. scholaris')
),
levels=c('All Species',
         'No E. malaccense',
         'No E. malaccense and A. scholaris')
)

seg_acc$Model<- factor(plyr::revalue(seg_acc$Model, 
                                       c('glm' = 'Lasso Regression',
                                         'svm' = 'Support Vector Machine',
                                         'rf' = 'Random Forest')
),
levels=c('Lasso Regression',
         'Support Vector Machine',
         'Random Forest')
)

# create facet labels
lab_data <- data.frame(x=rep(0.55,3),
                       y=rep(100,3),
                       Version = factor(c('All Species',
                                          'No E. malaccense',
                                          'No E. malaccense and A. scholaris'),
                                        levels=c('All Species',
                                                 'No E. malaccense',
                                                 'No E. malaccense and A. scholaris')),
                       labs = c('(a)','(b)','(c)'),
                       Model = 'Lasso Regression'
                       
)


# panel - version, x - train/test, bars - model
fig2 <- ggplot(seg_acc, aes(x=Set, y=value, fill=Model, label=sprintf('%0.1f',signif(value,3))))+
        geom_bar(position='dodge',stat='identity')+
        facet_grid(. ~ Version)+
        labs(y='Accuracy (%)')+
        geom_text(vjust=1.2,color="white", size=3.0, position = position_dodge(0.9)) +
        scale_fill_manual(values=c('#a6cee3','#1f78b4','#b2df8a')) +
        geom_text(aes(x,y,label=labs,group=NULL),data=lab_data)
ggsave('2020_04_29_segs_accuracy_plot_10fold.pdf',fig2,width=9,height=4.5)