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
library(cowplot)

setwd("E:/SLICUAV_manuscript_code/2_testing_imagery_and_features/2020_05_04_2_create_plots")
################################################################################
#                                                                              #
#                            PRE-PROCESS AND SPLIT DATA                        #                      
#                                                                              #
################################################################################

acc_data <- read.csv('2020_05_04_summary_of_accuracies.csv',stringsAsFactors = T,header=T)
acc_data <- acc_data[,-1] # remove the column added by r with row number
acc_data <- subset(acc_data, select = -c(RGB_data,MS_data,DSM_data,band_data,text_data)) # remove true/false columns


################################################################################
#                                                                              #
#                       PRETTY FIGURE                                          #
#                                                                              #
################################################################################


fig_data <- acc_data %>%
        dplyr::select(-model) %>%
        dplyr::filter(form %in% c('all','sen','ran')) %>%
        reshape::melt(id=c('form','imagery','feats')) %>%
        dplyr::rename(Set = variable, Imagery = imagery, Features = feats, Model = form)

fig_data$Set <- plyr::revalue(fig_data$Set, c('train_acc'='Training','test_acc'='Test'))
fig_data$Imagery <- factor(plyr::revalue(fig_data$Imagery, 
                                             c('noDSM'='RGB + MS',
                                               'noMS'='RGB + DSM',
                                               'noRGB'='MS + DSM',
                                               'all'='All')
                                             ),
                               levels=c('All',
                                        'RGB + MS',
                                        'RGB + DSM',
                                        'MS + DSM',
                                        'RGB',
                                        'MS',
                                        'DSM')
                               )
fig_data$Features <- plyr::revalue(fig_data$Features, c('all'='All','band'='Spectral','text'='Texture'))
fig_data$Model <- factor(plyr::revalue(fig_data$Model, 
                                         c('all'='All Species',
                                           'sen'='No E. malaccense',
                                           'ran'='No E. malaccense and A. scholaris')
                                       ),
                         levels=c('All Species',
                                  'No E. malaccense',
                                  'No E. malaccense and A. scholaris')
                         )

fig_data$value <- fig_data$value*100

# Highlight effect of imagery type
# panel rows - Model, panel cols - feats, x - train, bars - imagery
fig1 <- ggplot(fig_data, aes(x=Set, y=value, fill=Imagery, label=sprintf('%0.1f',round(value,1))))+
        geom_bar(position='dodge',stat='identity')+
        facet_grid(Model ~ Features)+
        labs(y='Accuracy (%)')+
        geom_text(vjust=1.2,color="white", size=3.0, position = position_dodge(0.9))+
        scale_fill_manual(values=c("#ca5f4c",
                                   "#ce872d",
                                   "#8a8d4a",
                                   "#74af41",
                                   "#4aac8d",
                                   "#8d72c9",
                                   "#c65c8a"))
ggsave('2020_05_04_fig_panel_row_model_panel_col_feats_x_train_bars_imagery.pdf',fig1,width=15,height=10)


####### CONSTRUCT THE MAIN FIGURE FOR THE PAPER

pan_a_data <- fig_data %>%
        dplyr::filter(Features == 'All') %>%
        dplyr::filter(Imagery %in% c('All','RGB + DSM','RGB'))
pan_a_data$Imagery <- factor(pan_a_data$Imagery,levels = c('RGB','RGB + DSM','All'))

# create facet labels
lab_data_a <- data.frame(x=rep(0.55,3),
                       y=rep(100,3),
                       Model = factor(c('All Species',
                                          'No E. malaccense',
                                          'No E. malaccense and A. scholaris'),
                                        levels=c('All Species',
                                                 'No E. malaccense',
                                                 'No E. malaccense and A. scholaris')),
                       labs = c('(a)','(b)','(c)'),
                       Imagery = 'All',
                       Features = 'All'
)

fig2 <- ggplot(pan_a_data, aes(x=Set, y=value, fill = Imagery, label=sprintf('%0.1f',round(value,1))))+
        geom_bar(position='dodge',stat='identity')+
        facet_grid(. ~ Model)+
        labs(y='Accuracy (%)')+
        geom_text(vjust=1.2,color="white", size=4.0, position = position_dodge(0.9))+
        scale_fill_manual(values=c('#a6cee3','#1f78b4','#b2df8a'))+
        geom_text(aes(x,y,label=labs,group=NULL),data=lab_data_a)+
        ggtitle("All Features")+
        theme(plot.title = element_text(hjust = 0.5),axis.title.x=element_blank(),
              legend.justification = c(0,0.5))


pan_b_data <- fig_data %>%
        dplyr::filter(Imagery == 'All')
pan_b_data$Features <- factor(pan_b_data$Features,levels = c('Spectral','Texture','All'))

# create facet labels
lab_data_b <- data.frame(x=rep(0.55,3),
                       y=rep(100,3),
                       Model = factor(c('All Species',
                                        'No E. malaccense',
                                        'No E. malaccense and A. scholaris'),
                                      levels=c('All Species',
                                               'No E. malaccense',
                                               'No E. malaccense and A. scholaris')),
                       labs = c('(d)','(e)','(f)'),
                       Imagery = 'All',
                       Features = 'All'
)

fig3 <- ggplot(pan_b_data, aes(x=Set, y=value, fill = Features, label=sprintf('%0.1f',round(value,1))))+
        geom_bar(position='dodge',stat='identity')+
        facet_grid(. ~ Model)+
        labs(y='Accuracy (%)')+
        geom_text(vjust=1.2,color="white", size=4.0, position = position_dodge(0.9))+
        scale_fill_manual(values=c('#66c2a5','#fc8d62','#8da0cb'))+
        geom_text(aes(x,y,label=labs,group=NULL),data=lab_data_b)+
        ggtitle("All Imagery")+
        theme(plot.title = element_text(hjust = 0.5),legend.justification = c(0,0.5))


g1 <- cowplot::plot_grid(
        cowplot::plot_grid(
                fig2 + theme(legend.position = 'none') + background_grid(major='xy',minor='y',size.major = 0.2,size.minor=0.2,colour.major='grey90',colour.minor='grey90'),
                fig3 + theme(legend.position = 'none') + background_grid(major='y',minor='y',size.major = 0.2,size.minor=0.2,colour.major='grey90',colour.minor='grey90'),
                ncol=1,
                align = 'hv'
        ),
        cowplot::plot_grid(
                get_legend(fig2),
                get_legend(fig3),
                ncol=1
                
        ),
        rel_widths = c(9,1)
)

cowplot::save_plot('2020_05_04_SVM_panels_plot_tidied.pdf',g1,base_width=12,base_height=9)
cowplot::save_plot('2020_05_04_SVM_panels_plot_tidied.png',g1,base_width=12,base_height=9)
