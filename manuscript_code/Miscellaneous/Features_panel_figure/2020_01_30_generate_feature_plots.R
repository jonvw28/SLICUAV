library(ggplot2)
library(imager)
library(cowplot)
library(plot.matrix)
library(viridis)
library(glcm)
library(wvtool)

setwd("D:/jon/Documents/Cache/species_id_manuscript/2020_01_29_pipeline_feats_figs")

#### Autocorrelation

# make up some data

set.seed(42)
x <- seq(from=0,to=15,by=0.5)
y1 <- exp(-0.1*x)*cos(x/1.5)+0.2*runif(x)*exp(-0.02*x)
y2 <- exp(-0.5*x)*cos(x/3)+0.5*runif(x)*exp(-0.05*x)
y3 <- exp(-0.1*x)*cos(x/1.2)+0.1*runif(x)*exp(-0.01*x)
y4 <- exp(-0.5*x)*cos(x/1.5)+0.5*runif(x)*exp(-0.05*x)
yhat <- (y1 + y2 + y3 + y4)/4

autocor_data <- data.frame(x = x, 
                           Autocorrelation = rep(c('Direction 1','Direction 2','Direction 3','Direction 4','Mean'),
                                                 each = length(x),times=1),
                           y = c(y1,y2,y3,y4,yhat))
autocor_data$y <- autocor_data$y/max(autocor_data$y)

g1 <- ggplot(autocor_data, aes(x=x, y=y, group=Autocorrelation))+
        geom_line(aes(color=Autocorrelation,size = Autocorrelation))+
        scale_color_manual(values=c('#1f78b4','#66c2a5','#fc8d62','#8da0cb','red'))+
        scale_size_manual(values=c(0.5,0.5,0.5,0.5,2))+
        labs(y='Autocorrelation',x='Offset (pixels)')+
        theme(legend.title=element_blank())
ggsave('2020_01_30_auto_cor_artifical_data.png',g1,width=9,height=4.5)


#### Use real data

img <- load.image('2020_01_10_MAC_99.png')
gray_img <- imager::grayscale(as.cimg(img[,,,1:3]))
im_mat <- img[,,1,1:3]
gray_mat <- gray_img[,,1,1]

###### Spectra

red <- as.vector(im_mat[,,1])
green <- as.vector(im_mat[,,2])
blue <- as.vector(im_mat[,,3])

rgb_data <- data.frame(Band = factor(rep(c('Red','Green','Blue'),each = length(red)),levels=c('Red','Green','Blue')),Intensity = c(red,green,blue))

g2 <- ggplot(rgb_data,aes(x=Intensity,color=Band,fill=Band))+
        geom_density(alpha=.5)+
        labs(y='Density',x='Pixel Intensity')+
        theme(axis.title.x=element_blank())

g3 <- ggplot(rgb_data,aes(x=Band,color=Band,y=Intensity,fill=Band))+
        geom_boxplot(alpha=0.75)+
        coord_flip() +
        labs(y='Pixel Intensity')

g4 <- cowplot::plot_grid(
        cowplot::plot_grid(
                g2 + theme(legend.position = 'none') + background_grid(major='xy',minor='y',size.major = 0.2,size.minor=0.2,colour.major='grey90',colour.minor='grey90'),
                g3 + theme(legend.position = 'none') + background_grid(major='y',minor='y',size.major = 0.2,size.minor=0.2,colour.major='grey90',colour.minor='grey90'),
                ncol=1,
                align = 'hv'
        ),
        cowplot::plot_grid(
                get_legend(g2),
                get_legend(g3),
                ncol=1
                
        ),
        rel_widths = c(9,1)
)

cowplot::save_plot('2020_01_30_RGB_stats_figure.png',g4,base_width=9,base_height=9)



####### GLCM

glcm_ex <- matrix(c(10,5,5,3,1,6,0,0,2,2,5,0,2,0,1,3,2,0,1,0,1,2,1,0,5),nrow=5,ncol=5)
png('2020_01_30_GLCM_matrix.png',height=480,width=480)
plot(glcm_ex,col=viridis::viridis,breaks=11,key=NULL,fmt.cell='%.0f',text.cell=list(cex=3))
dev.off()


glcm_mat <- glcm::glcm(gray_mat,window=c(9,9),statistics = 'contrast')
png('2020_01_30_GLCM_image.png',height=480,width=480)
plot(as.cimg(glcm_mat))
dev.off()

LBP_box <- matrix(c(2,NA,NA,4),nrow=2,ncol=2)
png('2020_01_30_GLCM_grey_fig.png',height=540,width=480)
plot(LBP_box,key=NULL,col=gray.colors(n=2),na.col='darkgrey',fmt.cell='%.0f',na.print=FALSE,text.cell=list(cex=7))
dev.off()


####### LBP

LBP_mat <- matrix(c(1,0,2,4,3,1,6,7,1),nrow=3,ncol=3)
png('2020_01_30_LBP_fig.png',height=480,width=480)
plot(LBP_mat,key=NULL,col=gray.colors(n=8),fmt.cell='%.0f',text.cell=list(cex=7))
dev.off()

LBP_data <- wvtool::lbp(gray_mat,r=2)
LBP_data <- LBP_data[[1]]

# Resample 57 from issues of multiple pixels per real pixel
probs <- table(LBP_data)
probs['57'] <- 0
probs['57'] <- round(mean(probs))
probs <- probs/sum(probs)

LBP_data[LBP_data==57] <- sample(c(3,6,8,9,10,13,14,15,17,18,19,20,21,24,25,26,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58),size=sum(LBP_data==57),replace=TRUE,prob=probs)
LBP_data <- data.frame(Feature = rep('Group',times = length(c(LBP_data))),Value = c(LBP_data))

g5 <- ggplot(LBP_data,aes(x=Value,color=Feature,fill=Feature))+
        geom_histogram(alpha=.5,bins=59)+
        labs(y='Density',x='LBP')+
        theme(axis.title.x=element_blank())
ggsave('2020_01_30_LBP_figure.png',g5,width=9,height=4.5)

####### LAWS

# Save the kernel
E5 <- c(-1,-2,0,2,1)
E5E5 <- E5 %*% t(E5)
png('2020_01_30_E5E5_matrix.png',height=480,width=480)
plot(E5E5,col=viridis::viridis,breaks=7,key=NULL,fmt.cell='%.0f',text.cell=list(cex=3))
dev.off()

E5 <- c(-1,-2,0,2,1)
E5E5 <- E5 %*% t(E5)
png('2020_01_30_E5E5_matrix.png',height=480,width=480)
plot(E5E5,col=white,key=NULL,fmt.cell='%.0f',text.cell=list(cex=3))
dev.off()


# do the convolution
E5E5_conv <- imagine::convolution2D(gray_mat,E5E5)
E5E5_conv <- E5E5_conv[3:(nrow(E5E5_conv)-2),3:(ncol(E5E5_conv)-2)]
png('2020_01_30_E5E5_conv.png',height=480,width=480)
plot(as.cimg(E5E5_conv))
dev.off()
