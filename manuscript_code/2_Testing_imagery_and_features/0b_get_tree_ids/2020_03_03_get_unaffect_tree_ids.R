setwd("E:/SLICUAV_manuscript_code/2_Testing_imagery_and_features/0b_get_tree_ids")

data <- read.csv('E:/SLICUAV_manuscript_data/2_Crowns/2020_03_03_crowns_no_MS_issue_50m_buffer.csv',stringsAsFactors = F)
tree_ids <- data[,c('tag_string','Spp_tag','split')]

table(tree_ids[,2],tree_ids[,3])
# generally the split is OK, though not perfect. Slightly more in test than before. However, for now we will stick with it (by doing merging - to be consistent)

write.csv(tree_ids,'2020_03_03_unaffected_tree_ids.csv')
