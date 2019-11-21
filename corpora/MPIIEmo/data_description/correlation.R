library(corrplot)
library(RColorBrewer)
library(tidyverse)
# 
# This script does two things.
# 1) For each annotator, computes correlations between different affects/emotions. 
# These are aggregated and averaged to get av. correlation for each possible pair.
# 
# 2) For all possible annotator pairs, computes correlation between annotations for same emotions
# ie correlation of annotators b and c wrt anger, corelation of annotators b and c wrt sadness, etc...
# Doing this we can plot a range of min and max correlation between annotators wrt a certain emotion.
# We also get mean correlation for each emotion
# Using these mean correlations we further compute mean to get global correlation for emotions
# and global correlation for affect

# read dataframes
df0 <- read.csv("../data/df0.csv")
df1 <- read.csv("../data/df1.csv")
df2 <- read.csv("../data/df2.csv")
df3 <- read.csv("../data/df3.csv")
df4 <- read.csv("../data/df4.csv")

# make list containing emotion/affect cols of all dataframes
df_list <- list(a = df0[,8:16], b = df1[,8:16], c = df2[,8:16], d = df3[,8:16], e = df4[,8:16])
# permute to get all possible annotator pairs
annotator_pairs <- combn(names(df_list), 2)

# each correlation matrix will contain correlations between different emotions, wrt one annotator
all_correlations <- list()
for(name in names(df_list)){
  all_correlations[[name]] <- cor(df_list[[name]], method=c("pearson")) 
}

# sum all correlation matrices together and average by num annotators
av_correlations <- Reduce("+", all_correlations)
av_correlations <- av_correlations / 5

# plot
corrplot(av_correlations, method="number", type="upper", order="hclust",
         tl.col="black", col=brewer.pal(n=8, name="RdYlBu"))

# plot av. correlation between emotions
emotion_av_correlations <- av_correlations[c("Anger","Sadness","Happiness","Surprise"), c("Anger","Sadness","Happiness","Surprise")]
corrplot(emotion_av_correlations, method="number", type="upper", order="hclust",
         tl.col="black", col=brewer.pal(n=8, name="RdYlBu"))


# plot av. correlation between affects
affect_av_correlations <- av_correlations[c("Activation","Anticipation","Power","Valence"), c("Activation","Anticipation","Power","Valence")]
corrplot(affect_av_correlations, method="number", type="upper", order="hclust",
         tl.col="black", col=brewer.pal(n=8, name="RdYlBu"))

# That concludes part 1


# correlation vectors contains only the diagonal from correlation matrix between two annotators
# represents how emotion x correlates between annotators a and b
correlation_vectors <- list()
for(col in 1:ncol(annotator_pairs)) {
  anno1 <- annotator_pairs[1,col]
  anno2 <- annotator_pairs[2,col]
  pair_label <- paste(anno1, anno2, sep='')
  cor_frame <- cor(df_list[[anno1]], df_list[[anno2]], method = c("pearson"))
  correlation_vectors[[pair_label]] <- diag(cor_frame)
  }

# row bind all cor vectors together into matrix
correlations <- do.call(rbind, correlation_vectors)
correlations <- as.data.frame(correlations)
# convert to df and subset into emotion, affect matrices
emotion_correlations <- select(correlations, Anger, Happiness, Sadness, Surprise)
affect_correlations <- select(correlations, Activation, Anticipation, Power, Valence)

# plot emotion
# transpose for plotting
emotion_correlations_t <- as.data.frame(t(emotion_correlations))
ggplot(emotion_correlations_t, aes(colour=rownames(emotion_correlations_t))) +
  geom_segment(aes(x=apply(emotion_correlations_t,1,min), xend=apply(emotion_correlations_t,1,max), 
                   y=rownames(emotion_correlations_t), yend=rownames(emotion_correlations_t)), size=10) +
  geom_segment(aes(colour="black", x=apply(emotion_correlations_t,1,mean), xend=apply(emotion_correlations_t,1,mean)+.01, 
                   y=rownames(emotion_correlations_t), yend=rownames(emotion_correlations_t), size=15)) +
  xlim(0,1) +
  theme(legend.position ="none",
        axis.title.y=element_blank(),
        plot.title = element_text(hjust=0.5)) +
        labs(x = "Pearson correlation coefficient", y = "Emotion")

# get global correlation measure for emotions
global_emotion_cor_mean <- mean(apply(emotion_correlations_t,1,mean))
print(global_emotion_cor_mean)

# plot affect
# transpose for plotting
affect_correlations_t <- as.data.frame(t(affect_correlations))
ggplot(affect_correlations_t, aes(colour=rownames(affect_correlations_t))) +
  geom_segment(aes(x=apply(affect_correlations_t,1,min), xend=apply(affect_correlations_t,1,max), 
                   y=rownames(affect_correlations_t), yend=rownames(affect_correlations_t)), size=10) +
  geom_segment(aes(colour="black", x=apply(affect_correlations_t,1,mean), xend=apply(affect_correlations_t,1,mean)+.01, 
                   y=rownames(affect_correlations_t), yend=rownames(affect_correlations_t), size=15)) +
  xlim(0,1) +
  theme(legend.position ="none",
        axis.title.y=element_blank(),
        plot.title = element_text(hjust=0.5)) +
  labs(x = "Pearson correlation coefficient", y = "Affect")

# get global correlation measure for affect
global_affect_cor_mean <- mean(apply(affect_correlations_t,1,mean))
print(global_affect_cor_mean)
 