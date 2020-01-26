library(tidyverse)

df0 <- read.csv("../data/df0.csv")
df1 <- read.csv("../data/df1.csv")
df2 <- read.csv("../data/df2.csv")
df3 <- read.csv("../data/df3.csv")
df4 <- read.csv("../data/df4.csv")

emotions <- c("Anger","Happiness", "Sadness", "Surprise")

center_apply <- function(x) {
  apply(x, 2, function(y)(y - mean(y)) / var(y) )
}

binarize_emotions <- function(x) {
  x$Anger[x$Anger >= 1] <- 1
  x$Anger[x$Anger < 1] <- 0
  x$Sadness[x$Sadness >= 1] <- 1
  x$Sadness[x$Sadness < 1] <- 0
  x$Surprise[x$Surprise >= 1] <- 1
  x$Surprise[x$Surprise < 1] <- 0
  x$Happiness[x$Happiness >= 1] <- 1
  x$Happiness[x$Happiness < 1] <- 0
  return(x)
}

aggregate_binarized_emotions <- function(x) {
  
  x$Anger[x$Anger < 2] <- 0
  x$Anger[x$Anger >= 2] <- 1
  
  x$Sadness[x$Sadness < 2] <- 0
  x$Sadness[x$Sadness >= 2] <- 1
  
  x$Surprise[x$Surprise < 2] <- 0
  x$Surprise[x$Surprise >= 2] <- 1
  
  x$Happiness[x$Happiness < 2] <- 0
  x$Happiness[x$Happiness >= 2] <- 1
  return(x)
}

df0[emotions] <- center_apply(df0[emotions])
df0 <- binarize_emotions(df0)

df1[emotions] <- center_apply(df1[emotions])
df1 <- binarize_emotions(df1)

df2[emotions] <- center_apply(df2[emotions])
df2 <- binarize_emotions(df2)

df3[emotions] <- center_apply(df3[emotions])
df3 <- binarize_emotions(df3)

df4[emotions] <- center_apply(df4[emotions])
df4 <- binarize_emotions(df4)

aggregate_emotions <- df0[emotions] + df1[emotions] + df2[emotions] + df3[emotions] + df4[emotions]
aggregate_emotions$video_ids <- df0$video_ids
aggregate_emotions <- cbind(df0[1:7], aggregate_emotions, df0[17:19])
aggregate_emotions <- aggregate_binarized_emotions(aggregate_emotions)
aggregate_emotions$A_or_B[aggregate_emotions$ratedActor == aggregate_emotions$actorA] <- "A"
write.csv(aggregate_emotions, "../data/aggregate_emotion_labels.csv")
