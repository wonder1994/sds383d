install.packages("lda")
library(lda)
data(cora.documents)
data(cora.vocab)
K <- 10 ## Num clusters
result <- lda.collapsed.gibbs.sampler(cora.documents,
                                      K, ## Num clusters
                                      cora.vocab,
                                      1000, ## Num iterations
                                      0.1,
                                      0.1)
## Get the top words in the cluster
top.words <- top.topic.words(result$topics, 5, by.score=TRUE)
