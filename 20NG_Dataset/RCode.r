# Name: Brijesh Mavani
# CWID: A20406960
# University: Illinois Institute of Technology
# Course: Advanced Data Mining
# Assignment: 1

# set output length on R console to print more information.
options(max.print=999999) 


#Problem 1.a
# set working directory 
setwd("C:/STUDY/MS/Spring18/ADM/Assignments/Assignment1/DataSet/20news-18828") 

# Importing all library which will be used in this code

library(e1071)
library(NbClust)
library(cluster)
library(ggplot2)
library(FunCluster)
library(tm)
library(SnowballC)
library(wordcloud)
library(RColorBrewer)
library(factoextra)
library(lsa)
library(topicmodels)
library(caret)
  
           
dataset<- c( 
  "C:/STUDY/MS/Spring18/ADM/Assignments/Assignment1/DataSet/20news-18828/alt.atheism",
  "C:/STUDY/MS/Spring18/ADM/Assignments/Assignment1/DataSet/20news-18828/comp.windows.x",
  "C:/STUDY/MS/Spring18/ADM/Assignments/Assignment1/DataSet/20news-18828/misc.forsale",
  "C:/STUDY/MS/Spring18/ADM/Assignments/Assignment1/DataSet/20news-18828/rec.sport.hockey",
  "C:/STUDY/MS/Spring18/ADM/Assignments/Assignment1/DataSet/20news-18828/talk.politics.guns")  

news <- Corpus(DirSource(dataset, encoding = "UTF-8"), readerControl=list(reader=readPlain,language="en"))


category <- vector("character", length(news))
category[1:799] <- "alt.atheism"
category[800:1778] <- "comp.windows.x"
category[1779:2748] <- "misc.forsale"
category[2749:3747] <- "rec.sport.hockey"
category[3748:4657] <- "talk.politics.guns"
table(category)

dtmpreproc <- DocumentTermMatrix(news,control=list(wordLengths=c(4,Inf)))
dtmpreproc
 
#processing the data
news<-tm_map (news, content_transformer(tolower))
news<-tm_map (news, removePunctuation)
news<-tm_map (news, stripWhitespace)
news<-tm_map (news, removeNumbers)


#Transforming data by performing basic actions like removing white spaces , stop words etc.

myStopwords<-stopwords('english')
news<-tm_map (news, removeWords,myStopwords)
news<-tm_map (news, stemDocument)
news<-tm_map(news,removeWords,"Subject")
news<-tm_map(news,removeWords,"subject")
news<-tm_map(news,removeWords,"Organization")
news<-tm_map(news,removeWords,"writes")
news<-tm_map(news,removeWords,"From")
news<-tm_map(news,removeWords,"lines")
news<-tm_map(news,removeWords,"NNTP-Posting-Host")
news<-tm_map(news,removeWords,"article")


news<-tm_map (news, content_transformer(tolower))
news<-tm_map (news, removePunctuation)
news<-tm_map (news, stripWhitespace)
news<-tm_map (news, removeNumbers)

#Stop Words: words which do not contain important significance to be used in Search Queries. 
#Usually these words are filtered out from search queries because they return vast amount of unnecessary information

myStopwords<-stopwords('english')
news<-tm_map (news, removeWords,myStopwords)

#stemming : Reduce the count of terms occurring in Document Term matrix which helps to delete the sparse items
#Simplifying them int to single words.
news<-tm_map (news, stemDocument)


#Document Term Matrix
dtmpostproc <- DocumentTermMatrix(news,control=list(wordLengths=c(4,Inf)))
dtmpostproc

# Term Document Matrix
tdmpostproc <- TermDocumentMatrix(news,control=list(wordLengths=c(4,Inf)))
tdmpostproc



#Using TDM to find frequency of words:

tdmspostproc <- removeSparseTerms(tdmpostproc, 0.98)
tdmspostproc
m <- as.matrix(tdmspostproc)
v <- sort(rowSums(m), decreasing=TRUE) 
d <- data.frame(word = names(v),freq=v) 
head(d, 10)


dtmspostproc <- removeSparseTerms(dtmpostproc, 0.98)
dtmspostproc
dtm_tfidf <- weightTfIdf(dtmspostproc)
dtm_tfidf1 <- as.matrix((dtm_tfidf))

dms <- as.matrix(dtmspostproc)
rownames(dms) <- 1:nrow(dms)

freq <- sort(colSums(dms),decreasing = TRUE)
dark2 <- brewer.pal(8, "Dark2")
wordcloud(names(freq), freq, max.words=200, rot.per=0.3,colors=dark2)



#Euclidean distance
norm_eucl <- function(dtm_tfidf1) dtm_tfidf1/apply(dtm_tfidf1, MARGIN=1, FUN=function(x) sum(x^2)^.5)
dtm_norm <- norm_eucl(dtm_tfidf1)


# NB Clust:
# nb<-NbClust(dtm_norm, min.nc=2, max.nc=15, method="kmeans")
# nb <- NbClust(dtm_norm, distance = "euclidean", method = "complete", index ="all")
# fviz_nbclust(nb, kmeans, method="wss")

#Kmeans with 3 clusters

kmeans3 <- kmeans((dtm_norm), centers=3,nstart=25)
table(kmeans3$cluster)
kmeans3$withinss 
kmeans3$tot.withinss
kmeans3$totss
fviz_cluster(kmeans3, data=dtm_norm,centers=3, nstart=25,main="K Means cluster with 3 clusters")
#plot(prcomp(dtm_norm$x),col=kmeans3$cluster)
 

#Kmeans with 4 clusters

kmeans4 <- kmeans((dtm_norm), centers=4,nstart=25)
table(kmeans4$cluster)
kmeans4$withinss 
kmeans4$tot.withinss    
kmeans4$totss
fviz_cluster(kmeans4, data=dtm_norm,centers=4, nstart=25,main="K Means cluster with 4 clusters")


#Kmeans with 5 clusters

kmeans5 <- kmeans((dtm_norm), centers=5,nstart=25)
table(kmeans5$cluster)
kmeans5$withinss
kmeans5$tot.withinss    
kmeans5$totss
fviz_cluster(kmeans5, data=dtm_norm,centers=5, nstart=25,main="K Means cluster with 5 clusters")


#Kmeans with 6 clusters

kmeans6 <- kmeans((dtm_norm), centers=6,nstart=25)
table(kmeans6$cluster)
kmeans6$withinss 
kmeans6$tot.withinss    
kmeans6$totss
fviz_cluster(kmeans6, data=dtm_norm,centers=6, nstart=25,main="K Means cluster with 6 clusters")




for (i in 1:length(kmeans5$withinss)) {
  #For each cluster, this defines the documents in that cluster
  inGroup <- which(kmeans5$cluster==i)
  within <- dtm_norm[inGroup,]
  if(length(inGroup)==1) within <- t(as.matrix(within))
  out <- dtm_norm[-inGroup,]
  words <- apply(within,2,mean) - apply(out,2,mean) #Take the difference in means for each term
  print(c("Cluster", i), quote=F)
  labels <- order(words, decreasing=T)[1:20] #Take the top 20 Labels
  print(names(words)[labels], quote=F) #From here down just labels
  if(i==length(kmeans5$withinss)) {
    print("Cluster Membership")
    print(table(kmeans5$cluster))
    print("Within cluster sum of squares by cluster")
    print(kmeans5$withinss)
  }
}




conf_mat<-data.frame(kmeans5$cluster)
colnames(conf_mat)<-"cluster_num"

conf_mat$actual <- sapply(1:nrow(conf_mat), assign_cluster_label)

#caret package
confusionMatrix(conf_mat$cluster_num,conf_mat$actual)


Confm <- table(category, kmeans5$cluster)
table(category)
table(kmeans5$cluster)
#Confusion Matrix
Confm
#Accuracy
(sum(apply(Confm, 1, max))/sum(kmeans3$size))*100
#Precision
Precision <- (apply(Confm, 1, max))/(apply(Confm, 1, sum))
Precision
#Recall
Recall<-(apply(Confm, 1, max))/((apply(Confm, 1, max))+ ((apply(Confm, 1, sum))-((apply(Confm, 1, max)))))
Recall           
#F1
F1<-2/((1/Recall)+(1/Precision))
F1
#SSE
kmeans3$tot.withinss
kmeans3$totss
100-(kmeans5$tot.withinss/kmeans5$totss)*100


#20NG Precision,Recall,F1
num_instances = sum(confmmatrix) # number of instances
num_classes = nrow(confmmatrix) # number of classes
correct_classifier = diag(confmmatrix) # number of correctly classified instances per class
n_inst_pCLass = apply(confmmatrix, 1, sum) # number of instances per class
n_pred_pCLass = apply(confmmatrix, 2, sum) # number of predictions per class
actual = n_inst_pCLass / num_instances # distribution of instances over the actual classes
predicted = n_pred_pCLass / num_instances # distribution of instances over the predicted classes
NG20accuracy = sum(correct_classifier) / num_instances
NG20accuracy
NG_20Precision = correct_classifier / n_pred_pCLass
NG_20Precision
NG_20Recall = correct_classifier / n_inst_pCLass
NG_20Recall
NG_20F1 = 2 * NG_20Precision * NG_20Recall / (NG_20Precision + NG_20Recall)
NG_20F1
data.frame(NG_20Precision, NG_20Recall, NG_20F1) 


#LSA
LSA<-function(input,dim){
  s<-svd(input)
  u<-as.matrix(s$u[,1:dim])
  v<-as.matrix(s$v[,1:dim])
  d<-as.matrix(diag(s$d)[1:dim,1:dim])
  return(as.matrix(u%*%d%*%t(v),type="green"))
}

#SVD With 5 Dimensions
svd_dtm5<-LSA(dtmspostproc,5)
svd_norm5<-norm_eucl(svd_dtm5)

#Clustering with 5 dimensions
lsacluster5 <- kmeans(svd_norm5, 5)
lsa_Confm5 <- table(category, lsacluster5$cluster)
table(lsacluster5$cluster)
dim(lsacluster5$cluster)
#Confusion Matrix
lsa_Confm5
#SSE
lsacluster5$totss
lsacluster5$tot.withinss
100-(lsacluster5$tot.withinss/lsacluster5$totss)*100
#Accuracy
(sum(apply(lsa_Confm5, 1, max))/sum(lsacluster5$size))*100
#plot
pr<-prcomp(svd_norm5)$x
plot( pr,col=lsacluster5$cluster,main='SVD for d=5')

#SVD With 50 Dimensions
svd_dtm50<-LSA(dtmspostproc,50)
svd_norm50<-norm_eucl(svd_dtm50)


#Clustering with 50 dimensions
lsacluster50 <- kmeans(svd_norm50, 5)
lsa_Confm50 <- table(category, lsacluster50$cluster)
#Confusion Matrix
lsa_Confm50
#SSE
lsacluster50$totss
lsacluster50$tot.withinss
100-(lsacluster50$tot.withinss/lsacluster50$totss)*100
#Accuracy
(sum(apply(lsa_Confm50, 1, max))/sum(lsacluster50$size))*100
#plot
pr<-prcomp(svd_norm50)$x
plot( pr,col=lsacluster50$cluster,main='SVD for d=50')


#SVD With 100 Dimensions
svd_dtm100<-LSA(dtmspostproc,100)
svd_norm100<-norm_eucl(svd_dtm100)

#Clustering with 100 dimensions
lsacluster100 <- kmeans(svd_norm100, 5)
lsa_Confm100 <- table(category, lsacluster100$cluster)
#Confusion Matrix
lsa_Confm100
#SSE
lsacluster100$totss
lsacluster100$tot.withinss
100-(lsacluster100$tot.withinss/lsacluster100$totss)*100
#Accuracy
(sum(apply(lsa_Confm100, 1, max))/sum(lsacluster100$size))*100
#plot
pr<-prcomp(svd_norm100)$x
plot( pr,col=lsacluster100$cluster,main='SVD for d=100')



#SVD With 200 Dimensions
svd_dtm200<-LSA(dtmspostproc,200)
svd_norm200<-norm_eucl(svd_dtm200)

#Clustering with 200 dimensions
lsacluster200 <- kmeans(svd_norm200, 5)
lsa_Confm200 <- table(category, lsacluster200$cluster)
#Confusion Matrix
lsa_Confm200
#SSE
lsacluster200$totss
lsacluster200$tot.withinss
100-(lsacluster200$tot.withinss/lsacluster200$totss)*100
#Accuracy
(sum(apply(lsa_Confm200, 1, max))/sum(lsacluster200$size))*100
#plot
pr<-prcomp(svd_norm200)$x
plot( pr,col=lsacluster200$cluster,main='SVD for d=200')

#SVD With 300 Dimensions
svd_dtm300<-LSA(dtmspostproc,300)
svd_norm300<-norm_eucl(svd_dtm300)

#Clustering with 300 dimensions
lsacluster300 <- kmeans(svd_norm300, 5)
lsa_Confm300 <- table(category, lsacluster300$cluster)
#Confusion Matrix
lsa_Confm300
#SSE
lsacluster300$totss
lsacluster300$tot.withinss
100-(lsacluster300$tot.withinss/lsacluster300$totss)*100
#Accuracy
(sum(apply(lsa_Confm300, 1, max))/sum(lsacluster300$size))*100
#plot
pr<-prcomp(svd_norm300)$x
plot( pr,col=lsacluster300$cluster,main='SVD for d=300')



# Find representative words
num=5
concept<-function(num){
  sv<-sort.list((svd(dtmspostproc))$v[,num],decreasing = TRUE)
  dm<-dtmspostproc$dimnames$Terms[head(sv,10)]
  dm
}
i<-(1:num)
lapply(i,concept)


#LDA
burnin <- 4000
iter <- 1000
thin <- 500
seed <-list(2003,5,63,100001,765)
nstart <- 5
best <- TRUE

lda5 <-LDA(dtmspostproc,5, method="Gibbs", control=list(nstart=nstart, seed = seed, best=best, burnin = burnin, iter = iter, thin=thin))
#top l terms in each topic
lda5.terms <- as.matrix(terms(lda5,10))
lda5.terms
lda_data<-as.data.frame(lda5@gamma)
lda_data_matrix<-as.matrix(lda_data)
rownames(lda_data_matrix)<-1:nrow(lda_data_matrix)
norm_eucl <- function(m) m/apply(m, MARGIN=1, FUN=function(x) sum(x^2)^.5)
lda_norm<-norm_eucl(lda_data_matrix)


# LDA with 5 clusters
lda_cluster5<-kmeans(lda_norm,centers=5,nstart=50)
plot(prcomp(lda_norm)$x,col=lda_cluster5$cluster,main='LDA for d=5')
lda_Confm <- table(category, lda_cluster5$cluster)  
#Confusion Matrix
lda_Confm
#SSE
lda_cluster5$totss
lda_cluster5$tot.withinss
100-(lda_cluster5$tot.withinss/lda_cluster5$totss)*100
#Accuracy
(sum(apply(lda_Confm, 1, max))/sum(lda_cluster5$size))*100



#LDA 50: 
lda50 <-LDA(dtmspostproc,50, method="Gibbs", control=list(nstart=nstart, seed = seed, best=best, burnin = burnin, iter = iter, thin=thin))
#top l terms in each topic
lda50.terms <- as.matrix(terms(lda5,10))
lda50.terms
lda_data50<-as.data.frame(lda5@gamma)
lda_data_matrix50<-as.matrix(lda_data50)
rownames(lda_data_matrix50)<-1:nrow(lda_data_matrix50)
norm_eucl <- function(m) m/apply(m, MARGIN=1, FUN=function(x) sum(x^2)^.5)
lda_norm50<-norm_eucl(lda_data_matrix50)


# LDA with 5 clusters
lda_cluster50<-kmeans(lda_norm50,centers=5,nstart=50)
plot(prcomp(lda_norm50)$x,col=lda_cluster50$cluster,main='LDA for d=50')
lda_Confm <- table(category, lda_cluster50$cluster)  
#Confusion Matrix
lda_Confm
#SSE
lda_cluster50$totss
lda_cluster50$tot.withinss
(lda_cluster50$tot.withinss/lda_cluster50$totss)*100
#Accuracy
(sum(apply(lda_Confm50, 1, max))/sum(lda_cluster50$size))*100



#LDA 100: 
lda100 <-LDA(dtmspostproc,100, method="Gibbs")
#, control=list(nstart=nstart, seed = seed, best=best, burnin = burnin, iter = iter, thin=thin))
#top l terms in each topic
lda100.terms <- as.matrix(terms(lda100,10))
lda100.terms
lda_data100<-as.data.frame(lda100@gamma)
lda_data_matrix100<-as.matrix(lda_data100)
rownames(lda_data_matrix100)<-1:nrow(lda_data_matrix100)
norm_eucl <- function(m) m/apply(m, MARGIN=1, FUN=function(x) sum(x^2)^.5)
lda_norm100<-norm_eucl(lda_data_matrix100)


# LDA with 5 clusters
lda_cluster100<-kmeans(lda_norm100,centers=5,nstart=50)
plot(prcomp(lda_norm100)$x,col=lda_cluster100$cluster,main='LDA for d=100')
lda_Confm100 <- table(category, lda_cluster100$cluster)  
#Confusion Matrix
lda_Confm100
#SSE
lda_cluster100$totss
lda_cluster100$tot.withinss
(lda_cluster100$tot.withinss/lda_cluster100$totss)*100
#Accuracy
(sum(apply(lda_Confm100, 1, max))/sum(lda_cluster100$size))*100


#LDA 200: 
lda200 <-LDA(dtmspostproc,200, method="Gibbs")
#, control=list(nstart=nstart, seed = seed, best=best, burnin = burnin, iter = iter, thin=thin))
#top l terms in each topic
lda200.terms <- as.matrix(terms(lda200,10))
lda200.terms
lda_data200<-as.data.frame(lda200@gamma)
lda_data_matrix200<-as.matrix(lda_data200)
rownames(lda_data_matrix200)<-1:nrow(lda_data_matrix200)
norm_eucl <- function(m) m/apply(m, MARGIN=1, FUN=function(x) sum(x^2)^.5)
lda_norm200<-norm_eucl(lda_data_matrix200)


# LDA with 5 clusters
lda_cluster200<-kmeans(lda_norm200,centers=5,nstart=50)
plot(prcomp(lda_norm200)$x,col=lda_cluster200$cluster,main='LDA for d=200')
lda_Confm200 <- table(category, lda_cluster200$cluster)  
#Confusion Matrix
lda_Confm200
#SSE
lda_cluster200$totss 
lda_cluster200$tot.withinss
(lda_cluster200$tot.withinss/lda_cluster200$totss)*100
#Accuracy
(sum(apply(lda_Confm200, 1, max))/sum(lda_cluster200$size))*100


#Confusion Matrix
lda_Confm
#SSE
lda_cluster6$totss
lda_cluster6$tot.withinss
100-(lda_cluster6$tot.withinss/lda_cluster6$totss)*100
#Accuracy
(sum(apply(lda_Confm, 1, max))/sum(lda_cluster6$size))*100


num_instances = sum(lda_Confm) # number of instances
num_classes = nrow(lda_Confm) # number of classes
correct_classifier = diag(lda_Confm) # number of correctly classified instances per class
n_inst_pCLass = apply(lda_Confm, 1, sum) # number of instances per class
n_pred_pCLass = apply(lda_Confm, 2, sum) # number of predictions per class
actual = n_inst_pCLass / num_instances # distribution of instances over the actual classes
predicted = n_pred_pCLass / num_instances # distribution of instances over the predicted classes
NG20accuracy = sum(correct_classifier) / num_instances
NG20accuracy
NG_20Precision = correct_classifier / n_pred_pCLass
NG_20Precision
NG_20Recall = correct_classifier / n_inst_pCLass
NG_20Recall
NG_20F1 = 2 * NG_20Precision * NG_20Recall / (NG_20Precision + NG_20Recall)
NG_20F1
data.frame(NG_20Precision, NG_20Recall, NG_20F1) 
