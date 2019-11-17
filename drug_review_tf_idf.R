# https://medium.com/@oyewusiwuraola/opinion-mining-using-the-uci-drug-review-data-set-part-1-data-loading-and-pre-processing-using-49d3fb6025a8
# https://medium.com/@oyewusiwuraola/opinion-mining-using-the-uci-drug-review-data-set-part-2-sentiment-prediction-using-a-machine-f9f7e5a0f4ec
# https://towardsdatascience.com/my-first-adventures-in-nlp-631faa6aadd4
# https://towardsdatascience.com/sentiment-analysis-with-python-part-1-5ce197074184
# https://towardsdatascience.com/sentiment-analysis-with-python-part-2-4f71e7bde59a
# http://rpubs.com/mkivenson/sentiment-reviews
setwd('C:/Users/krish/Documents/Masters/Hackathons/Project Competition')

library(quanteda)
library(caret)
library(e1071)
library(randomForest)

drug.review.df <- read.csv('./drugsReview.csv', stringsAsFactors = FALSE, header = TRUE)

drug.review.df$sentiment <- factor(ifelse(drug.review.df$rating > 5, 1, 0))

drug.sentiment.df <- drug.review.df[, c('cleanReview', 'sentiment')]

drug.sentiment.df <-  drug.sentiment.df[sample(nrow(drug.sentiment.df), 10000), ]

# 70%/30% stratified split
set.seed(42)
indexes <- createDataPartition(drug.sentiment.df$sentiment, times = 1, p = 0.7, list = FALSE)
train.df <- drug.sentiment.df[indexes,]
test.df <- drug.sentiment.df[-indexes,]

# Verify proportions.
prop.table(table(train.df$sentiment))
prop.table(table(test.df$sentiment))

set.seed(42)
train.df <- downSample(train.df, train.df$sentiment)[, -3]
prop.table(table(train.df$sentiment))

# Tokenize reviews
train.tokens <- quanteda::tokens(train.df$cleanReview, what = 'word')

# Create bag-of-words model
train.tokens.dfm <- dfm(train.tokens)
train.tokens.dfm <- dfm_trim(train.tokens.dfm, min_termfreq = 10, min_docfreq = 5)
train.tokens.matrix <- as.matrix(train.tokens.dfm)
dim(train.tokens.matrix)
# colnames(train.tokens.matrix)

# TF
term.frequency <- function(row){
  row/sum(row)
}

# IDF
inverse.doc.freq <- function(col){
  corpus.size <- length(col)
  doc.count <- length(which(col > 0))
  
  log10(corpus.size/ doc.count)
}

# TF-IDF
tf.idf <- function(tf, idf){
  tf*idf
}


# Normalize all documents via TF for training dataset
train.tokens.df <- apply(train.tokens.matrix, 1, term.frequency)
dim(train.tokens.df)
# View(train.tokens.df[1:20, 1:100])

# Calculate IDF vector
train.tokens.idf <- apply(train.tokens.matrix, 2, inverse.doc.freq)
str(train.tokens.idf)

# Calculate TF-IDF
train.tokens.tfidf <-  apply(train.tokens.df, 2, tf.idf, idf = train.tokens.idf)
dim(train.tokens.tfidf)
# View(train.tokens.tfidf[1:25, 1:25])

# Transpose the matrix
train.tokens.tfidf <- t(train.tokens.tfidf)
dim(train.tokens.tfidf)
# View(train.tokens.tfidf[1:25, 1:25])

# Check for incopmlete cases.
train.incomplete.cases <- which(!complete.cases(train.tokens.tfidf))
train.df$cleanReview[train.incomplete.cases]

# Fix incomplete cases
train.tokens.tfidf[train.incomplete.cases,] <- rep(0.0, ncol(train.tokens.tfidf))
dim(train.tokens.tfidf)
sum(which(!complete.cases(train.tokens.tfidf)))


# Make a clean data frame
train.tokens.tfidf.df <- cbind(sentiment = train.df$sentiment, as.data.frame(train.tokens.tfidf))
names(train.tokens.tfidf.df) <- make.names(names(train.tokens.tfidf.df))
# View(train.tokens.tfidf.df[1:25, 1:25])


test.tokens <- quanteda::tokens(test.df$cleanReview, what = 'word')

# Create bag-of-words model for train set
test.tokens.dfm <- dfm(test.tokens)
# test.tokens.dfm <- dfm_trim(test.tokens.dfm, min_termfreq = 10, min_docfreq = 5)
test.tokens.matrix <- as.matrix(test.tokens.dfm)
dim(test.tokens.matrix)
# colnames(test.tokens.matrix)



# Normalize all documents via TF for test dataset
test.tokens.df <- apply(test.tokens.matrix, 1, term.frequency)
dim(test.tokens.df)

# Calculate TF-IDF for our test corpus using TF-test and IDF-train
test.tokens.tfidf <-  apply(test.tokens.df, 2, tf.idf, idf = train.tokens.idf)
dim(test.tokens.tfidf)

# Transpose
test.tokens.tfidf <- t(test.tokens.tfidf)

# Check for incopmlete cases.
test.incomplete.cases <- which(!complete.cases(test.tokens.tfidf))
test.df$cleanReview[test.incomplete.cases]

# Fix incomplete cases
test.tokens.tfidf[test.incomplete.cases,] <- rep(0.0, ncol(test.tokens.tfidf))
dim(test.tokens.tfidf)
sum(which(!complete.cases(test.tokens.tfidf)))

# Make a clean data frame
test.tokens.tfidf.df <- cbind(sentiment = test.df$sentiment, as.data.frame(test.tokens.tfidf))
names(test.tokens.tfidf.df) <- make.names(names(test.tokens.tfidf.df))

# Naive Bayes Classifier
nb.classifier <- naiveBayes(sentiment ~ ., data = train.tokens.tfidf.df, laplace = 1)
nb.test.pred <- predict(nb.classifier, test.tokens.tfidf.df[, -1])
confusionMatrix(nb.test.pred, test.df$sentiment, positive = '1')

library(pROC)
nb.roc <- roc(as.numeric(nb.test.pred), as.numeric(test.df$sentiment))
plot.roc(nb.roc, col = 'red', lwd = 2, main = 'ROC of Naive Bayes Model')


# Random Forest Classifier
# rf.classifier <- randomForest(sentiment ~., data = train.tokens.tfidf.df, importance = TRUE)
# rf.test.pred <- predict(rf.classifier, newdata = test.tokens.tfidf.df[, -1], type = 'class')
# confusionMatrix(rf.test.pred, test.df$sentiment, positive = '1')