# https://medium.com/@oyewusiwuraola/opinion-mining-using-the-uci-drug-review-data-set-part-1-data-loading-and-pre-processing-using-49d3fb6025a8
# https://medium.com/@oyewusiwuraola/opinion-mining-using-the-uci-drug-review-data-set-part-2-sentiment-prediction-using-a-machine-f9f7e5a0f4ec
# https://towardsdatascience.com/my-first-adventures-in-nlp-631faa6aadd4
# https://towardsdatascience.com/sentiment-analysis-with-python-part-1-5ce197074184
# https://towardsdatascience.com/sentiment-analysis-with-python-part-2-4f71e7bde59a
# http://rpubs.com/mkivenson/sentiment-reviews
setwd('C:/Users/krish/Documents/Masters/Hackathons/Project Competition')

library(stringr)
library(dplyr)
library(textstem)
library(stopwords)
library(tm)


# Part 1 - Preaparing the data

train.df <- read.csv('drugsComTrain_raw.csv', stringsAsFactors = FALSE, header = TRUE)
test.df <- read.csv('drugsComTest_raw.csv', stringsAsFactors = FALSE, header = TRUE)

drug.review.df <- rbind(train.df, test.df)

drug.review.df <- drug.review.df[, !(colnames(drug.review.df) %in% c('date'))]

drug.review.df <- drug.review.df[!apply(drug.review.df, 1, function(x) any(x == '')),]

drug.df <- drug.review.df[, c('condition', 'drugName')] %>% group_by(condition, drugName) %>% 
  summarise(reviewCount = n()) %>% filter(reviewCount >= 10 & !grepl('users found this comment helpful', 
                                                                     condition))

drug.review.df <- drug.review.df %>% filter(condition %in% unique(drug.df$condition) & 
                                              drugName %in% unique(drug.df$drugName))

drug.review.df <- drug.review.df %>% inner_join(drug.df, by = c('condition', 'drugName'))


# Part 2 - Cleaning reviews

clean_string <- function(string){
  temp <- tolower(string)
  temp <- str_replace_all(temp, "&#039;", "'")
  temp <- str_replace_all(temp,"[^a-zA-Z'\\s]", " ")
  temp <- str_replace_all(temp, "[\\s]+", " ")
  temp <- str_trim(temp, side = 'both')
  return(temp)
}

# not_stop_list = c("after", "against", "aren't", "before", "but", "can't", "cannot", "couldn't", "didn't", 
#                   "doesn't", "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't", "needn't", 
#                   "no", "nor", "not", "shan't","shouldn't", "won't", "wasn't", "weren't", "wouldn't")

not_stop_list = c("after", "against", "before", "but", "cannot", "no", "not", "nor")

nlp_clean_string <- function(string){
  temp <- lemmatize_strings(string)
  temp <- str_replace_all(temp, "n't", " not")
  corpus <- Corpus(VectorSource(temp), readerControl = list(language = 'en'))
  corpus <- tm_map(corpus, removeWords, c(setdiff(c(stopwords('en')), not_stop_list)))
  # corpus <- tm_map(corpus, stripWhitespace)
  temp <- corpus[[1]]$content
  return(temp)
}

clean_string_after_nlp <- function(string){
  temp <- tolower(string)
  temp <- str_replace_all(temp,"[^a-zA-Z\\s]", " ")
  temp <- str_replace_all(temp, "[\\s]+", " ")
  temp <- str_trim(temp, side = 'both')
  return(temp)
}

drug.review.df$cleanReview <-sapply(drug.review.df$review, clean_string)

drug.review.df$cleanReview <-sapply(drug.review.df$cleanReview, nlp_clean_string)

drug.review.df$cleanReview <-sapply(drug.review.df$cleanReview, clean_string_after_nlp)

write.csv(drug.review.df, file = "drugsReview.csv", row.names = FALSE)

rm(list = ls(all.names = TRUE)) 
gc()