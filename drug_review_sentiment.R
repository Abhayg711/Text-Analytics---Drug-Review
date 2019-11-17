# https://medium.com/@oyewusiwuraola/opinion-mining-using-the-uci-drug-review-data-set-part-1-data-loading-and-pre-processing-using-49d3fb6025a8
# https://medium.com/@oyewusiwuraola/opinion-mining-using-the-uci-drug-review-data-set-part-2-sentiment-prediction-using-a-machine-f9f7e5a0f4ec
# https://towardsdatascience.com/my-first-adventures-in-nlp-631faa6aadd4
# https://towardsdatascience.com/sentiment-analysis-with-python-part-1-5ce197074184
# https://towardsdatascience.com/sentiment-analysis-with-python-part-2-4f71e7bde59a
# http://rpubs.com/mkivenson/sentiment-reviews
setwd('C:/Users/krish/Documents/Masters/Hackathons/Project Competition')

library(dplyr)
library(stringr)
library(tidytext)
library(textstem)
library(ggplot2)

drug.review.df <- read.csv('./drugsReview.csv', stringsAsFactors = FALSE, header = TRUE)

# Part 3 - Sentiment analysis of condition - drug combination

# length(unique(unlist(strsplit(drug.review.df$cleanReview, ' '))))

words.df <- drug.review.df %>% 
  select(c('uniqueID', 'condition', 'drugName', 'rating', 'reviewCount', 'cleanReview')) %>% 
  unnest_tokens(word, cleanReview)

# Afinn sentiment
afinn.df <- get_sentiments('afinn') %>% 
  mutate(word = lemmatize_words(word)) %>%
  unique()

# review.afinn.df <- words.df %>% inner_join(afinn.df, by = 'word')
# 
# condition.drug.summ.afinn.df <- review.afinn.df %>%
#   group_by(condition, drugName, reviewCount) %>%
#   summarise(mean_rating = mean(rating), sentiment = mean(value))
# 
# cor(condition.drug.summ.afinn.df$sentiment, condition.drug.summ.afinn.df$mean_rating)


# Bing sentiment
bing.df <- get_sentiments('bing') %>% 
  mutate(word = lemmatize_words(word)) %>% 
  rename(value =  sentiment) %>%
  unique()

bing.df$value <- ifelse(bing.df$value == 'positive', 1, -1)
# 
# review.bing.df <- words.df %>% 
#   inner_join(bing.df, by = 'word')
# 
# condition.drug.summ.bing.df <- review.bing.df %>%
#   group_by(condition, drugName, reviewCount) %>%
#   summarise(mean_rating = mean(rating), sentiment = mean(value))
# 
# cor(condition.drug.summ.bing.df$sentiment, condition.drug.summ.bing.df$mean_rating)


# Afinn-Bing sentiment
afinn.bing.df <- rbind(afinn.df, bing.df[bing.df$word %in% setdiff(bing.df$word, afinn.df$word),])
afinn.bing.df <-afinn.bing.df [!(afinn.bing.df$word == 'work' & afinn.bing.df$value < 0),]

review.afinn.bing.df <- words.df %>% 
  inner_join(afinn.bing.df, by = 'word')

condition.drug.summ.afinn.bing.df <- review.afinn.bing.df %>%
  group_by(condition, drugName, reviewCount) %>%
  summarise(mean_rating = mean(rating), sentiment = mean(value))

cor(condition.drug.summ.afinn.bing.df$sentiment, condition.drug.summ.afinn.bing.df$mean_rating)


# review.summ.afinn.bing.df <- review.afinn.bing.df %>%
#   group_by(uniqueID) %>%
#   summarise(sentiment = mean(value)) %>% 
#   inner_join(drug.review.df, by = 'uniqueID') %>% 
#   select(c('rating', 'sentiment'))
# 
# cor(review.summ.afinn.bing.df$sentiment, review.summ.afinn.bing.df$mean_rating)


y_mid = 0
x_mid = 5.5

condition.drug.summ.afinn.bing.df %>% 
  mutate(quadrant = case_when(mean_rating > x_mid & sentiment > y_mid ~ 'Positive Review / Postive Sentiment',
                              mean_rating <= x_mid & sentiment > y_mid ~ 'Negative Review / Positive Sentiment',
                              mean_rating <= x_mid & sentiment <= y_mid ~ 'Negative Review / Negative Sentiment',
                              TRUE ~ 'Positive Review / Negative Sentiment')) %>% 
  ggplot(aes(x = mean_rating, y = sentiment, color = quadrant)) + 
  geom_hline(yintercept = y_mid, color = 'black', size = 0.5) + 
  geom_vline(xintercept = x_mid, color = 'black', size = 0.5) +
  guides(color = FALSE) +
  scale_color_manual(values=c('blue', 'red', 'red','blue')) +
  ggtitle('Mean User Rating vs Mean Sentiment Value of Review for Drug - Condition Combination') +
  annotate('text', x = 7, y = 1, label = 'Positive Review / Postive Sentiment') +
  annotate('text', x = 2.5, y = 1, label = 'Negative Review / Positive Sentiment') +
  annotate('text', x = 7, y = -1, label = 'Positive Review / Negative Sentiment') +
  annotate('text', x = 2.5, y = -1, label = 'Negative Review / Negative Sentiment') +
  geom_point() +
  theme(plot.title = element_text(hjust = 0.5))


# Handling negation using bigrams
bigrams.df <- drug.review.df %>% 
  select(c('uniqueID', 'condition', 'drugName', 'rating', 'reviewCount', 'cleanReview')) %>% 
  unnest_tokens(ngram, cleanReview, token = 'ngrams', n = 2) %>% 
  filter(startsWith(ngram, 'not ') | 
           startsWith(ngram, 'no ') | 
           startsWith(ngram, 'never ') | 
           startsWith(ngram, 'without ')) %>% 
  rename(word =  ngram)

bigrams_separated <- as.data.frame(str_split_fixed(bigrams.df$word, " ",2))
colnames(bigrams_separated) <- c("word1", "word2")

not_words <- bigrams_separated %>%
  filter(word1 == "not") %>%
  inner_join(afinn.bing.df, by = c(word2 = "word")) %>%
  count(word2, value, sort = TRUE)

never_words <- bigrams_separated %>%
  filter(word1 == "never") %>%
  inner_join(afinn.bing.df, by = c(word2 = "word")) %>%
  count(word2, value, sort = TRUE)

no_words <- bigrams_separated %>%
  filter(word1 == "no") %>%
  inner_join(afinn.bing.df, by = c(word2 = "word")) %>%
  count(word2, value, sort = TRUE)

without_words <- bigrams_separated %>%
  filter(word1 == "without") %>%
  inner_join(afinn.bing.df, by = c(word2 = "word")) %>%
  count(word2, value, sort = TRUE)

#plot for not words
not_words %>%
  mutate(contribution = n * value) %>%
  arrange(desc(abs(contribution))) %>%
  head(20) %>%
  mutate(word2 = reorder(word2, contribution)) %>%
  ggplot(aes(word2, n * value, fill = n * value > 0)) +
  geom_col(show.legend = FALSE) +
  xlab("Words preceded by \"not\"") +
  ylab("Sentiment value * number of occurrences") +
  coord_flip()

#plot for never words
never_words %>%
  mutate(contribution = n * value) %>%
  arrange(desc(abs(contribution))) %>%
  head(20) %>%
  mutate(word2 = reorder(word2, contribution)) %>%
  ggplot(aes(word2, n * value, fill = n * value > 0)) +
  geom_col(show.legend = FALSE) +
  xlab("Words preceded by \"never\"") +
  ylab("Sentiment value * number of occurrences") +
  coord_flip()

#plot for no words
no_words %>%
  mutate(contribution = n * value) %>%
  arrange(desc(abs(contribution))) %>%
  head(20) %>%
  mutate(word2 = reorder(word2, contribution)) %>%
  ggplot(aes(word2, n * value, fill = n * value > 0)) +
  geom_col(show.legend = FALSE) +
  xlab("Words preceded by \"no\"") +
  ylab("Sentiment value * number of occurrences") +
  coord_flip()

#plot for without words
without_words %>%
  mutate(contribution = n * value) %>%
  arrange(desc(abs(contribution))) %>%
  head(20) %>%
  mutate(word2 = reorder(word2, contribution)) %>%
  ggplot(aes(word2, n * value, fill = n * value > 0)) +
  geom_col(show.legend = FALSE) +
  xlab("Words preceded by \"without\"") +
  ylab("Sentiment value * number of occurrences") +
  coord_flip()

# Handling negation in reviews
drug.review.df$cleanReview <- gsub('\\S+@\\S+', '', gsub("not ", "not@", drug.review.df$cleanReview))
drug.review.df$cleanReview <- gsub('\\S+@\\S+', '', gsub("no ", "no@", drug.review.df$cleanReview))
drug.review.df$cleanReview <- gsub('\\S+@\\S+', '', gsub("never ", "never@", drug.review.df$cleanReview))
drug.review.df$cleanReview <- gsub('\\S+@\\S+', '', gsub("without ", "without@", drug.review.df$cleanReview))

unigrams.df <- drug.review.df %>% 
  select(c('uniqueID', 'condition', 'drugName', 'rating', 'reviewCount', 'cleanReview')) %>% 
  unnest_tokens(word, cleanReview)

bigrams.df <- rbind(bigrams.df, unigrams.df)

# Updating sentiment corpus
not.afinn.bing.df <- afinn.bing.df
not.afinn.bing.df$word <- paste0('not ', afinn.bing.df$word)
not.afinn.bing.df$value <- afinn.bing.df$value * -1
bi.afinn.bing.df <- rbind(afinn.bing.df, not.afinn.bing.df)

no.afinn.bing.df <- afinn.bing.df
no.afinn.bing.df$word <- paste0('no ', afinn.bing.df$word)
no.afinn.bing.df$value <- afinn.bing.df$value * -1
bi.afinn.bing.df <- rbind(bi.afinn.bing.df, no.afinn.bing.df)

never.afinn.bing.df <- afinn.bing.df
never.afinn.bing.df$word <- paste0('never ', afinn.bing.df$word)
never.afinn.bing.df$value <- afinn.bing.df$value * -1
bi.afinn.bing.df <- rbind(bi.afinn.bing.df, never.afinn.bing.df)

without.afinn.bing.df <- afinn.bing.df
without.afinn.bing.df$word <- paste0('without ', afinn.bing.df$word)
without.afinn.bing.df$value <- afinn.bing.df$value * -1
bi.afinn.bing.df <- rbind(bi.afinn.bing.df, without.afinn.bing.df)

review.bi.afinn.bing.df <- bigrams.df %>% 
  inner_join(bi.afinn.bing.df, by = 'word')

# Afinn-Bing sentiment after handling negation terms
condition.drug.summ.bi.afinn.bing.df <- review.bi.afinn.bing.df %>%
  group_by(condition, drugName, reviewCount) %>%
  summarise(mean_rating = mean(rating), sentiment = mean(value))

cor(condition.drug.summ.bi.afinn.bing.df$sentiment, condition.drug.summ.bi.afinn.bing.df$mean_rating)

y_mid = 0
x_mid = 5.5

condition.drug.summ.bi.afinn.bing.df %>% 
  mutate(quadrant = case_when(mean_rating > x_mid & sentiment > y_mid ~ 'Positive Review / Postive Sentiment',
                              mean_rating <= x_mid & sentiment > y_mid ~ 'Negative Review / Positive Sentiment',
                              mean_rating <= x_mid & sentiment <= y_mid ~ 'Negative Review / Negative Sentiment',
                              TRUE ~ 'Positive Review / Negative Sentiment')) %>% 
  ggplot(aes(x = mean_rating, y = sentiment, color = quadrant)) + 
  geom_hline(yintercept = y_mid, color = 'black', size = 0.5) + 
  geom_vline(xintercept = x_mid, color = 'black', size = 0.5) +
  guides(color = FALSE) +
  scale_color_manual(values=c('blue', 'red', 'red','blue')) +
  ggtitle('Mean User Rating vs Mean Sentiment Value of Review for Drug - Condition Combination\n
          (Common Negation Terms Control)') +
  annotate('text', x = 7, y = 1, label = 'Positive Review / Postive Sentiment') +
  annotate('text', x = 2.5, y = 1, label = 'Negative Review / Positive Sentiment') +
  annotate('text', x = 7, y = -1, label = 'Positive Review / Negative Sentiment') +
  annotate('text', x = 2.5, y = -1, label = 'Negative Review / Negative Sentiment') +
  geom_point() +
  theme(plot.title = element_text(hjust = 0.5))

# rm(list = ls(all.names = TRUE))
# gc()