# https://medium.com/@oyewusiwuraola/opinion-mining-using-the-uci-drug-review-data-set-part-1-data-loading-and-pre-processing-using-49d3fb6025a8
# https://medium.com/@oyewusiwuraola/opinion-mining-using-the-uci-drug-review-data-set-part-2-sentiment-prediction-using-a-machine-f9f7e5a0f4ec
# https://towardsdatascience.com/my-first-adventures-in-nlp-631faa6aadd4
# https://towardsdatascience.com/sentiment-analysis-with-python-part-1-5ce197074184
# https://towardsdatascience.com/sentiment-analysis-with-python-part-2-4f71e7bde59a
# http://rpubs.com/mkivenson/sentiment-reviews
setwd('C:/Users/krish/Documents/Masters/Hackathons/Project Competition')

library(dplyr)
library(ggplot2)

drug.review.df <- read.csv('./drugsReview.csv', stringsAsFactors = FALSE, header = TRUE)

drug.df <- drug.review.df[, c('condition', 'drugName')] %>% 
  group_by(condition, drugName) %>% 
  summarise(reviewCount = n())

drug.review.df$drugName <- as.factor(drug.review.df$drugName)
drug.review.df$condition <- as.factor(drug.review.df$condition)
drug.review.df$rating <- as.factor(drug.review.df$rating)


# Part 3 - Exploration

# (1) Top 10 conditions
condition.freq.ten.df <- as.data.frame(table(drug.review.df$condition)) %>% 
  arrange(desc(Freq)) %>% 
  top_n(10)
ggplot(data = condition.freq.ten.df, aes(x = Var1,y = Freq)) + 
  geom_bar(stat = 'identity') + 
  ylab('Number of Reviews') + 
  xlab('Condition') + 
  ggtitle('Top 10 Conditions') + 
  theme(plot.title = element_text(hjust = 0.5))

# (2) Top 5 common drugs prescribed for each of top 10 conditions
drug.ten.condition.list <- list()
for (i in condition.freq.ten.df$Var1) {
  drug.ten.condition.list <- append(drug.ten.condition.list, 
                                    list(drug.df[drug.df$condition == i,][,c('drugName', 'reviewCount')] %>% 
                                           arrange(desc(reviewCount)) %>% 
                                           top_n(5)))
}
names(drug.ten.condition.list) <- condition.freq.ten.df$Var1
rm(i)

# (3) Broad spectr(um drugs which are used for 10 or more conditions
broad.spectrum.ten.list <- list()
for (i in as.list(drug.df %>% group_by(drugName) %>% 
                  tally() %>% 
                  filter(n >= 10) %>% 
                  select(drugName))$drugName) {
  broad.spectrum.ten.list <- append(broad.spectrum.ten.list, 
                                    list(drug.df[drug.df$drugName == i,]$condition))
}
names(broad.spectrum.ten.list) <- as.list(drug.df %>% group_by(drugName) %>% 
                                            tally() %>% 
                                            filter(n >= 10) %>% 
                                            select(drugName))$drugName
rm(i)

# (4) Unique conditions with only one drug prescription
unique.condition.drug.df <- drug.df[, c('condition', 'drugName')] %>% 
  inner_join(drug.df %>% 
               group_by(condition) %>% 
               tally() %>% 
               filter(n == 1) %>% 
               select(condition))

# (5) Percentage frequency of ratings
rating.percent.freq.df <- as.data.frame(prop.table(table(drug.review.df$rating))*100)
ggplot(data = rating.percent.freq.df, aes(x = Var1,y = Freq)) + 
  geom_bar(stat = 'identity') + 
  ylab('Percentage of Rating') + 
  xlab('Rating') + 
  ggtitle('Percentage Frequency of Ratings') + 
  theme(plot.title = element_text(hjust = 0.5))

rm(list = ls(all.names = TRUE)) 
gc()