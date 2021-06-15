# DSI 21 Project 3: Reddit Classifier

## Project Overview:

Fake news and misinformation have become such an integral feature of social media that it can no longer be avoided. According to researchers at MIT Sloan, fake news can be defined as content that is entirely fabricated and often partisan that is presented as factual. The issue with misinformation is that it spreads at a much faster and wider rate than true information. According to a 2019 study published in Science by MIT, fake news is 70% more likely to retweeeted on Twitter than the truth, and reach their first 1500 people at 6 times the rate. In another study conducted by Columbia university, 59% of shared links are unclicked. The spread of misinformation can be detrimental and lead to the creation of conspiracy theories and division.

A subcategory of misinformation or fake news is satire. Satire is defined as the use of humor to expose and criticize people's stupidity in the context of contemperoray politics and other topical issues. Due to how big of a part social media plays in our daily lives, satirical media manages to take up its own space on the internet as well. Satirical news is however, sometimes mistaken for real news.

Reddit, a social news aggregation and discussion website, consists of many sub-forums called subreddits where people can post any kind of content that they like. Though there are many factually correct subreddits, satirical ones exist as well.

In order to combat the spread of misinformation and fake news, we have decided to come up with an algorithm to classify satirical and non-satrical news on Reddit using the posts from r/TheOnion and r/worldnews. The Onion is a US-based satirical news website which posts satirical articles which look like real news. The aim of this project is to reduce the spread of misinformation using machine learning algorithm. In this case, I plan on exploring the data and building a model to minimize the rate of incorrect satire prediction and properly identify if a headline is satire or real news and properly identify the subreddits they belong to.


#### Data Dictionary
|Feature|Type|Dataset|Description|
|---|---|---|---|
|subreddit|object|onion_posts.json|Name of the subreddit|
|title|object|onion_posts.json|Title of post|
|wordcount|int|onion_posts.json|Number of words in title of post|
|charcount|int|onion_posts.json|Number of characters in title of post|
|subreddit|object|news_posts.json|Name of the subreddit|
|title|object|news_posts.json|Title of post|
|wordcount|int|news_posts.json|Number of words in title of post|
|charcount|int|news_posts.json|Number of characters in title of post|
|word|str|top_words_news.csv|Word taken from title using CountVectorizer|
|count|int|top_words_news.csv|Number of times the word appears in data set|
|frequency|float|top_words_news.csv|Frequency of how often the word appears in the data set|
|word|str|top_words_onion.csv|Word taken from title using CountVectorizer|
|count|int|top_words_onion.csv|Number of times the word appears in data set|
|frequency|float|top_words_onion.csv|Frequency of how often the word appears in the data set|
|bigram|str|top_bigrams_news.csv|Bigrams taken from title using CountVectorizer|
|count|int|top_bigrams_news.csv|Number of times the bigram appears in data set|
|frequency|float|top_bigrams_news.csv|Frequency of how often the bigram appears in the data set|
|bigram|str|top_bigrams_onion.csv|Bigram taken from title using CountVectorizer|
|count|int|top_bigrams_onion.csv|Number of times the bigram appears in data set|
|frequency|float|top_bigrams_onion.csv|Frequency of how often the bigram appears in the data set|
|subreddit|object|combined_posts.json|Name of the subreddit|
|title|object|combined_posts.json|Title of post|
|wordcount|int|combined_posts.json|Number of words in title of post|
|charcount|int|combined_posts.json|Number of characters in title of post|
|news|int|combined_posts.json|Classifier for which subreddit the post belongs to 1 for news 0 for onion|
|BestModelScore|float|scores.csv|Best score for corresponding model|
|TrainScore|float|scores.csv|Train score for corresponding model|
|TestScore|float|scores.csv|Test score for corresponding model|
|TrueNeg|float|scores.csv|Number of true negative posts matched from the test set in the model|
|FalsePos|float|scores.csv|Number of false positives posts matched from the test set in the model|
|FalseNeg|float|scores.csv|Number of false negative posts matched from the test set in the model|
|TruePos|float|scores.csv|Number of true positives posts matched from the test set in the model|
|Accuracy|float|scores.csv|Accuracy Score of the corresponding model (tp+tn)/(tp+fp+fn+tn)|
|Sensitivity|float|scores.csv|Sensitivity score for corresponding model tp/(tp+fn)|
|Specifity|float|scores.csv|Specificity score for corresponding model tn/(tn+fp)|
|Precision|float|scores.csv|Precision score for corresponding model tp/(tp+fp)|
|F1|float|scores.csv|F1 score for corresponding model|
|ROUAUC|float|scores.csv|ROUAUC score for corresponding model|
|log|str|scores.csv|Model name for logistic regression with countvectorizer transformer|
|log2|str|scores.csv|Model name for logistic regression with countvectorizer transformer with charcount and wordcount as features|
|log_gs|str|scores.csv|Model name for logistic regression with countvectorizer transformer using gridsearch|
|log_gs2|str|scores.csv|Model name for logistic regression with countvectorizer transformer with char/wordcount as features using grid search|
|tvec_log|str|scores.csv|Model name for logistic regression with tfidfvectorizer transformer|
|tvec_log_gs|str|scores.csv|Model name for logistic regression with tfidfvectorizer transformer using grid search|
|nb|str|scores.csv|Model name for Naive Bayes Model with countvectorizer transformer|
|tvec_nb|str|scores.csv|Model name for Naive Bayes Model with tfidfvectorizer transformer|
|nb_gs|str|scores.csv|Model name for Naive Bayes Model with countvectorizer transformer using grid search|
|tvec_nb_gs|str|scores.csv|Model name for Naive Bayes Model with tfidfvectorizer transformer using grid search|
|rf|str|scores.csv|Model name for Random Forest Model with countvectorizer transformer|
|rf|str|scores.csv|Model name for Random Forest Model with tfidfvectorizer transformer|

## Structure

## 1.1 Scraping Reddit Data

Using pushshift API to scrape reddit posts, I was able to scrape about 2500 posts from each subreddit in order to build a dataset to build my models on.

##  1.2/1.3 Data Cleaning

Cleaning the scraped data and identifying which features to be used for the exploratory data analysis

## 2.1 EDA: Exploratory Data Analysis

Finding general observations about the data, feature building, finding top words and bigrams using CountVectorizer() to finalize the dataset that will be used for modelling

## 3.1 Modelling

A total of 3 models were used to model the data. Logistic Regression, Random Forest, Logistic Regression. A total of 2 transformers were used CountVectorizer and TfidfVectorizer to transform the data to fit the models.

## Conclusion

The main goal of this project was to build a model that could differentiate the headlines of satirical news and real news. In order to find the best model, it was important to run different version of the model with different transformations and features. Ideally we wanted to optimize for both specificity and sensitity to reduce both of the false positive and negatives. However, since it is more important to reduce the satire that is being mistaken for real news, minimizing false positives was favorable over optimizing for both.
Future recommendations would include collecting more data to have a larger data set to work with, including the contents of the article as features and possibly article length and word count as well. Also collecting more data from other satirical news articles would help improve our model as well. The distribution of the articles in terms of the content was definitely slightly biased since posts from r/worldnews included articles from many different news sources while posts from r/TheOnion were exclusively from the onion.

There definitely is a different kind of danger involved with categorizing news as satire but not to the degree that categorizing satire as news has considering the statistics of how quickly misinformation spreads.




