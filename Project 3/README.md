# Project 3: Subreddit Classifier

### Problem Statement

Reddit is a very diverse collection of forums, divided categorically into smaller forums called subreddits. The main purpose of the subreddit, however, tends to tread a very fine line. For example, /r/stocks and /r/wallstreetbets are both about stocks, but the direction the posts tend to head to are quite different.

Imagine that Reddit would like to venture into the news industry. The news industry would feed them articles, and Reddit would automatically propagate them to subreddits via a bot, where they think people will be interested in clicking on them. But they also need to strike a balance so that people will not treat it as spam and thus get annoyed by their website. 

The final bot should be able to take in the headlines of articles and predict what kind of subreddits they should be propagated into, without using the subject of the article.

This project aims to train a classifier that will serve as the basis of such a bot, by classifying what kind of subreddit a length of text (news article header) will fit in based on the contents of posts that were already on that subreddit.


### Background

The two subreddits chosen are **/r/magicTCG** and **/r/mtgfinance**. While both are related, the latter discusses the cards' financial prices while the former discusses gameplay and the metagame. What is interesting is that both subreddits tend to also contain posts you would more commonly find in the other. For example, it is possible to find finance discussions on /r/magicTCG and metagame discussions on /r/mtgfinance (the moderators try to prevent this from happening, but the system isn't perfect). 

If the classifier was able to classify posts between these two subreddits with a good accuracy, it should be able to distinguish between most subreddits as well.


### Approach

#### Step 1: Cleaning the Data

From the data pulled using the PushShift API, I first removed duplicate posts and crossposts from the two subreddits. I then also removed purely pictorial posts and rows with null values. Finally, I removed urls from the text in the title and post text. I ended up with 900+ posts for each subreddit.


#### Step 2: EDA

I took a preliminary look at the effect that lemmatization and stemming would have on the number of unique words in the data. Conclusion was that lemmatization and stemming would have an impact on the model results.

I also took a look at the unigrams, bigrams and trigrams to see if there were anything strange. After adding some reddit tags and metadata to the list of stop words, they began to make sense.


#### Step 3: Modelling

I first use the logistic regression model as a base model to verify if lemmatization and stemming had a positive effect on model results. It was true. Also, I found that using a transformer like CountVectorizer or TF-IDF gave better results than not using one at all.

Next, I tried out several other models like Naive Bayes, Bagging, RandomForest and AdaBoost. With the exception of Naive Bayes, none of the rest came close to logistic regression.


### Conclusion and Statement


Even though the stemmed TF-IDF logistic regression model did not have the best specificity or sensitivity score, it did have the lowest number of misclassifications. As the problem statement did not require a minimization of false negatives or positives, we can conclude that the stemmed TF-IDF logistic regression model is the model of choice.