# Project 2: Ames Housing Data and Kaggle Challenge

### Problem Statement

Suppose you were working at a newly established real estate company that plans to buy and flip properties. The CEO tasks you to find the top 10 features of a house that affects its sale price.

This project aims to answer this question by going through various models to see which one performs the best.

### Approach

#### Step 1: Cleaning the Data

The data had many null values which I had to explore in order to imput a proper value. Most of the null values were due to the lack of said feature (eg. if a house did not have a pool, its Pool QC which is an ordinal measure of the pool's quality, would be a null value). In these cases, I would imput a 0.

The data had many string value columns as well which will be handled in the feature engineering section.


#### Step 2: Feature Engineering and Selection

Next, I took a look at the correlation between features with a heatmap and dropped features with high correlation with each other to try to reduce multi-collinearity. For example, the number of bedrooms had a large correlation with the overall living space, which makes logical sense because the higher the number of bedroom, the higher the overall living space. Thus, the more specific of these feature pairings (in this case the number of bedrooms) would be dropped.

I also took a look at pairs of features that could be grouped together and binarized them.


#### Step 3: Visualization

Checking the summary statistics, I found several features with high standard deviation. I binarized features with way too many 0 values and removed some outliers by looking at histograms. I then used boxplots to check for remaining outliers.


#### Step 4: Modelling

I used 4 kinds of models: linear, ridge, LASSO and ElasticNet. For Ridge, LASSO and ElasticNet, I tuned the hyperparameters and got LASSO with an alpha of 200 as the best performing model.

This model received a RMSE score of 31763.91 on Kaggle.


### Conclusion and Statement


Using the coefficients as a measure, the result was that area features like General Living Area, Garage Area, Lot Area and Basement Area were the top contributors to price. Also, the quality of the house and the age of the house were also important factors.