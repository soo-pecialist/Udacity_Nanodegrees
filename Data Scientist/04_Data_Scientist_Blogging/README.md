# Predicting Airbnb Price in Seattle and Boston
You can see my blog on Medium: https://medium.com/@soopecialist/this-will-make-you-know-how-much-you-need-to-travel-with-airbnb-442abd51c8c4

## Table of Contents
1. [Installation](#installation)
2. [File Descriptions](#files)
3. [Project Motivation](#motivation) 
4. [Findings](#findings)
5. [More Details on Concepts](#concepts)
6. [Acknowledgements](#acknowledgements)


## Installation <a name="installation"></a>
The libraries/packages used are:

- pandas
- numpy
- matplotlib
- seaborn
- sklearn
- xgboost
- ast
- os

There should be no impediments to running the code beyond the Anaconda distribution of Python. My Python version is 3.6.6, Jupyter Notebook 5.6.0, and Anaconda 1.7.2


## File Descriptions <a name="files"></a>
- Predicting Airbnb Price in Seattle and Boston.ipynb: You can see my code and detailed analysis in this notebook. The notebook adheres to CRSP-DM process (Cross Industry Process for Data Mining).
- Predicting Airbnb Price in Seattle and Boston.html: The static html version of the notebook
- helper.py: This python script contains all modularized functions for the analysis conducted in the notebook.
- data/boston: Contains 'listings.csv' file about Boston Airbnb infromation I used.
- data/seattle: Contains 'listings.csv' file about Boston Airbnb infromation I used.
- data/image: Contains all images produced and used in the notebook.


## Project Motivation <a name="motivation"></a>
As a travel lover, I hoppoed around many places with Airbnb. Visiting different cities, I have observed the Airbnb price varies. My goal was "understanding Airbnb price". I used [Seattle] and [Boston] Airbnb datasets. Below are questions I had on mind:

1. Is there actually a price difference in two cities?
2. How many Airbnb properties are owned by the same host?
3. How does the price spread based on location — evenly or unevenly?
4. What are the most important predictors for the price?

[Seattle]: https://www.kaggle.com/airbnb/seattle
[Boston]: https://www.kaggle.com/airbnb/boston


## Findings <a name="findings"></a>
These are brief summary of what I found:

1. Boston is more expensive city to travel with Airbnb than Seattle. This phenomenon complies with the current median rent per month trend in two cities — Boston is more expensive to rent a property.
2. Hosts in Boston own more Airbnb properties on average and there are more ‘super’ hosts. Therefore, we might imply that a few super hosts can sway average price in the city.
3. High proportion locations correspond to high rent areas in both Seattle and Boston. Therefore, we may assume rent, location, and Airbnb price are closely related.
4. From important features, we know that popularity, location, level of comfort, and property rules impact on the Airbnb price.


## More Details on Concepts <a name="concepts"></a>
I employed the notorious Extreme Gradient Boosting (a.k.a. XGBoost) regressor as a framework together with exhaustive grid search cross validation

As the name reveals, XGBoost uses Gradient Boosting approach in Ensemble techniques. I will not get very technical, but in case there are some folks struggling with jargons, here is is how I understand Boosting, Ensemble, and Gradient Boosting.

- Ensemble is a collection of multiple learning algorithms. Intuition behind the Ensemble method is a certain decision tree model can be hardly trusted since it can make good predictions with a particular problem but poor with others. Therefore, ensembles combine multiple hypothesis (weak learners) to form a better, strong learner that gives more stable predictors. Think of this way: a Congressman doesn't decide a policy, but 435 Congressmen and Congresswomen together for a better result.
- Boosting is a sequential technique which works on the principle of an ensemble. It combines a set of weak learners and delivers improved prediction accuracy. It is actually very similar to our experiences! When you work on a team project, you need to check which members are falling behind at each step. The team needs extra care and push for the member headed in the wrong direction. Likewise, in Boosting model if we observe data points wrongly predicted in each step, we put more weights on those points so the model corrects the prediction.
- Gradient Boosting is a part of Boosting techniques. The objective of Gradient Boosting is minimizing the loss from wrongly predicting the value. Using gradient descent and updating our predictions based on a learning rate, we can find the values where the loss is minimum. It is like coming down a mountain through shortest paths. Gradient Boosting models want to find a pattern to make the loss minimum because it believes less model performs better.

XGBoost is one of the most popular machine learning algorithm these days and its - 1) speed and performance, 2) parallelizable core algorithm, 3) outperforming other algorithm methods in many cases, and 4) wide variety of tuning parameters - helps its popularity among data geeks.


## Acknowledgements <a name="acknowledgements"></a>
You can find the Licensing for the data and other descriptive information at the Kaggle link available [here](https://www.kaggle.com/airbnb/seattle/home) and [here](https://www.kaggle.com/airbnb/boston/home)

You may use the code here as you would here. We data scientists share!
