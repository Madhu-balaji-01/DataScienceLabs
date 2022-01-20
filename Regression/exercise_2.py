import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectPercentile, f_regression

# Loading data
dataset = pd.read_csv('chicago_hotel_reviews (2).csv')
X = dataset['review']
vectorizer = CountVectorizer()
X_tfidf = vectorizer.fit_transform(X)
y = dataset['rating']


# Without feature selection
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_tfidf,y,test_size=0.2)
lr_1 = LinearRegression()
lr_1.fit(X_train_1, y_train_1)
y_pred_1 = lr_1.predict(X_test_1)
error_1 = mean_squared_error(y_test_1, y_pred_1)
print("Mean squared error without feature selection:", error_1)


# With feature selection
X_new = SelectPercentile(f_regression, percentile=10).fit_transform(X_tfidf, y)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_new,y,test_size=0.2)
lr_2 = LinearRegression()
lr_2.fit(X_train_2, y_train_2)
y_pred_2 = lr_2.predict(X_test_2)
error_2 = mean_squared_error(y_test_2, y_pred_2)
print("Mean squared error with feature selection:", error_2)