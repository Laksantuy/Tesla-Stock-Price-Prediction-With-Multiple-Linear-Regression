# Importing Essential Modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Path to the csv dataset
datapath = "C:\\Tesla.csv"

# Loading the csv file
df = pd.read_csv(datapath)

# Printing the first 3 rows of the dataset
print(df.head(3))

# Printing statistical information of the dataset
print(df.describe())

# Checking for empty data fields in the dataset
print(df.isnull().sum())

# Checking data types of all columns
print(df.info())

# Dropping date column from our dataset
df = df.drop(['Date'], axis=1)

# Checking whether or not the date column has been dropped
print(df.head(3))

# Creating a shadow copy
copydf = df.copy()

# Seperating Dependent and Independent Variables
# Dependent Variable
y = df["Close"]

# Independent Variables
x = df.drop(["Close", "Adj Close"], axis=1)

#Splitting the data in 80%, 20% for training and testing
xTrain, xTest, yTrain, yTest = train_test_split(x,y,test_size=0.2)

#Making the model using linear regression
model = LinearRegression()

#Fitting the training data in the model
model.fit(xTrain, yTrain)

#Checking the accuracy using R2
print(model.score(xTest, yTest))

#Getting the intercept and Coefficients
print("Intercept: ",model.intercept_)
print("Coefficients: ")
for _ in model.coef_:
    print(_)

#Making predictions using the predict() and xTest data
predictions = model.predict(xTest)
comparison = pd.DataFrame({'Predicted Values':predictions,'Actual Values':yTest})
print(comparison)