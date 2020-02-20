#IMPORTING THE REQUIRED LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns

wine = pd.read_csv('winequality-red.csv')





#seeing a few values of the csv files
wine.head()
wine.info()
print(wine.isnull())



#since we find there is no null values we can proceed
#now plotting the data to find some good asssumptions as to best fitfig = plt.figure(figsize = (10,6))
fig = plt.figure(figsize = (10,10))
sns.barplot(x = 'quality', y = 'fixed acidity', data = wine)
plt.show()

#assigning input values to x and y
x = wine[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]
y = wine['quality'].values
#y=y*10
# showing the wine dataset in tabular cloumn
wine.describe()

#information about the wine datatypes
wine.info()



#TRAIN AND TEST SPLIT
#SPLITTING THE DATA USING SIMPLE TEST TRAIN SLIT OF DATA
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x ,y , test_size = 0.2, random_state=0)

#printing the dimensions of splitted data
print("x_train shape :", x_train.shape)
print("x_test shape : ", x_test.shape)
print("y_train shape :",y_train.shape)
print("y_test shape :", y_test.shape)

#applying linear regression model to the dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# predicting the test results
y_pred = regressor.predict(x_test)


#plotting the scatter plot  between y_test and y_predicited
plt.scatter(y_test, y_pred, c='green')
plt.xlabel("Input parameters")
plt.ylabel("Wine quality /10 ")
plt.title("True value vs predicted value : Linear Regression ")
plt.show()


#Result from the MULTI LINEAR REGRESSION MODEL
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(" Mean Square Error : ", mse)
m =math.sqrt(mse)
print(" SQUARE ROOT OF MEAN SQUARED ERROR")
print (m)
print(y_pred)
for x in range(len(y_pred)):
      if y_test[x] >= 70:
         print ('Good')
      else:
         print('Bad')

#print (" Model Accuracy :", 100-mse)


#Mean absolute error
print("test accuracy: {} %".format(100 - np.mean(np.abs(y_pred- y_test))))

#predictiing quality via giving lables



