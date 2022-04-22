import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

train_data=pd.read_csv('Medical_Train.csv ')
test_data=pd.read_csv('Medical_Test.csv ')

train_data['GT']=train_data['Gioi_tinh'].apply(lambda x:1 if str(x)=='male' else 0)
train_data['HT']=train_data['Hut_thuoc'].apply(lambda x:1 if str(x)=='yes' else 0)
train_data['NS']=train_data['Noi_song'].apply(lambda x:1 if str(x)=='northwest' else 0)

test_data['GT']=test_data['Gioi_tinh'].apply(lambda x:1 if str(x)=='male' else 0)
test_data['HT']=test_data['Hut_thuoc'].apply(lambda x:1 if str(x)=='yes' else 0)
test_data['NS']=test_data['Noi_song'].apply(lambda x:1 if str(x)=='northwest' else 0)

x_train=train_data.loc[:,['Tuoi','GT','bmi','So_con','HT','NS']].values
y_train=train_data.loc[:,'Chi_phi'].values
x_test=test_data.loc[:,['Tuoi','GT','bmi','So_con','HT','NS']].values
y_test=test_data.loc[:,'Chi_phi'].values

lin = LinearRegression() 
  
lin.fit(x_train,y_train)
poly = PolynomialFeatures(degree = 4) 
X_poly = poly.fit_transform(x_train) 
  
poly.fit(X_poly,y_train) 
lin2 = LinearRegression() 
lin2.fit(X_poly,y_train)
test_data['CPDD']=lin2.predict(poly.fit_transform(x_test))
print(test_data['CPDD'])



