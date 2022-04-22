import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
lin = LinearRegression()

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 4)

from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score

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

lin.fit(x_train,y_train)

print("Hệ số hồi quy", lin.coef_)
print("Hệ số chắn:", lin.intercept_)
print(f"Công thức y= mX+c : [chi phí(Y)] = [Hệ số hồi quy(m)] x [x_train(X)] +[Hệ số chắn(c)] ")

y_pred_lin=lin.predict(x_test)
test_data['Lin_CPDD']=y_pred_lin
  
X_poly = poly.fit_transform(x_train)  
poly.fit(X_poly,y_train)  
lin.fit(X_poly,y_train)

y_pred_poly=lin.predict(poly.fit_transform(x_test))
test_data['Poly_CPDD']=y_pred_poly

print(test_data)

lin_mse = mean_squared_error(y_test,y_pred_lin,squared=True)
print("Tổng bình phương sai số trên tập mẫu SSE hay MSE mô hinh LinearRegressionlà: ", lin_mse)
poly_mse = mean_squared_error(y_test,y_pred_poly,squared=True)
print("Tổng bình phương sai số trên tập mẫu SSE hay MSE mô hinh Polynomial Regression là: ",poly_mse)

n=len(y_test)
print('Tổng số biến quan sát là n : ',n)
import math
lin_rmse = math.sqrt(lin_mse/n)
print('Giá trị độ đo RMSE mô hinh LinearRegression là: ', lin_rmse)
poly_rmse = math.sqrt(poly_mse/n)
print('Giá trị độ đo RMSE mô hinh Polynomial Regression là: ', poly_rmse)

lin_mae = mean_absolute_error(y_test,y_pred_lin )
print('Giá trị độ đo MAE mô hinh LinearRegression là: ', lin_mae)
poly_mae = mean_absolute_error(y_test,y_pred_poly)
print('Giá trị độ đo MAE mô hinh Polynomial Regression là: ', poly_mae)

lin_r2_score = r2_score(y_test,y_pred_lin)
print('Giá trị độ đo R2 mô hinh LinearRegression là: ', lin_r2_score)
poly_r2_score= r2_score(y_test,y_pred_poly)
print('Giá trị độ đo R2 mô hinh Polynomial Regression là: ', poly_r2_score)




