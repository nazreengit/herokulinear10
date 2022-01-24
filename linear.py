
import pandas as pd

import matplotlib.pyplot as plt
from sklearn import linear_model
import joblib

df = pd.read_csv(r'C:\\Users\\Admil\Desktop\\homepretection.csv')
df

df.shape

df.plot(kind='scatter',x='price',y='area',color ='green')

df.corr()

area = pd.DataFrame(df['area'])
price = pd.DataFrame(df['price'])

area

price

lm = linear_model.LinearRegression()
model = lm.fit(area,price)

model.coef_
model.intercept_

model.score(area,price)

area_new = [[3000]]
price_predict=model.predict(area_new)
price_predict

print(price_predict)

vars = 'model_file.sav'
joblib.dump(model,vars)











