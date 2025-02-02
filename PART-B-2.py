import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


df = pd.read_csv('solar_efficiency_temp.csv')

print(df.head())

X = df[['temperature']]
y = df[['efficiency']]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train,y_train)

plt.scatter(X_train, y_train, color='red')
plt.xlabel('X-Train set')
plt.ylabel('Y-Train set')
plt.show()

plt.scatter(X_test, model.predict(X_test), color='blue')
plt.xlabel('X-Test set')
plt.ylabel('Y-Test set')
plt.show()

X = sm.add_constant(X)
model = sm.OLS(y,X).fit()

f_stat = model.fvalue
f_p_stat = model.f_pvalue

t_stat = model.tvalues['temperature']
t_p_stat = model.pvalues['temperature']

print(f"F-statistic: {f_stat:.2f}")
print(f"t-statistic for temperature: {t_stat:.2f}")
