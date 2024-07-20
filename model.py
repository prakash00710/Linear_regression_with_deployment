import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
R=pd.read_csv('Real_Estate.csv')
R.drop(columns=['Transaction date'], inplace=True)
X=R.iloc[:,:-1]
Y=R.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)
from sklearn.preprocessing import StandardScaler
Sc=StandardScaler()
X_train_scaled=Sc.fit_transform(X_train)
X_test_scaled=Sc.transform(X_test)
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X_train_scaled,Y_train)
pickle.dump(Sc,open('scal.pkl','wb'))
pickle.dump(LR,open('LR.pkl','wb'))