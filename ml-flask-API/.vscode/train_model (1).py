import pandas as pd
import pickle

a=pd.read_csv(r"C:\Users\dhivy\insurance.csv")
a

a= pd.get_dummies(a, columns=['region'], drop_first=True)

a['sex'] = a['sex'].map({'male': 1, 'female': 0})
a['smoker'] = a['smoker'].map({'yes': 1, 'no': 0})

a.head()

x=a[['age','bmi','children','region_northwest','region_southeast','region_southwest','smoker','sex']]
y=a['charges']
x

#model selection
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
x_train


#linear regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)


pickle.dump(lr, open("model.pkl", "wb"))

y_pred=lr.predict(x_test)
y_pred




#accuracy measure
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

