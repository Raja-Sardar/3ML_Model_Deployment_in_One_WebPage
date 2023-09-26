import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
file=load_iris()
x=file.data
y=file.target
x=pd.DataFrame(x,columns=file.feature_names)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


from sklearn.linear_model import LogisticRegression
model_leaf=LogisticRegression()
model_leaf.fit(x_train,y_train)
pred_logistic_reg=model_leaf.predict(x_test)
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,pred_logistic_reg)
print(score)

#Dumping the model object
import pickle
pickle.dump(model_leaf,open("model_leaf.pkl","wb"))

