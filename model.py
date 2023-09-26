import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
df=pd.read_csv("car_data.csv")
final_dataset=df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven','Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
final_dataset["Current_Year"]=2023
final_dataset["No_of_Year"]=final_dataset["Current_Year"]-final_dataset["Year"]
final_dataset.drop(["Year","Current_Year"],inplace=True,axis=1)
final_dataset=pd.get_dummies(final_dataset,drop_first=True)
x=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,:1]
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestRegressor()
model.fit(x_train,y_train)
pred=model.predict(x_test)
print(pred)


import pickle
pickle.dump(model,open("model.pkl","wb"))


