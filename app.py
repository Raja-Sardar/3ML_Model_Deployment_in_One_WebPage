from flask import Flask,redirect,url_for,render_template,request
import pickle
import numpy as np

model=pickle.load(open("model.pkl","rb"))
model_leaf=pickle.load(open("model_leaf.pkl","rb"))

app=Flask(__name__)
@app.route('/')
def home():
    return render_template('index_main.html')

@app.route("/calci",methods=["GET","POST"])
def math():
    if request.method=="POST":
        a=float(request.form["A"])
        b=float(request.form["B"])
        c='Addition:- ',a+b,'Subtraction:- ',a-b,'Multiplication:- ',a*b,'Division:- ',a/b
        return render_template("index_calculator.html", result=c)
    else:
        return render_template("index_calculator.html")
    
@app.route("/price",methods=["GET","POST"])
def price():
    if request.method=="POST":
        k = float(request.form["Present_Price"])
        l = float(request.form["Kms_Driven"])
        m = float(request.form["Owner"])
        d = float(request.form["No_of_Year"])
        e = float(request.form["Fuel_Type_Diesel"])
        f = float(request.form["Fuel_Type_Petrol"])
        g = float(request.form["Seller_Type_Individual"])
        h = float(request.form["Transmission_Manual"])
        value=[k,l,m,d,e,f,g,h]
        final_features = np.array(value).reshape(1, -1)
        result=model.predict(final_features)
        k=result[0]
        x = "The Predicted Price Will Be " + str(k)
        return render_template("index_car.html", answer=x)
    else:
        return render_template("index_car.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method=="POST":
        n = float(request.form["Sepal_Length"])
        p = float(request.form["Sepal_Width"])
        q = float(request.form["Petal_Length"])
        r = float(request.form["Petal_Width"])
        int_features = [n, p, q, r]
        final_features = np.array(int_features).reshape(1, -1)
        prediction = model_leaf.predict(final_features)
        output = prediction[0]
        if output==0:
            results="The Expected Leaf Name Will Be : Iris-Setosa"
        elif output==1:
            results="The Expected Leaf Name Will Be : Iris-Versicolor"
        else:
            results="The Expected Leaf Name Will Be : Iris-Virginica"
        return render_template("index_leaf.html", prediction_text=results)
    else:
        return render_template("index_leaf.html")

if __name__ == '__main__':
    app.run(port=5000,debug=True)