from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PassiveAggressiveRegressor

app = Flask(__name__)

data = pd.read_csv("Instagram data.csv", encoding='latin1')

x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']])
y = np.array(data['Impressions'])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    model_choice = request.form.get("model")

    if model_choice == "RandomForest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = PassiveAggressiveRegressor(random_state=42)

    model.fit(xtrain, ytrain)

    features = np.array([[ 
        int(request.form["likes"]),
        int(request.form["saves"]),
        int(request.form["comments"]),
        int(request.form["shares"]),
        int(request.form["visits"]),
        int(request.form["follows"])
    ]])

    reach = int(model.predict(features)[0])
    return render_template("index.html", prediction=reach)

if __name__ == "__main__":
    app.run(debug=True)
