'''                     Import libraries.                       '''
from flask import Flask, render_template, request
import joblib
import polars as pl
# import numpy as np

'''
https://code.visualstudio.com/docs/python/tutorial-flask
'''

'''                     Create the app.                       '''
app = Flask(__name__)

# Load the model.
filename_model = 'model_rfc.pkl'
model_rfc = joblib.load(filename_model)
filename_html = 'flask_app.html'

# Welcome page.
@app.route('/')
def welcome():
    # return """
    # <h1>Hello from 'docker-demo-with-python-uv'!</h1>
    # <h2>Welcome to the Iris Flower Classification App!</h2>
    # <p>This app classifies Iris flowers based on their features.</p>
    # """
    return render_template(filename_html)

# Prediction page.
@app.route('/predict', methods=['POST'])
def predict_iris():
    # Get the input values from the form.
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Create a DataFrame from the input values.
    df_input = pl.DataFrame({
        'sepal length (cm)': [sepal_length],
        'sepal width (cm)': [sepal_width],
        'petal length (cm)': [petal_length],
        'petal width (cm)': [petal_width]
    })

    # Make a prediction using the model.
    prediction = model_rfc.predict(df_input)

    # Map the prediction to the target names.
    target_names = ['Setosa', 'Versicolor', 'Virginica']
    predicted_class = target_names[prediction[0]]

    return render_template(filename_html, predicted_class=predicted_class)

    # return render_template('predict.html')


if __name__ == "__main__":
    # app.run()
    app.run(host="0.0.0.0", port=5000)
    # app.run(debug=True)