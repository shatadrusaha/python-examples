'''                     Import libraries.                       '''
from fastapi import FastAPI, Form
import uvicorn
import joblib
import polars as pl


'''                     Create the app.                       '''
app_fastapi = FastAPI()

# Load the model.
filename_model = 'model_rfc.pkl'
model_rfc = joblib.load(filename_model)
filename_html = 'flask_app.html'

# Welcome page.
@app_fastapi.get('/')
def welcome():
    return """
    <h1>Hello from 'docker-demo-with-python-uv'!</h1>
    <h2>Welcome to the Iris Flower Classification FastApi App!</h2>
    <p>This app classifies Iris flowers based on their features.</p>
    """
    # return render_template(filename_html)

# Prediction page.
@app_fastapi.post('/predict')
def predict_iris(
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
    ):
    # Get the input values from the form.
    # sepal_length = float(request.form['sepal_length'])
    # sepal_width = float(request.form['sepal_width'])
    # petal_length = float(request.form['petal_length'])
    # petal_width = float(request.form['petal_width'])
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
    return {
        "predicted_class": predicted_class
    }
    # return render_template(filename_html, predicted_class=predicted_class)

if __name__ == "__main__":
    # uvicorn.run(app=app_fastapi, host="0.0.0.0", port=5000)
    uvicorn.run(app=app_fastapi, host="0.0.0.0")

# uvicorn app_fastapi:app_fastapi --host 0.0.0.0 --port 5000 --reload
# uvicorn app_fastapi:app_fastapi --reload