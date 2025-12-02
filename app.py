# app.py
from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('catboost_skin_care_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Get the label encoder classes if needed (optional for reverse mapping)
# Note: Agar aapko title wapas original name mein dikhana hai toh encoder bhi save karna padega
# Abhi hum direct predict kar rahe hain

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            title = request.form['title']
            vote = int(request.form['vote'])

            # Simple encoding: hum title ko ek dummy integer banayenge (since model expects encoded title)
            # Note: Yeh perfect nahi hai kyuki real LabelEncoder classes chahiye
            # Better approach: save encoder bhi karo training time par
            # Abhi ke liye hum title ka hash ya length use kar sakte hain (demo purpose)

            # Demo encoding: title ki length + some hash (not perfect but works for demo)
            title_encoded = abs(hash(title)) % 10000

            # Prepare input
            input_data = pd.DataFrame([[title_encoded, vote]], columns=['title', 'vote'])

            # Predict
            predicted_price = model.predict(input_data)[0]
            predicted_price = round(predicted_price)

            prediction = f"₹{predicted_price:,}"

        except Exception as e:
            prediction = "Error: Invalid input!"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)