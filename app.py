from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

with open('house_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    total_sqft = float(request.form['total_sqft'])
    bath = int(request.form['bath'])
    balcony = int(request.form['balcony'])
    bedrooms = int(request.form['bedrooms'])

    input_data = np.array([[total_sqft, bath, balcony, bedrooms]])
    prediction = model.predict(input_data)[0]

    if prediction >= 100: 
        price_str = f"₹{prediction/100:.2f} Cr"
    else:
        price_str = f"₹{prediction:.2f} Lakh"

    return render_template('index.html', prediction_text=f"Estimated House Price: {price_str}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
