from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("car_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Extract only 6 features (as used during training)
        present_price = float(request.form["present_price"])
        km_driven = int(request.form["km_driven"])
        fuel = int(request.form["fuel"])
        seller_type = int(request.form["seller_type"])
        transmission = int(request.form["transmission"])
        owner = int(request.form["owner"])

        # Create input array for prediction
        input_data = np.array([[present_price, km_driven, fuel, seller_type, transmission, owner]])
        prediction = model.predict(input_data)[0]

        return render_template("index.html", prediction_text=f"Predicted Selling Price: ₹{int(prediction):,}")

if __name__ == "__main__":
    app.run(debug=True)
