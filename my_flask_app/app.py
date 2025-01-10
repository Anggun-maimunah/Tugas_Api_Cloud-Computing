from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

# Inisialisasi Flask
app = Flask(__name__)

# Load model
with open("walmart_sales_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil data dari form
        store = float(request.form["store"])
        holiday_flag = float(request.form["holiday_flag"])
        temperature = float(request.form["temperature"])
        fuel_price = float(request.form["fuel_price"])
        cpi = float(request.form["cpi"])
        unemployment = float(request.form["unemployment"])

        # Format input untuk model
        input_data = np.array([[store, holiday_flag, temperature, fuel_price, cpi, unemployment]])

        # Prediksi
        prediction = model.predict(input_data)

        # Kirim hasil prediksi ke halaman
        return jsonify({"prediction": round(prediction[0], 2)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
