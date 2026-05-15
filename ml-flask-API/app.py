from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return "ML API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)

        return jsonify({
    "prediction": float(prediction[0])
    })
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)


