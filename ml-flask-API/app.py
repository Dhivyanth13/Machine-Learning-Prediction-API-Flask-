from flask import Flask, request, jsonify
import pickle
from flask import render_template

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features'"}), 400

    try:
        features = data["features"]

    except:
        return jsonify({"error": "Invalid input"}), 400

    prediction = model.predict([features])

    return jsonify({
        "prediction": round(float(prediction[0]), 2)
    })

if __name__ == "__main__":
    app.run(debug=True)


