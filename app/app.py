from flask import Flask, request, jsonify, render_template
from src.predict import predict

app = Flask(__name__)

# Home page (HTML form)
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict_route():    
    try:
        # Get form data (from HTML)
        input_data = request.form.to_dict()

        # Convert values to float/int
        for key in input_data:
            input_data[key] = float(input_data[key])

        result = predict(input_data)

        return render_template("index.html", prediction=int(result))

    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run(debug=True)