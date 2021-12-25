from flask import Flask, jsonify, request
from model import get_prediction

app = Flask(__name__)

@app.route("/predict-alphabet", methods = ['POST'])
def predicted_data():
    image = request.files.get('alphabet')
    prediction = get_prediction(image)
    return jsonify({
        "prediction": prediction
    }), 200
    
if(__name__ == '__main__'):
    app.run(debug = True)       