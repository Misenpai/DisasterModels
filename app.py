from flask import Flask,request,jsonify
import numpy as np
import pickle

modelEarthquake = pickle.load(open("earthquake.pkl",'rb'))


app = Flask(__name__)
@app.route('/')
def index():
    return "Hello World"


@app.route('/predict/earthquake',methods=['POST'])

def predictEarthquake():
    latitude = request.form.get("latitude")
    longitude = request.form.get("longitude")
    depth = request.form.get("depth")

    input_query = np.array([[latitude,longitude,depth]])
    result = modelEarthquake.predict(input_query)[0]
    return jsonify({"Maginitude":str(result)})

if __name__ == "__main__":
    app.run(debug=True)