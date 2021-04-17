import numpy as np
from flask import Flask, request, jsonify
from model import standadization
#for prdocution we use gavent library
#from gevent.pywsgi import WSGIServer
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
_, _, _, scaler = standadization()

@app.route('/predict',methods=['POST'])
def results():

    data = request.get_json(force=True)
    pred_data = [np.array(list(data.values()))]
    new_pred_data = scaler.transform(pred_data)

    if(len(pred_data[0]) != 19):
        return jsonify({'InputErorr':'invalid number of features given'})
    else:
        prediction = model.predict(new_pred_data)
        print(prediction)
        output = prediction[0]
        return jsonify({'prediction': int(output)})
    

if __name__ == "__main__":
    app.run(debug=True)