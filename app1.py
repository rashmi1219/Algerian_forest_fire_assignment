import pickle
from flask import Flask, app, request, jsonify
import numpy as np
import pandas as pd
app =Flask(__name__)
model = pickle.load(open('model2.pkl', 'rb'))


@app.route('/predict_single_output', methods=['POST'])
def predict_single_output():
    data = request.json['data']
    new_data = [list(data.values())]
    output = model.predict(new_data)[0]
    print(output)
    return str(output)
@app.route('/predict_bulk_output', methods=['POST'])
def predict_bulk_output():
    data1 = request.json['data']
    print(data1)
    df = pd.DataFrame.from_dict(data1, orient='index')
    #print(df.head())
    predict = model.predict(df)
    print(predict)
    return str(predict)



if __name__ == "__main__":
    app.run()