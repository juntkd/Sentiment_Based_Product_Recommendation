from flask import Flask
#import pandas as pd
import joblib

app = Flask(__name__)

df = joblib.load('Pickles/reviews_data.pkl')

@app.route('/')
def home():

    return 'hellow world'


if __name__ == '__main__':
    app.run(debug=True)