from flask import Flask
#import pandas as pd
import joblib

app = Flask(__name__)

#reviews_data = pd.read_csv('Sample30.csv')
xgb_model = joblib.load('Pickles/final_gb_model.pkl')

@app.route('/')
def home():

    return 'hellow world'


if __name__ == '__main__':
    app.run(debug=True)