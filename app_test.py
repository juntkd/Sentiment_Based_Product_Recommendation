from flask import Flask
import pandas as pd

app = Flask(__name__)

reviews_data = pd.read_csv('DataFiles/Sample30.csv')

@app.route('/')
def home():

    return 'hellow world'


if __name__ == '__main__':
    app.run(debug=True)