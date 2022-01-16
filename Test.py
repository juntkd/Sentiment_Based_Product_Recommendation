import joblib
import Model as mdl
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from joblib import load

def getRecommendation(UserName)

    reviews_data = pd.read_csv("DataFiles/Sample30.csv")
    df_rec = joblib.load('Pickles/final_recomm_model.pkl')
    xgb_model = joblib.load('Pickles/final_gb_model.pkl')
    tfidf = joblib.load('Pickles/tfidf_vectorizer.pkl')
    #'nmm2592'
    user_rec = df_rec.loc[UserName].sort_values(ascending=False)[0:20]
    pred_df = reviews_data[reviews_data.name.isin(user_rec.index)]
    #print(user_rec)

    #print(pred_df)


    pred_df = mdl.Preprocess_Data(pred_df)
    review_text = pred_df['reviews_text']
    review_text = tfidf.transform(review_text)
    #print(review_text)


    pred_sntmnt = xgb_model.predict(review_text)
    pred_df['pred_sntmnt'] = pred_sntmnt
    pos_sent_pcnt_df = pred_df[['name', 'pred_sntmnt']] .groupby('name')['pred_sntmnt'].mean()
    final_recmnd_pdts = pos_sent_pcnt_df.sort_values(ascending=False)[0:5].index.tolist()
    return final_recmnd_pdts


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        user = request.form['search']
        return redirect(url_for('success', name=user))


@app.route('/success/<name>')
def success(name):
    return "<xmp>" + str(requestResults(name)) + " </xmp> "


if __name__ == '__main__' :
    app.run(debug=True)