import joblib
import Model as mdl
import pandas as pd
from flask import Flask, render_template, request

reviews_data = pd.read_csv('DataFiles/Sample30.csv')
df_rec = joblib.load('Pickles/final_recomm_model.pkl')
xgb_model = joblib.load('Pickles/final_gb_model.pkl')
tfidf = joblib.load('Pickles/tfidf_vectorizer.pkl')

def getRecommendation(UserName):

    #'nmm2592'
    user_rec = df_rec.loc[UserName].sort_values(ascending=False)[0:20]
    pred_df = reviews_data[reviews_data.name.isin(user_rec.index)]

    pred_df = mdl.Preprocess_Data(pred_df)
    review_text = pred_df['reviews_text']
    review_text = tfidf.transform(review_text)

    pred_sntmnt = xgb_model.predict(review_text)
    pred_df['pred_sntmnt'] = pred_sntmnt
    pos_sent_pcnt_df = pred_df[['name', 'pred_sntmnt']] .groupby('name')['pred_sntmnt'].mean()
    final_recmnd_pdts = pos_sent_pcnt_df.sort_values(ascending=False)[0:5].index.tolist()

    return final_recmnd_pdts

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        user = request.form.get('username')

        try:

            prdcts = getRecommendation(user)

        except:

            prdcts = 'No User found'

        return render_template("results.html", recommendation=prdcts)


if __name__ == '__main__':
    app.run(debug=True)