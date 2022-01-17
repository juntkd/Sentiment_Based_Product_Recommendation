import pandas as pd
import numpy as np
import warnings
from nltk.tokenize import word_tokenize
#from nltk.corpus import stopwords
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics.pairwise import pairwise_distances

warnings.filterwarnings("ignore")
#reviews_data = pd.read_csv("DataFiles/Sample30.csv")
reviews_data = joblib.load('Pickles/reviews_data.pkl')
stop = joblib.load('Pickles/englsh_stopwords.pkl')
#stop = stopwords.words('english')


def clean_text(text):

    # change sentence to lower case
    text = text.lower()
    text = text.replace('[^\w\s]','')

    # tokenize into words
    words = word_tokenize(text)

    # remove stop words
    words = [word for word in words if word not in stopwords.words("english")]

    #if stem:
        #words = [stemmer.stem(word) for word in words]
    #else:
        #words = [wordnet_lemmatizer.lemmatize(word, pos='v') for word in words]

    # join words to make sentence
    text = " ".join(words)

    return text


def Preprocess_Data(df_ip):

    df_cleaned = df_ip[['name', 'reviews_rating', 'reviews_text', 'user_sentiment']]
    df_cleaned.loc[df_cleaned.user_sentiment.isnull(), 'user_sentiment'] = 'Positive'
    df_cleaned.loc[df_cleaned.reviews_rating > 3, 'user_sentiment'] = 'Positive'
    df_cleaned.loc[df_cleaned.reviews_rating < 3, 'user_sentiment'] = 'Negative'
    df_cleaned['reviews_text'] = df_cleaned['reviews_text'].str.lower()
    df_cleaned['reviews_text'] = df_cleaned['reviews_text'].str.replace('[^\w\s]', '')
    df_cleaned['reviews_text'] = df_cleaned['reviews_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    #df_cleaned['reviews_text'] = df_cleaned['reviews_text'].apply(clean_text)
    return df_cleaned



def train_classification_model(input_df):
    final_df = Preprocess_Data(input_df)
    x = final_df['reviews_text']
    y = final_df['user_sentiment'].map({'Positive': 1, 'Negative': 0})
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    word_vectorizer = TfidfVectorizer(
        strip_accents='unicode',
        # Remove accents and perform other character normalization during the preprocessing step.
        analyzer='word',  # Whether the feature should be made of word or character n-grams.
        token_pattern=r'\w{1,}',
        # Regular expression denoting what constitutes a “token”, only used if analyzer == 'word'
        stop_words='english',
        sublinear_tf=True)

    word_vectorizer.fit(X_train)
    joblib.dump(word_vectorizer, 'Pickles/tfidf_vectorizer.pkl')

    X_train_transformed = word_vectorizer.transform(X_train.tolist())
    #X_test_transformed = word_vectorizer.transform(X_test.tolist())

    sm = SMOTE()

    X_train_transformed_sm, y_train_sm = sm.fit_resample(X_train_transformed, y_train)

    xgb_clsfr = xgb.XGBClassifier()
    xgb_clsfr.fit(X_train_transformed_sm, y_train_sm)
    return xgb_clsfr


def recommendation_model(df):

    Reccom_df = df[['name', 'reviews_rating', 'reviews_username']].drop_duplicates()
    Reccom_df = Reccom_df[Reccom_df.reviews_username.notnull()]
    df_train, df_test = train_test_split(Reccom_df, test_size=0.3, random_state=42)
    dummy_train = df_train.copy()
    dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)
    dummy_train = dummy_train.pivot_table(index='reviews_username',
    columns='name', values='reviews_rating', fill_value=1)

    df_pivot = df_train.pivot_table(
    index='reviews_username',
    columns='name',
    values='reviews_rating'
    )

    user_correlation = 1 - pairwise_distances(df_pivot.fillna(0), metric='cosine')
    user_correlation[np.isnan(user_correlation)] = 0
    user_correlation[user_correlation<0]=0
    user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))
    user_final_rating = np.multiply(user_predicted_ratings,dummy_train)

    return user_final_rating


if __name__ == '__main__':

    final_model = train_classification_model(reviews_data)
    recomm_model = recommendation_model(reviews_data)

    joblib.dump(final_model, 'Pickles/final_gb_model.pkl')
    joblib.dump(recomm_model, 'Pickles/final_recomm_model.pkl')


