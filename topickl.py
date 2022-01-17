import pandas as pd
import joblib
reviews_data = pd.read_csv('DataFiles/Sample30.csv')
reviews_data = reviews_data[['name', 'reviews_rating', 'reviews_text', 'user_sentiment']]
joblib.dump(reviews_data, 'Pickles/reviews_data.pkl')
