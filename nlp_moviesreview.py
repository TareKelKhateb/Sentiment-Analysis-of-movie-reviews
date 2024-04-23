print("Welcome to Sentiment Analysis of Movie Reviews")

import re
import unicodedata
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import os
import glob
import pandas as pd

def load_reviews(folder_path, label):
    reviews = []
    file_paths = glob.glob(os.path.join(folder_path, '*.txt'))
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            review_text = file.read()
            reviews.append({'review': review_text, 'label': label})
    return reviews


pos_reviews = load_reviews('C:/Users/tarek/PycharmProjects/SentimentAnalysis of movie reviews/data/txt_sentoken/pos', 'positive')
neg_reviews = load_reviews('C:/Users/tarek/PycharmProjects/SentimentAnalysis of movie reviews/data/txt_sentoken/neg', 'negative')
all_reviews = pos_reviews + neg_reviews
df = pd.DataFrame(all_reviews)


df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(df.head(5))
print(df["label"].value_counts())




def preprocessing(text):
    # Deleting punctuation & special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    text = text.lower()

    tokens = word_tokenize(text)

    # Remove stopwords
    S_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in S_words]

    lemma = WordNetLemmatizer()
    tokens = [lemma.lemmatize(w) for w in tokens]

    return tokens


#print(df.loc[0,"review"])

for i, review in enumerate(df["review"]):
    df.loc[i, "review"] = preprocessing(review)


print(df.loc[0,"review"])