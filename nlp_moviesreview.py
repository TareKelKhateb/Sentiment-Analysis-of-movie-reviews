
print("Welcome to Sentiment Analysis of Movie Reviews")

import re
import unicodedata
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
from nltk.corpus import wordnet

from matplotlib import pyplot as plt
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np



from sklearn.feature_extraction.text import TfidfVectorizer

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

    S_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in S_words]

    lemma = WordNetLemmatizer()
    tokens = [lemma.lemmatize(w) for w in tokens]
    processed_text = ' '.join(tokens)

    return processed_text


df['review'] = df['review'].apply(preprocessing)

print(df.shape)


tf_idf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tf_idf.fit_transform(df['review'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tf_idf.get_feature_names_out())
df = pd.concat([df, tfidf_df], axis=1)
print(df.columns)

#apply label encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
print(le.classes_)




from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier , GradientBoostingClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,plot_confusion_matrix,ConfusionMatrixDisplay ,classification_report



X=df.drop(['review','label'], axis=1)
Y=df['label']
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.15 ,random_state=42)


model_decision_tree = DecisionTreeClassifier()
model_RandomForest = RandomForestClassifier(n_estimators=500, random_state=42,max_depth=10)
model_AdaBoost = AdaBoostClassifier()
model_GradientBoost = GradientBoostingClassifier()

model_decision_tree.fit(X_train, y_train)
model_RandomForest.fit(X_train, y_train)
model_AdaBoost.fit(X_train, y_train)
model_GradientBoost.fit(X_train, y_train)



y_pred_decision_tree = model_decision_tree.predict(X_test)
y_pred_RandomForest = model_RandomForest.predict(X_test)
y_pred_AdaBoost = model_AdaBoost.predict(X_test)
y_pred_GradientBoost = model_GradientBoost.predict(X_test)


print("decesion tree acc %s" , accuracy_score(y_test, y_pred_decision_tree))
print("Random forest acc %s",accuracy_score(y_test, y_pred_RandomForest))
print("AdaBoost acc %s",accuracy_score(y_test, y_pred_AdaBoost))
print("GradientBoost acc %s",accuracy_score(y_test, y_pred_GradientBoost))


accuracies = [accuracy_score(y_test, y_pred_decision_tree),
              accuracy_score(y_test, y_pred_RandomForest),
              accuracy_score(y_test, y_pred_AdaBoost),
              accuracy_score(y_test, y_pred_GradientBoost)]

# Define labels for the bars
models = ['Decision Tree', 'Random Forest', 'AdaBoost', 'Gradient Boost']

# Plotting the bar plot
plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'green', 'orange', 'red'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Models')
plt.ylim(0, 1)  # Set the y-axis limits to be between 0 and 1
plt.show()



print(confusion_matrix(y_test, y_pred_RandomForest))

print("Classification report for model Random forest :")
print(classification_report(y_test, y_pred_RandomForest))

disp_rf = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_RandomForest), display_labels=['Negative', 'Positive'])
disp_rf.plot(cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.show()


import joblib

# Save the Random Forest model
joblib_file = "random_forest_model.pkl"
joblib.dump(model_RandomForest, joblib_file)

print(f"Model saved to file :  {joblib_file}")
