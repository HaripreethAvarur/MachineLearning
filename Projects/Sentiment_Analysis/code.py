'''-----Load Files-----'''
import pandas as pd
train_df = pd.read_csv("D:\\myProjects\\Sentiment Analysis\\Data\\train.csv")
test_df = pd.read_csv("D:\\myProjects\\Sentiment Analysis\\Data\\test.csv")

'''-----Data Preprocessing-----'''
import re
import nltk
from nltk.corpus import stopwords # Stopwords ex.: of, it, is, and, the, in, a, etc.
from nltk.stem import PorterStemmer

nltk.download("stopwords")

def preprocessing(text):
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text.lower())
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(tokens)

train_df["processed_tweet"] = train_df["tweet"].apply(preprocessing)
test_df["processed_tweet"] = test_df["tweet"].apply(preprocessing)

'''-----Feature Extraction-----'''
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_df["processed_tweet"])
X_test = vectorizer.transform(test_df["processed_tweet"])
y_train = train_df["label"]

'''-----Data Spliting-----'''
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

'''-----Model Selection-----'''
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

'''-----Model Training-----'''
model.fit(X_train, y_train)

'''-----Model Evaluation-----'''
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_val_pred = model.predict(X_val)

print("Accuracy:", accuracy_score(y_val, y_val_pred))
print("Classification Report:\n", classification_report(y_val, y_val_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))

'''-----Inference-----'''
y_test_pred = model.predict(X_test)

