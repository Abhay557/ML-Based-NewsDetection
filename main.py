import pandas as pd
import numpy as np

# fake_df = pd.DataFrame = pd.read_csv('fake.csv')
# print(fake_df.head())

fake_df = pd.read_csv('fake.csv')
true_df = pd.read_csv('true.csv')


# true_df = pd.DataFrame = pd.read_csv('true.csv')
# print(true_df.head())

fake_df['label'] = 0 
true_df['label'] = 1

df = pd.concat([fake_df, true_df], ignore_index=True)

print(df.head())
df.info()

duplicate_count = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")

df.drop_duplicates(inplace=True)
print(f"Number of duplicate rows after removal: {df.duplicated().sum()}")


import string
import nltk
from nltk.corpus import stopwords

try:
    stopwords.words('english')
except LookupError:
    print("Downloading stopwords...")
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text_final(text):
    # 1. Lowercase and remove punctuation (from our previous step)
    text = text.lower()
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    # 2. Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)
    
    return text

df['text'] = df['text'].apply(preprocess_text_final)

#print(df['text'].head(1).iloc[0])

from sklearn.model_selection import train_test_split

# Identify our features (X) and our target (y)
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Size of training set: {len(X_train)}")
print(f"Size of testing set: {len(X_test)}")

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TF-IDF Vectorizer
# We'll set a max_features limit to keep the vocabulary size manageable.
# This means it will only use the 10,000 most frequent words.
vectorizer = TfidfVectorizer(max_features=10000)

X_train_tfidf = vectorizer.fit_transform(X_train)


X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF vectorization complete.")
print(f"Shape of the training data matrix: {X_train_tfidf.shape}")
print(f"Shape of the testing data matrix: {X_test_tfidf.shape}")


from sklearn.linear_model import PassiveAggressiveClassifier

# Initialing the PassiveAggressiveClassifier
model = PassiveAggressiveClassifier(max_iter=1000, random_state=42)


# Train the model on the training data
model.fit(X_train_tfidf, y_train)
print("Model training complete. Meow!!")

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("TESTING!!")
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy Score: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


import joblib

# Save the vectorizer
joblib.dump(vectorizer, 'vectorizer.pkl')

# Save the model
joblib.dump(model, 'model.pkl')

print("\nVectorizer and model saved to files.")