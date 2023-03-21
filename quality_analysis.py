import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report




def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub('[^a-zA-Z]+', ' ', text)
    # Convert to lowercase
    text = text.lower()
    return text



data = pd.read_csv('dataset/product_reviews/original/amazon_reviews.txt', delimiter='\t')
data['REVIEW_TEXT'] = data['REVIEW_TEXT'].apply(preprocess_text)

X = data['REVIEW_TEXT']
y = data['LABEL']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


classifier = MultinomialNB()
classifier.fit(X_train_vec, y_train)

y_pred = classifier.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

misclassified = X_test[y_test != y_pred]

for index, text in misclassified.items():
    print(f"Index: {index}, Text: {text}\n")


