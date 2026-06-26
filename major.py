import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Download NLTK data (run once)
nltk.download('stopwords')
nltk.download('wordnet')

# 1. Load Data
df = pd.read_csv('disaster_tweets_data(DS) (1).csv')


# 2. Check and handle null values
print(df.isnull().sum())
df = df.dropna()

# 3. Text Preprocessing

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Convert to lower case
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize words
    words = text.split()
    # Remove stop words
    words = [word for word in words if word not in stop_words]
    # Stemming or Lemmatization (choose one)
    # words = [stemmer.stem(word) for word in words]  # For stemming
    words = [lemmatizer.lemmatize(word) for word in words]  # For lemmatization
    return ' '.join(words)

df['clean_tweets'] = df['tweets'].apply(preprocess_text)

# 4. Vectorization: Choose either CountVectorizer or TfidfVectorizer
vectorizer = TfidfVectorizer()  # or CountVectorizer()
X = vectorizer.fit_transform(df['clean_tweets'])

# 5. Feature and Target
y = df['target']

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Models
models = {
    'Multinomial Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier()
}

results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"Model: {model_name}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = accuracy
    print(f"Accuracy: {accuracy}\n")

# 8. Best Model
best_model = max(results, key=results.get)
print(f"Best model based on accuracy: {best_model} with accuracy {results[best_model]:.4f}")
