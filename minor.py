import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 1. Load Data
df = pd.read_csv('credit_customers (DS) (1).csv')


# 2. Data Cleaning
# Check null values
print(df.isnull().sum())

# Fill or drop nulls as necessary, example:
df = df.dropna()  # or df.fillna(method='ffill') depending on data

# Remove duplicates
df = df.drop_duplicates()

# Encoding categorical features
# Identify categorical columns
cat_cols = df.select_dtypes(include=['object']).columns

# Apply Label Encoding or get dummies
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 3. Feature and Target Selection
X = df.drop('class', axis=1)  # replace 'class' with your exact target column name
y = df['class']

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optional: Feature scaling (especially for KNN and SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Models

models = {
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(),
    'SVM Linear': SVC(kernel='linear'),
    'SVM RBF': SVC(kernel='rbf')
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

# 6. Best Model
best_model = max(results, key=results.get)
print(f"Best model based on accuracy: {best_model} with accuracy {results[best_model]:.4f}")
