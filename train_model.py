import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

print("Step 1: Loading your uploaded dataset...")
# Load dataset with specific encoding safely
data = pd.read_csv('all-data.csv', encoding='latin-1', header=None)

# Unga data-la 0th column sentiment labels, 1st column news headline texts
X = data[1] # News Text Features
y = data[0] # Sentiment Target Labels

print("Step 2: Converting text elements into vector tokens numerical arrays...")
# Text mapping transformation configurations setup
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vectors = vectorizer.fit_transform(X)

# Splitting data dynamically for 80% training and 20% validation testing checks
X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)

print("Step 3: Processing machine learning model training sequences...")
# Using Logistic Regression classifier engine for high text prediction rates
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Checking overall system training test accuracy scores
accuracy = model.score(X_test, y_test)
print(f"🎯 Model Training Accuracy Score: {accuracy * 100:.2f}%")

print("Step 4: Saving trained weights pkl objects locally...")
# Export variables models for continuous pipeline predictions
joblib.dump(model, 'news_sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("AI Model Trained & Saved Successfully! 🎉 Object files generated.")