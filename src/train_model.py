import pandas as pd
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the cleaned dataset
df = pd.read_csv('../dataset/cleaned_spam.csv')

# Ensure that 'cleaned_message' does not contain any NaN values
df['cleaned_message'] = df['cleaned_message'].fillna('')

# Load the pre-trained vectorizer
vectorizer = pickle.load(open('../models/vectorizer.pkl', 'rb'))

# Transform the cleaned message using the loaded vectorizer
X = vectorizer.transform(df['cleaned_message'])
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report: \n", classification_report(y_test, y_pred))

# Save the trained model
with open('../models/naive_bayes_model.pkl', 'wb') as f:
    pickle.dump(model, f)
