import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

nltk.download('stopwords')

# Load dataset with utf-8-sig to handle BOM
df = pd.read_csv('../dataset/spam.csv', encoding='utf-8-sig')

# Clean up the column names (remove spaces and make lowercase)
df.columns = df.columns.str.strip().str.lower()

# Check the cleaned column names
print("Columns in the dataset:", df.columns)

# Handle missing values in 'message' column by filling NaNs with an empty string
df['message'] = df['message'].fillna('')

# Make sure the 'label' and 'message' columns exist
if 'label' not in df.columns or 'message' not in df.columns:
    raise KeyError("Columns 'label' and 'message' are required in the dataset")

# Select necessary columns
df = df[['label', 'message']]

# Convert labels to binary (spam=1, ham=0)
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Text cleaning function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = "".join([char for char in text if char not in string.punctuation])  # Remove punctuation
    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])  # Remove stopwords
    return text

# Apply the cleaning function to the 'message' column
df['cleaned_message'] = df['message'].apply(clean_text)

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['cleaned_message'])
y = df['label']

# Save vectorizer
with open('../models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Save processed data
df.to_csv('../dataset/cleaned_spam.csv', index=False)
print("Data Preprocessing Done!")
