import pickle
import pandas as pd

# Load the pre-trained model
model = pickle.load(open('../models/naive_bayes_model.pkl', 'rb'))

# Load the pre-trained vectorizer
vectorizer = pickle.load(open('../models/vectorizer.pkl', 'rb'))

# Function to predict whether an email is spam or not
def predict_spam(input_message):
    # Preprocess the message using the vectorizer
    processed_message = vectorizer.transform([input_message])

    # Predict using the loaded model
    prediction = model.predict(processed_message)

    # Output the prediction
    if prediction == 1:
        return "This is a Spam email."
    else:
        return "This is a Ham email."

# Main logic to take user input
choice = input("Enter 1 to input a message or 2 to provide a file path: ")

if choice == "1":
    # User inputs message directly
    input_message = input("Enter the message to be classified: ")
    result = predict_spam(input_message)
    print(result)

elif choice == "2":
    # User provides a file path
    file_path = input("Enter the path of the file containing the message: ")
    try:
        # Read the content of the file
        with open(file_path, 'r') as file:
            content = file.read()
            result = predict_spam(content)
            print(result)
    except FileNotFoundError:
        print("The specified file was not found.")
else:
    print("Invalid input. Please choose 1 or 2.")
