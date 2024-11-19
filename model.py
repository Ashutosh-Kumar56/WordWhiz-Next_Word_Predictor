import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import numpy as np
# import re
from sklearn.feature_extraction.text import CountVectorizer



from sklearn.naive_bayes import MultinomialNB
import pickle


def load_sentence_from_file(file_path):
    with open(file_path, 'r') as file:
        sentences = file.readlines()
        sentences = [sentence.strip() for sentence in sentences]
    return sentences



# Sample dataset of sentences
# sentences = [
#     "I like to eat apples",
#     "She likes to eat oranges",
#     "We like to eat fruit",
#     "They love to eat bananas",
#     "I enjoy reading books",
#     "She enjoys listening to music",
#     "We enjoy learning new things"
# ]



def preprocess_text(text):

    text = text.lower()  # Convert to lowercase

    tokens = nltk.word_tokenize(text)  # Tokenize

    stop_words = set(stopwords.words('english'))  # Load stop words

    filtered_tokens = [w for w in tokens if w not in stop_words and w.isalnum()]  # Remove stop words and non-alphanumeric characters
    # filtered_tokens = [w for w in tokens if w.isalnum()]

    return filtered_tokens




file_path = "data.txt"
sentences = load_sentence_from_file(file_path)
# sentences = [preprocess_text(sentence) for sentence in sentences]
# print("Preprocessed Sentence: ", sentences)





# Apply preprocessing
sentences = [preprocess_text(sentence) for sentence in sentences]
# print("Preprocessed Sentences:", sentences) --Printing


sequence_data = []
next_words = []




for sentence in sentences:
    if isinstance(sentence, list):
        sentence = " ".join(sentence)  # Join words if it's a list
    words = sentence.split()
    for i in range(1, len(words)):
        sequence_data.append(" ".join(words[:i]))  # Sequence of words
        next_words.append(words[i])

# print(sequence_data)  #--Printing
# print(next_words)     #--Printing


# Convert sequences to numerical format using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sequence_data)
y = np.array(next_words)

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# print("Shape of Training Data:", X_train.shape)       --Printing
# print("Shape of Test Data:", X_test.shape)            --Printing


from sklearn.naive_bayes import MultinomialNB


#Check best fitting
# Train a Multinomial Naive Bayes model
model = MultinomialNB()
# print(model)                                  --Printing
model.fit(X_train, y_train)
# print("Model training completed.")            --Printing
model.score(X_train, y_train)


from sklearn.metrics import accuracy_score

# Predict on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
# print("Model Accuracy:", accuracy)            --Printing




# Save the model and vectorizer
with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# print("Vectorizer and model saved successfully!")       --Printing


#Backend run

# def predict_next_word(sequence):
#     # Preprocess input if necessary
#     sequence = preprocess_text(sequence)  # Ensure this returns a string
    
#     print("Sequence:", sequence)
#     print("Type of sequence:", type(sequence))

#     if isinstance(sequence, list):
#         sequence = " ".join(sequence)
        
#     # Vectorize input (ensure sequence is a string, not a list)
#     sequence_vector = vectorizer.transform([sequence])  # Pass a list of strings
#     predicted_word = model.predict(sequence_vector)  # Predict next word
#     return predicted_word[0]



# # Testing the prediction function
# test_sequence = "she likes"  # Input must be a string
# predicted_word = predict_next_word(test_sequence)
# print("Predicted next word for '{}': {}".format(test_sequence, predicted_word))