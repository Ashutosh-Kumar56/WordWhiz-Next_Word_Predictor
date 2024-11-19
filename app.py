
from flask import Flask, render_template, request
import pickle

# Load model and vectorizer
with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('form.html')  # Render the input form



@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sequence = request.form['sequence'].lower()  # Get input from form
        
        # Vectorize input and predict
        sequence_vector = vectorizer.transform([sequence])
        predicted_word = model.predict(sequence_vector)[0]
        
        return render_template('form.html', prediction=f"Next word: {predicted_word}")

if __name__ == '__main__':
    app.run(debug=True)





