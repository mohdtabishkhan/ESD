import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import joblib
import wikipediaapi

app = Flask(__name__)

# Load the pre-trained model
model = load_model('epilepsy_seizure_detection_model.h5')

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Define the class meanings
class_meanings = {0: "healthy", 1: "interictal", 2: "ictal"}

# Initialize Wikipedia API with a proper user agent
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='EpilepsySeizureDetectionApp/1.0 (myemail@example.com)'
)

def get_wikipedia_summary(query):
    page = wiki_wiki.page(query)
    if page.exists():
        return page.summary
    else:
        return "No information available on Wikipedia."

# Home route to upload a file
@app.route('/')
def upload_file():
    return render_template('index.html')

# Route to handle file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file part")
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No selected file")
    if file:
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)
        
        try:
            # Load and preprocess the uploaded file
            segment = np.loadtxt(filepath)
            segment_length = 4097
            if len(segment) > segment_length:
                segment = segment[:segment_length]
            elif len(segment) < segment_length:
                segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
            
            # Normalize the example input using the same scaler used for the training data
            segment = scaler.transform([segment])
            segment = segment.reshape(segment.shape[0], segment.shape[1], 1)

            # Predict using the loaded model
            prediction = model.predict(segment)
            predicted_class = np.argmax(prediction, axis=1)[0]
            result = class_meanings[int(predicted_class)]

            # Fetch Wikipedia summary
            wiki_summary = get_wikipedia_summary(result)

            return render_template('index.html', result=result, wiki_summary=wiki_summary)
        except Exception as e:
            return render_template('index.html', error="An error occurred while processing the file")

if __name__ == "__main__":
    # Ensure the uploads directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
