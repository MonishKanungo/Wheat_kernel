from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os
import logging

# Initialize the Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
handler = logging.FileHandler('app.log')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Load the SVM model and StandardScaler from the pickle file
try:
    with open('Wheat_Kernel_Classification_Final.pkl', 'rb') as file:
        classifier, sc = pickle.load(file)
    logger.info('SVM model and scaler loaded successfully.')
except Exception as e:
    logger.error(f'Error loading model or scaler: {e}')
    raise

# Define the home route to display the input form
@app.route('/')
def home():
    logger.info('Home page accessed.')
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = [float(x) for x in request.form.values()]
        logger.info(f'Received input data: {data}')

        # Convert data into a numpy array and scale it
        final_data = sc.transform(np.array(data).reshape(1, -1))
        logger.debug(f'Scaled data: {final_data}')

        # Predict using the SVM classifier
        prediction = classifier.predict(final_data)
        logger.info(f'Raw prediction: {prediction}')

        # Map the prediction to the correct wheat kernel class
        if prediction[0] == 0:
            result = 'Kama'
        elif prediction[0] == 1:
            result = 'Rosa'
        else:
            result = 'Canadian'
        
        logger.info(f'Final prediction result: {result}')

        # Render the result
        return render_template('result.html', prediction_text=f'Wheat Kernel Class is: {result}')

    except Exception as e:
        logger.error(f'Error during prediction: {e}')
        return str(e)

# Run the Flask app
if __name__ == "__main__":
    # Bind to 0.0.0.0 and use PORT from the environment, with a fallback to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
