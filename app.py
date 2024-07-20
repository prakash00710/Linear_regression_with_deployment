from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the scaler and model
scaler = pickle.load(open('scal.pkl', 'rb'))
model = pickle.load(open('LR.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]  # Convert inputs to float
    final_features = [np.array(int_features)]  # Correct array creation
    scaled_data = scaler.transform(final_features)  # Transform the data
    prediction = model.predict(scaled_data)  # Make the prediction
    output = round(prediction[0], 2)  # Round the output
    
    return render_template('index.html', prediction_text='Predicted value:{}'.format(output))

if __name__ == '__main__':
    app.run(port=1800, debug=True)