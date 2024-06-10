from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import pandas as pd
import io

app = Flask(__name__)

# Load the trained model
model = load_model('cat_age_model.keras')

# Load the StandardScaler
scaler_full = joblib.load('scaler_full.pkl')

# Define label map for multi-class
label_map = {0: 'adult', 1: 'kitten', 2: 'senior'}

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if a file is uploaded
        if 'file' not in request.files:
            return render_template('error.html', message="No file part")

        file = request.files['file']

        if file.filename == '':
            return render_template('error.html', message="No selected file")

        if file:
            # Read the uploaded file content
            file_content = file.read().decode('utf-8')

            # Use StringIO to handle the file content as a stream
            file_stream = io.StringIO(file_content)

            try:
                # Try to read the file as a CSV
                df = pd.read_csv(file_stream, header=None)

                # Check if the file contains only one row or multiple rows
                if df.shape[0] == 1:
                    # Process as single row
                    data = df.iloc[0].values
                    X_demo = data[:-1].reshape(1, -1)
                    y_demo = int(data[-1])

                    # Scale the input data
                    input_data_scaled = scaler_full.transform(X_demo)

                    # Make predictions
                    probabilities = model.predict(input_data_scaled)
                    predicted_class = np.argmax(probabilities[0])

                    # Map predictions and actual labels to classes
                    mapped_prediction = label_map[predicted_class]
                    mapped_actual_label = label_map[y_demo]

                    # Convert probabilities to a dictionary for easy iteration in the template
                    probabilities_dict = {label_map[i]: round(prob, 4) for i, prob in enumerate(probabilities[0])}

                    print(probabilities_dict)

                    # Prepare the response
                    response = {
                        'mapped_prediction': mapped_prediction,
                        'prediction': round(probabilities[0][predicted_class], 2),
                        'actual_label': mapped_actual_label,
                        'predicted_class': predicted_class,
                        'probabilities': probabilities_dict
                    }

                    return render_template('result.html',
                                           mapped_prediction=response['mapped_prediction'],
                                           prediction=response['prediction'],
                                           actual_label=response['actual_label'],
                                           predicted_class=response['predicted_class'],
                                           probabilities=response['probabilities'])
                else:
                    # Process as multiple rows
                    X_demo = df.iloc[:, :-1].values
                    y_demo = df.iloc[:, -1].values

                    # Scale the input data
                    input_data_scaled = scaler_full.transform(X_demo)

                    # Make predictions
                    probabilities = model.predict(input_data_scaled)

                    results = []
                    correct_predictions = {0: 0, 1: 0, 2: 0}
                    total_samples = {0: 0, 1: 0, 2: 0}

                    for i in range(len(probabilities)):
                        actual_class = y_demo[i]
                        total_samples[actual_class] += 1
                        predicted_class = np.argmax(probabilities[i])
                        mapped_prediction = label_map[predicted_class]
                        mapped_actual_label = label_map[actual_class]

                        if predicted_class == actual_class:
                            correct_predictions[actual_class] += 1

                        # Convert probabilities to a dictionary for easy iteration in the template
                        probabilities_dict = {label_map[j]: round(prob, 4) for j, prob in enumerate(probabilities[i])}

                        result = {
                            'mapped_prediction': mapped_prediction,
                            'prediction': round(float(probabilities[i][predicted_class]), 2),
                            'actual_label': mapped_actual_label,
                            'predicted_class': predicted_class,
                            'probabilities': probabilities_dict
                        }
                        results.append(result)

                    accuracy_percentages = {
                        0: round((correct_predictions[0] / total_samples[0]) * 100, 2) if total_samples[0] > 0 else 0,
                        1: round((correct_predictions[1] / total_samples[1]) * 100, 2) if total_samples[1] > 0 else 0,
                        2: round((correct_predictions[2] / total_samples[2]) * 100, 2) if total_samples[2] > 0 else 0
                    }

                    return render_template('multi_result.html', results=results,
                                           accuracy_percentages=accuracy_percentages, label_map=label_map)

            except pd.errors.EmptyDataError:
                return render_template('error.html', message="No data found in file")
            except pd.errors.ParserError:
                return render_template('error.html', message="File parsing error. Please upload a valid CSV file.")

    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run(debug=True, port=5002)
