from flask import Flask, request, jsonify
from joblib import load
import pandas as pd

# Membuat instance Flask
app = Flask(__name__)

# Memuat model
model = load('xgb_model.joblib')

# Memuat objek preprocessing
label_encoder = load('label_encoder.joblib')
onehot_encoder = load('onehot_encoder.joblib')
min_max_scaler = load('min_max_scaler.joblib')

import pandas as pd

def preprocess_input(data, label_encoder, onehot_encoder, min_max_scaler):
    # Convert dictionary to DataFrame
    # Providing an index since the data contains scalar values
    data_df = pd.DataFrame([data])

    # Check and apply Label Encoding for 'Gender', if needed
    if 'Gender' in data_df and data_df['Gender'].dtype == 'object':
        data_df['Gender'] = label_encoder.transform(data_df['Gender'])

    # Check and apply One-Hot Encoding for 'Geography', if needed
    if 'Geography' in data_df and data_df['Geography'].dtype == 'object':
        geography_data = onehot_encoder.transform(data_df[['Geography']])
        geography_columns = onehot_encoder.get_feature_names_out(['Geography'])
        geography_df = pd.DataFrame(geography_data, columns=geography_columns)
        data_df = pd.concat([data_df.drop(['Geography'], axis=1), geography_df], axis=1)

    # Apply Min-Max Scaling for numerical features
    numeric_features = ['CreditScore', 'Age', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    if set(numeric_features).issubset(data_df.columns):
        data_df[numeric_features] = min_max_scaler.transform(data_df[numeric_features])

    return data_df

@app.route('/')
def index():
    return '''
    <form action="/predict" method="post">
        Credit Score: <input type="number" name="CreditScore"><br>
        Geography: <select name="Geography">
                       <option value="France">France</option>
                       <option value="Germany">Germany</option>
                       <option value="Spain">Spain</option>
                   </select><br>
        Gender: <select name="Gender">
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                </select><br>
        Age: <input type="number" name="Age"><br>
        Balance: <input type="number" step="0.01" name="Balance"><br>
        Number of Products: <input type="number" name="NumOfProducts"><br>
        Is Active Member: <select name="IsActiveMember">
                              <option value="1">Yes</option>
                              <option value="0">No</option>
                          </select><br>
        Estimated Salary: <input type="number" step="0.01" name="EstimatedSalary"><br>
        <input type="submit" value="Submit">
    </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form submission
    form_data = request.form
    data = {
        "CreditScore": float(form_data['CreditScore']),
        "Gender": form_data['Gender'],
        "Age": float(form_data['Age']),
        "Balance": float(form_data['Balance']),
        "NumOfProducts": float(form_data['NumOfProducts']),
        "IsActiveMember": int(form_data['IsActiveMember']),
        "EstimatedSalary": float(form_data['EstimatedSalary']),
        "Geography": form_data['Geography']
    }

    # Preprocess and predict
    preprocessed_data = preprocess_input(data, label_encoder, onehot_encoder, min_max_scaler)
    trained_model_columns = ["CreditScore", "Gender", "Age", "Balance", "NumOfProducts", "IsActiveMember", "EstimatedSalary", "Geography_France", "Geography_Germany", "Geography_Spain"]
    preprocessed_data = preprocessed_data[trained_model_columns]
    prediction = model.predict(preprocessed_data)

    # Return prediction result
    return jsonify({'prediction': int(prediction[0])})


if __name__ == '__main__':
    app.run(debug=True)
