from flask import Flask, request, render_template
from joblib import load
import pandas as pd

# Membuat instance Flask
app = Flask(__name__)

# Memuat model
model = load('xgboost_sani.joblib')

# Memuat objek preprocessing
label_encoder = load('label_encoder.joblib')
onehot_encoder = load('onehot_encoder.joblib')
min_max_scaler = load('min_max_scaler.joblib')

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

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
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
        preprocessed_data = preprocess_input(data, label_encoder, onehot_encoder, min_max_scaler)
        trained_model_columns = [
            "CreditScore", "Gender", "Age", "Balance", "NumOfProducts", 
            "IsActiveMember", "EstimatedSalary", "Geography_France", 
            "Geography_Germany", "Geography_Spain"
        ]
        preprocessed_data = preprocessed_data[trained_model_columns]
        prediction = int(model.predict(preprocessed_data)[0])

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
