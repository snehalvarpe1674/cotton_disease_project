from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load best model and encoders
model = pickle.load(open("best_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# Load dataset to get dropdown values
df = pd.read_csv("cotton_data")
feature_columns = ['Crop', 'Crop Stage']

dropdown_values = {col: sorted(df[col].unique()) for col in feature_columns}

@app.route("/")
def home():
    return render_template("index.html", dropdown_values=dropdown_values)

@app.route("/predict", methods=["POST"])
def predict():
    input_data = []
    for col in feature_columns:
        value = request.form.get(col)
        # Encode using LabelEncoder
        if col in encoders:
            le = encoders[col]
            try:
                value = le.transform([value])[0]
            except:
                value = -1  # unseen category
        input_data.append(value)

    final_input = np.array(input_data).reshape(1, -1)
    pred_encoded = model.predict(final_input)[0]

    # Decode disease name
    disease_le = encoders['Disease']
    predicted_disease = disease_le.inverse_transform([pred_encoded])[0]

    return render_template("result.html", predicted_disease=str(predicted_disease))

if __name__ == "__main__":
    app.run(debug=True)







