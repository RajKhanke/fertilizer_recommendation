from flask import Flask, render_template, request
import joblib
import pandas as pd
import google.generativeai as genai


app = Flask(__name__)

# Load the trained Random Forest models
rf_ferti_name = joblib.load('rf_ferti_name.pkl')
rf_ferti_value = joblib.load('rf_ferti_value.pkl')

# Manually define the encodings based on the provided dictionaries
soil_type_encodings = {'Black': 0, 'Clayey': 1, 'Loamy': 2, 'Red': 3, 'Sandy': 4}
crop_type_encodings = {'Barley': 0, 'Cotton': 1, 'Ground Nuts': 2, 'Maize': 3, 'Millets': 4,
                       'Oil seeds': 5, 'Other Variety': 6, 'Paddy': 7, 'Pulses': 8, 'Sugarcane': 9,
                       'Tobacco': 10, 'Wheat': 11}
fertilizer_name_encodings = {'10-26-26': 0, '14-35-14': 1, '15-15-15': 2, '17-17-17': 3, '20-20': 4,
                             '20-20-20': 5, '28-28': 6, 'Ammonium sulfate': 7, 'Biofertilizer (e.g., Rhizobium)': 8,
                             'Calcium nitrate': 9, 'DAP': 10, 'Ferrous sulfate': 11, 'Magnesium sulfate': 12,
                             'Potassium chloride/Muriate of potash (MOP)': 13, 'Potassium sulfate/Sulfate of potash (SOP)': 14,
                             'Rock phosphate (RP)': 15, 'Single superphosphate (SSP)': 16, 'Triple superphosphate (TSP)': 17,
                             'Urea': 18, 'Zinc sulfate': 19}


# AI configuration
genai.configure(api_key='AIzaSyBCYG1m-yKCufYF3UaNDmj9TUpJ5DZ1UOA')
model = genai.GenerativeModel("gemini-1.5-flash")

def generate_ai_suggestions(pred_fertilizer_name):
    prompt = (
        f"For {pred_fertilizer_name} fertlizer, generate 3-4  sentences each on a new line, note text shoudl be jsutidied should not contian anyu special character"
    )
    response = model.generate_content(prompt)
    return response.text



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve form data
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        moisture = float(request.form['moisture'])
        soil_type = request.form['soil_type']
        crop_type = request.form['crop_type']
        nitrogen = float(request.form['nitrogen'])
        potassium = float(request.form['potassium'])
        phosphorous = float(request.form['phosphorous'])

        # Encode categorical data
        soil_type_encoded = soil_type_encodings.get(soil_type, -1)
        crop_type_encoded = crop_type_encodings.get(crop_type, -1)

        # Create a DataFrame for the input
        user_input = pd.DataFrame({
            'Temperature': [temperature],
            'Humidity': [humidity],
            'Moisture': [moisture],
            'Nitrogen': [nitrogen],
            'Potassium': [potassium],
            'Phosphorous': [phosphorous],
            'Soil Type': [soil_type_encoded],
            'Crop Type': [crop_type_encoded]
        })

        # Predict Fertilizer Name
        pred_fertilizer_name = rf_ferti_name.predict(user_input)[0]
        pred_fertilizer_name = [name for name, value in fertilizer_name_encodings.items() if value == pred_fertilizer_name][0]

        # Predict Fertilizer Quantity
        pred_fertilizer_qty = rf_ferti_value.predict(user_input)[0]
        pred_info = generate_ai_suggestions(pred_fertilizer_name)

        return render_template('index.html', prediction=True, fertilizer_name=pred_fertilizer_name,
                               fertilizer_qty=pred_fertilizer_qty, optimal_usage=pred_fertilizer_qty,pred_info=pred_info)
    return render_template('index.html', prediction=False)

if __name__ == '__main__':
    app.run(debug=True)
