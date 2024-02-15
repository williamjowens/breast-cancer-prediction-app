from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load("best_model_xgb.pkl")

# Define feature names here if not loading from a file
feature_names = [
    'clump_thickness', 'uniformity_of_cell_size', 'uniformity_of_cell_shape',
    'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei',
    'bland_chromatin', 'normal_nucleoli', 'mitoses'
]

@app.route('/')
def home():
    # Pass feature names to the template
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from form inputs
    features = [request.form.get(feature, type=float) for feature in feature_names]
    
    # Validate that all features have been provided
    if not all(features):
        return render_template('index.html', feature_names=feature_names, 
                               error="All features must be provided and within the range 0-10.")
    
    # Assuming a single row of features
    prediction = model.predict([features])
    
    # Map prediction to actual value
    output = 'Malignant' if prediction[0] == 1 else 'Benign'
    
    return render_template('index.html', feature_names=feature_names, 
                           prediction_text=f'Predicted Condition: {output}')

if __name__ == "__main__":
    app.run(debug=True)
