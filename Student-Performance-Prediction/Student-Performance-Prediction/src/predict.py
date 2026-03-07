import joblib
import os

def predict_student_performance(study_hours, attendance, previous_marks, assignments):
    model_path = 'models/student_model.pkl'
    scaler_path = 'models/scaler.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return "Model not trained yet!"
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    input_data = [[study_hours, attendance, previous_marks, assignments]]
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)
    return "Pass" if prediction[0] == 1 else "Fail"
