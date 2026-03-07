from flask import Flask, render_template, request
from src.predict import predict_student_performance

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        try:
            study_hours = float(request.form['study_hours'])
            attendance = float(request.form['attendance'])
            previous_marks = float(request.form['previous_marks'])
            assignments = int(request.form['assignments'])
            
            result = predict_student_performance(study_hours, attendance, previous_marks, assignments)
        except Exception as e:
            result = f"Error: {str(e)}"
            
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
