# Student Performance Prediction System

A complete Machine Learning web application to predict student performance (Pass/Fail) based on study hours, attendance, previous marks, and assignments.

## Features
- **Data Generation:** Automatically generates a realistic dataset of student performance.
- **Preprocessing:** Handles missing values and scales features appropriately.
- **EDA:** Generates scatter plots inside the `data/` directory to visualize relationships.
- **Model Training:** Trains and evaluates Logistic Regression, Decision Tree, Random Forest, and SVM models.
- **Web App:** A modern, responsive Flask web application to make real-time predictions using the best trained model.

## Prerequisites
- Python 3.8+
- pip

## Setup & Running the Project

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model:**
   This step will generate the dataset, perform EDA visualizations, train the models, and output `student_model.pkl` in the `models/` folder.
   ```bash
   python src/train_model.py
   ```

3. **Run the Application:**
   ```bash
   python app.py
   ```

4. **Access the App:**
   Open your browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Folder Structure
- `data/` : Contains the generated dataset and exploratory data analysis (EDA) graphs.
- `models/` : Contains the saved best machine learning model and the feature scaler.
- `src/` : Contains Python scripts for training and prediction logic.
- `templates/` : Contains the HTML UI for the Flask application.
