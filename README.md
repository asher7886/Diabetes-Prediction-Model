# Smart Healthcare Prediction System for Diabetes Risk Assessment

## 1. Project Overview

This project develops a **Smart Healthcare Prediction System for Diabetes Risk Assessment** using clinical and lifestyle data. The system utilizes ensemble machine learning techniques to classify patients into three categories: **Non-Diabetic**, **Pre-Diabetic**, and **Diabetic**. This early classification aims to enable timely intervention and support healthcare professionals in decision-making, ultimately improving patient outcomes.

The solution encompasses data generation, preprocessing, training of multiple ensemble machine learning models, evaluation, and deployment of a user-friendly web-based interface for real-time predictions.

## 2. Dataset

Due to the unavailability of the original UCI Diabetes 130-US Hospitals dataset during the development phase, a synthetic dataset was generated to simulate clinical and lifestyle data relevant to diabetes prediction. This dataset includes features such as age, gender, BMI, blood pressure, glucose level, hemoglobin A1C, cholesterol, physical activity, smoking status, and family history. The target variable, `status`, is categorized into 'Non-Diabetic', 'Pre-Diabetic', and 'Diabetic' based on established clinical thresholds for glucose and HbA1c levels, with some added noise for realism.

**Key characteristics of the simulated dataset:**

*   **Number of Samples:** 5000
*   **Features:** 10 (Age, Gender, BMI, Blood Pressure, Glucose Level, Hemoglobin A1C, Cholesterol, Physical Activity, Smoking Status, Family History)
*   **Target Classes:** Non-Diabetic, Pre-Diabetic, Diabetic

### Dataset Statistics

```text
Dataset Statistics
==================
Total instances: 5000
Total features: 10

Target (status) Distribution:
status
Diabetic        4226
Pre-Diabetic     468
Non-Diabetic     306
Name: count, dtype: int64
```

### Exploratory Data Analysis (EDA) Visualizations

Visualizations generated during the EDA phase provide insights into the distribution of the target variable and key demographic features. These plots are saved in the `plots/` directory.

*   `plots/target_dist.png`: Distribution of Diabetes Status (Non-Diabetic, Pre-Diabetic, Diabetic).
*   `plots/age_dist.png`: Distribution of Age groups in the dataset.

## 3. Data Preprocessing

The raw data undergoes several preprocessing steps to prepare it for machine learning models:

1.  **Handling Missing Values:** Missing values (simulated for BMI) are imputed using the median of the respective feature.
2.  **Encoding Categorical Features:** Categorical features such as `gender`, `smoking_status`, and `family_history` are converted into numerical representations using `LabelEncoder`.
3.  **Target Encoding:** The `status` (target) variable is encoded into numerical labels (0: Non-Diabetic, 1: Pre-Diabetic, 2: Diabetic).
4.  **Data Splitting:** The dataset is split into training and testing sets (80% training, 20% testing).
5.  **Feature Scaling:** Numerical features are scaled using `StandardScaler` to ensure that all features contribute equally to the model training process.

## 4. Model Development

Four ensemble machine learning algorithms were implemented and trained to build the predictive model:

*   **Random Forest Classifier:** An ensemble learning method that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.
*   **Gradient Boosting Classifier:** A powerful technique that builds models in a stage-wise fashion; it builds an ensemble of weak prediction models, typically decision trees.
*   **AdaBoost Classifier:** (Adaptive Boosting) is a meta-algorithm that can be used in conjunction with many other types of learning algorithms to improve performance. It works by weighting misclassified samples more heavily in subsequent iterations.
*   **Voting Classifier:** A meta-classifier that trains several base models and predicts based on the majority vote of their predictions (for classification) or average (for regression). This project uses a 'soft' voting strategy, which predicts the class based on the argmax of the sums of the predicted probabilities.

All models are trained on the preprocessed training data.

## 5. Model Evaluation

Each model's performance was evaluated using standard classification metrics on the test set:

*   **Accuracy:** The proportion of correctly classified instances.
*   **Precision:** The ratio of correctly predicted positive observations to the total predicted positive observations.
*   **Recall (Sensitivity):** The ratio of correctly predicted positive observations to all observations in the actual class.
*   **F1-Score:** The weighted average of Precision and Recall.
*   **Confusion Matrix:** A table used to describe the performance of a classification model on a set of test data for which the true values are known.

### Model Comparison

The following table summarizes the performance of each ensemble model:

```
               Model  Accuracy  Precision  Recall  F1-Score
0      Random Forest     0.927   0.926325   0.927  0.920297
1  Gradient Boosting     0.924   0.922484   0.924  0.916678
2           AdaBoost     0.893   0.890189   0.893  0.876607
3  Voting Classifier     0.926   0.925080   0.926  0.919345
```

The **Random Forest** and **Voting Classifier** models demonstrated the highest accuracy and F1-scores, indicating robust performance in classifying diabetes risk. The **Voting Classifier** was chosen for the web application due to its ability to leverage the strengths of multiple models.

### Evaluation Visualizations

Confusion matrices for each model are saved in the `results/` directory:

*   `results/cm_Random_Forest.png`
*   `results/cm_Gradient_Boosting.png`
*   `results/cm_AdaBoost.png`
*   `results/cm_Voting_Classifier.png`

Additionally, feature importance for the Random Forest model is visualized in `results/feature_importance.png`.

## 6. Web Application

A Flask-based web application provides a user-friendly interface for real-time diabetes risk assessment. Users can input various clinical and lifestyle parameters, and the system will predict their diabetes status (Non-Diabetic, Pre-Diabetic, or Diabetic) along with a basic health recommendation.

**Key features of the web application:**

*   **Input Form:** Collects patient data through an intuitive web form.
*   **Real-time Prediction:** Utilizes the trained ensemble model to provide immediate risk assessment.
*   **Health Recommendations:** Offers basic, status-specific health advice.

## 7. How to Use the Solution

To set up and run the Smart Healthcare Prediction System, follow these steps:

### Prerequisites

Ensure you have Python 3.x and `pip` installed on your system.

### Step 1: Clone the Repository (or create files manually)

If you have access to the project files, ensure they are in a directory named `diabetes_prediction`. If not, create the files as described in the development steps.

### Step 2: Install Dependencies

Navigate to the project directory and install the required Python packages:

```bash
cd /home/ubuntu/diabetes_prediction
sudo pip3 install pandas scikit-learn matplotlib seaborn numpy flask joblib
```

### Step 3: Prepare Data and Train Models

Run the data preparation script to generate the synthetic dataset and preprocess it. Then, run the model training script to train the ensemble models and save them.

```bash
cd /home/ubuntu/diabetes_prediction
python3 prepare_data.py
python3 train_models.py
```

These scripts will create:

*   `diabetes_clinical_data.csv`: The generated raw dataset.
*   `processed_data.joblib`: A joblib file containing scaled training/testing data, scaler, and feature names.
*   `results/`: A directory containing trained models (`model_*.joblib`), confusion matrices (`cm_*.png`), feature importance plot (`feature_importance.png`), and a model comparison CSV (`model_comparison.csv`).

### Step 4: Run the Web Application

Start the Flask web server:

```bash
cd /home/ubuntu/diabetes_prediction
python3 app.py
```

The application will be accessible at `http://127.0.0.1:5000` (or `http://0.0.0.0:5000` if exposed). You can then open this URL in your web browser to interact with the system.

### Step 5: Interact with the Web Application

Open your web browser and go to the address where the Flask app is running. You will see a form where you can input patient parameters. Fill in the details and click "Predict Risk" to get the diabetes risk assessment and a health recommendation.

## Disclaimer

This system is developed for demonstration and educational purposes only. It should not be used for actual medical diagnosis or treatment decisions. Always consult with a qualified healthcare professional for any health concerns.
