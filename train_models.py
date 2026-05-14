import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directory for results
os.makedirs('results', exist_ok=True)

# Load data
data = joblib.load('processed_data.joblib')
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
status_map = data['status_map']
inv_status_map = {v: k for k, v in status_map.items()}

# Define models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
}

# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[(name, clf) for name, clf in models.items()],
    voting='soft'
)
models['Voting Classifier'] = voting_clf

# Evaluate models
results = []

for name, clf in models.items():
    print(f"Training {name}...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    })
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=status_map.keys(), 
                yticklabels=status_map.keys())
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'results/cm_{name.replace(" ", "_")}.png')
    plt.close()
    
    # Save the model
    joblib.dump(clf, f'results/model_{name.replace(" ", "_")}.joblib')

# Save comparison table
comparison_df = pd.DataFrame(results)
comparison_df.to_csv('results/model_comparison.csv', index=False)

print("\nModel Comparison:")
print(comparison_df)

# Feature Importance (for Random Forest)
rf_model = models['Random Forest']
importances = rf_model.feature_importances_
feature_names = data['features']
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance (Random Forest)')
plt.savefig('results/feature_importance.png')

print("\nModel training and evaluation completed. Results saved in 'results/' directory.")
