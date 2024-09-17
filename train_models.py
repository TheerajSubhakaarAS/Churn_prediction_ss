import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
from imblearn.over_sampling import SMOTE

# Define the folder where your datasets are located
dataset_folder = "./dataset"

# Define the folder to save trained models
models_folder = "./models"
os.makedirs(models_folder, exist_ok=True)

# List of datasets and their respective target variables
dataset_info = [
    {"file_path": os.path.join(dataset_folder, "BankChurners.csv"), "target_variable": "churn"},
    {"file_path": os.path.join(dataset_folder, "Bank_churn.csv"), "target_variable": "churn"},
    {"file_path": os.path.join(dataset_folder, "churn-bigml-80.csv"), "target_variable": "churn"},
    {"file_path": os.path.join(dataset_folder, "Customertravel.csv"), "target_variable": "churn"},
    {"file_path": os.path.join(dataset_folder, "E Commerce Dataset.csv"), "target_variable": "tenure"},
    {"file_path": os.path.join(dataset_folder, "ecom-user-churn-data.csv"), "target_variable": "churn"},
    {"file_path": os.path.join(dataset_folder, "internet_service_churn.csv"), "target_variable": "churn"},
    {"file_path": os.path.join(dataset_folder, "orange_telecom.csv"), "target_variable": "churn"},
    {"file_path": os.path.join(dataset_folder, "subscription_service_train.csv"), "target_variable": "churn"},
    {"file_path": os.path.join(dataset_folder, "Telco-Customer-Churn.csv"), "target_variable": "churn"},
    {"file_path": os.path.join(dataset_folder, "telecom_churn.csv"), "target_variable": "churn"}
]

def preprocess_data(df, target_variable='churn', cap_limit=0.95):
    irrelevant_cols = ['clientnum', 'customerid', 'surname', 'visitorid', 'id']
    df = df.drop([col for col in irrelevant_cols if col in df.columns], axis=1, errors='ignore')
    
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        upper_limit = df[col].quantile(cap_limit)
        df[col] = df[col].clip(upper=upper_limit)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].astype(str)
    
    X = df.drop(target_variable, axis=1)
    y = df[target_variable]
    
    return X, y

def build_pipeline(X):
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
    
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])
    
    return preprocessor

def check_and_handle_imbalance(X, y):
    # Print class distribution
    class_counts = Counter(y)
    print("Class distribution before handling imbalance:", class_counts)
    
    # Calculate imbalance ratio
    majority_class_count = max(class_counts.values())
    minority_class_count = min(class_counts.values())
    imbalance_ratio = majority_class_count / minority_class_count if minority_class_count > 0 else float('inf')
    
    print(f"Imbalance ratio: {imbalance_ratio:.2f}")
    
    # Apply SMOTE if imbalance ratio is high
    if imbalance_ratio > 1.5:  # You can adjust this threshold based on your needs
        print("Applying SMOTE to handle class imbalance...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print("Class distribution after SMOTE:", Counter(y_resampled))
        return X_resampled, y_resampled
    else:
        print("Class distribution is acceptable; skipping SMOTE.")
        return X, y

def train_and_save_models(file_path, target_variable):
    df = pd.read_csv(file_path)
    
    X, y = preprocess_data(df, target_variable)
    
    preprocessor = build_pipeline(X)
    X_transformed = preprocessor.fit_transform(X)
    
    # Handle class imbalance
    X_transformed, y = check_and_handle_imbalance(X_transformed, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=42)
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    }
    
    for model_name, model in models.items():
        print(f"Training {model_name} on {file_path}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"{model_name} Accuracy:", accuracy_score(y_test, y_pred))
        print(f"{model_name} Classification Report:\n", classification_report(y_test, y_pred))
        
        # Save the model and preprocessor
        joblib.dump(model, os.path.join(models_folder, f"{model_name}_{os.path.basename(file_path).replace('.csv', '.pkl')}"))
        joblib.dump(preprocessor, os.path.join(models_folder, f"preprocessor_{os.path.basename(file_path).replace('.csv', '.pkl')}"))

# Train and save models for all datasets
for dataset in dataset_info:
    train_and_save_models(dataset['file_path'], dataset['target_variable'])
