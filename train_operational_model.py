import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump, load

def train_and_save_model(data_path, model_path, preprocessor_path):
    """
    Trains a predictive model for length of stay and saves it.
    """
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: The data file '{data_path}' was not found.")
        return

    # Define features (X) and target (y)
    features = ['department', 'staff_on_duty', 'patients_treated']
    target = 'length_of_stay_days'

    X = df[features]
    y = df[target]

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define preprocessing steps
    # For 'department', we need one-hot encoding
    categorical_features = ['department']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Create a preprocessor with ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # Create the model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Train the model
    model_pipeline.fit(X_train, y_train)

    # Get the feature names after one-hot encoding
    feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
    feature_names.extend(['staff_on_duty', 'patients_treated'])

    # Save the trained model and feature names
    dump(model_pipeline, model_path)
    dump(feature_names, preprocessor_path)

    print(f"Model saved successfully to {model_path}")
    print(f"Feature names saved successfully to {preprocessor_path}")

if __name__ == '__main__':
    train_and_save_model('operational_data.csv', 'operational_model.joblib', 'feature_names.joblib')
