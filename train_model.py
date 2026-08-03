import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# 1. Load dataset
print("Loading dataset...")
df = pd.read_csv("credit_card_fraud_10k.csv")

# 2. Define features and target
features = [
    "amount",
    "transaction_hour",
    "merchant_category",
    "foreign_transaction",
    "location_mismatch",
    "device_trust_score",
    "velocity_last_24h",
    "cardholder_age",
]
target = "is_fraud"

X = df[features]
y = df[target]

# 3. Create preprocessing pipeline
categorical_features = ["merchant_category"]
numerical_features = [col for col in features if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough",
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(n_estimators=100, random_state=42),
        ),
    ]
)

# 4. Train model
print("Training Random Forest model...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
pipeline.fit(X_train, y_train)

# 5. Save model artifact
joblib.dump(pipeline, "fraud_model.joblib")
print("✅ Success: 'fraud_model.joblib' generated successfully!")