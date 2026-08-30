"""
Trains the CreditWise loan approval model, replicating the preprocessing
pipeline from the original notebook, and saves all artifacts needed by
the Streamlit app (model, scaler, encoders, column schema).
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv("loan_approval_data.csv")

# ---------------------------------------------------------------
# 2. Handle missing values
# ---------------------------------------------------------------
categorical_cols = df.select_dtypes(include=["object"]).columns
numerical_cols = df.select_dtypes(include=["float64"]).columns

num_imp = SimpleImputer(strategy="mean")
df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

cat_imp = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

# ---------------------------------------------------------------
# 3. Drop ID column
# ---------------------------------------------------------------
df = df.drop("Applicant_ID", axis=1)

# ---------------------------------------------------------------
# 4. Encoding
# ---------------------------------------------------------------
le_education = LabelEncoder()
df["Education_Level"] = le_education.fit_transform(df["Education_Level"])

le_target = LabelEncoder()
df["Loan_Approved"] = le_target.fit_transform(df["Loan_Approved"])  # No=0, Yes=1

ohe_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose",
            "Property_Area", "Gender", "Employer_Category"]
ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded = ohe.fit_transform(df[ohe_cols])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols), index=df.index)

df = pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)

# ---------------------------------------------------------------
# 5. Feature engineering (matches the notebook's best-performing setup)
# ---------------------------------------------------------------
df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
df["Credit_Score_sq"] = df["Credit_Score"] ** 2

X = df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio"])
y = df["Loan_Approved"]

feature_columns = list(X.columns)  # exact column order the model expects

# ---------------------------------------------------------------
# 6. Train/test split + scaling
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 7. Train Logistic Regression (best accuracy in the notebook: 0.88)
# ---------------------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# ---------------------------------------------------------------
# 8. Save all artifacts the app needs
# ---------------------------------------------------------------
joblib.dump(model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(ohe, "ohe.joblib")
joblib.dump(le_education, "le_education.joblib")
joblib.dump(le_target, "le_target.joblib")
joblib.dump(feature_columns, "feature_columns.joblib")
joblib.dump(ohe_cols, "ohe_cols.joblib")

print("\nAll artifacts saved successfully.")
