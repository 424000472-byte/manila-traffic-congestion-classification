import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. LOAD RAW DATASET
df = pd.read_csv("manila_traffic_raw.csv")
print(f"1. Initial Raw Rows: {len(df)}")

# 2. DUPLICATE HANDLING
df = df.drop_duplicates(subset=["Timestamp", "Road_Name"]).copy()
print(f"2. After Duplicate Handling: {len(df)}")

# 3. REDUNDANCY REMOVAL
df = df[["Road_Name", "Vehicle_Volume", "Travel_Time_Sec", "Congestion"]].copy()

# 4. NOISE DATA FILTERING
df.loc[df["Vehicle_Volume"] <= 0, "Vehicle_Volume"] = np.nan
df.loc[df["Travel_Time_Sec"] <= 0, "Travel_Time_Sec"] = np.nan

# 5. MISSING DATA HANDLING
df = df.dropna(subset=["Congestion"]).copy()
df["Congestion"] = df["Congestion"].astype(int)

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")
df[["Vehicle_Volume", "Travel_Time_Sec"]] = num_imputer.fit_transform(
    df[["Vehicle_Volume", "Travel_Time_Sec"]]
)
df[["Road_Name"]] = cat_imputer.fit_transform(df[["Road_Name"]])
print(f"3. After Noise & Missing Data Handling: {len(df)}")

# 6. OUTLIER DETECTION (IQR Method)
for col in ["Vehicle_Volume", "Travel_Time_Sec"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

print(f"4. After IQR Outlier Removal: {len(df)}")
print("Class distribution:\n", df["Congestion"].value_counts())

if df["Congestion"].nunique() < 2:
    raise ValueError("Dataset needs both 0 (Clear) and 1 (Congested) rows before training. Collect more off-peak data first.")

X = df[["Road_Name", "Vehicle_Volume", "Travel_Time_Sec"]]
y = df["Congestion"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. CATEGORICAL ENCODING & 8. SCALING
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Road_Name"]),
        ("num", StandardScaler(), ["Vehicle_Volume", "Travel_Time_Sec"]),
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 9. CLASSIFICATION MODEL DEVELOPMENT
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_processed, y_train)

# 10. MODEL EVALUATION & RESULTS
y_pred = model.predict(X_test_processed)
print("\n=== MODEL EVALUATION RESULTS ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["No (0 - Clear)", "Yes (1 - Congested)"]))

# 11. GRAPHS
ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, y_pred),
    display_labels=["No (0)", "Yes (1)"],
).plot(cmap="Blues")
plt.title("Manila Multi-Road Traffic Congestion - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

RocCurveDisplay.from_estimator(model, X_test_processed, y_test)
plt.title("ROC Curve - Traffic Congestion Classifier")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300)
plt.close()

joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(model, "congestion_model.pkl")
print("Saved: preprocessor.pkl, congestion_model.pkl, confusion_matrix.png, roc_curve.png")