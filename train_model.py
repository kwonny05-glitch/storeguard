import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
import joblib

df = pd.read_csv("output/storeguard_synthetic_data.csv")

X = df.drop("closed_within_24m", axis=1)
y = df["closed_within_24m"]

num_cols = [
    "daily_foot_traffic","competitors","vacancy_rate","trend_index",
    "rent_monthly","expected_sales","equity_ratio",
    "experience_level","franchise_scale","rent_to_sales_ratio"
]
cat_cols = ["region","category"]

preprocess = ColumnTransformer([
    ("num","passthrough",num_cols),
    ("cat",OneHotEncoder(handle_unknown="ignore"),cat_cols)
])

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

clf = Pipeline([
    ("prep", preprocess),
    ("model", rf)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

clf.fit(X_train, y_train)

proba = clf.predict_proba(X_test)[:,1]
pred = (proba >= 0.5).astype(int)
auc = roc_auc_score(y_test, proba)
f1 = f1_score(y_test, pred)

print(f"AUC: {auc:.3f}, F1: {f1:.3f}")

Path("output").mkdir(exist_ok=True)
joblib.dump(clf, "output/storeguard_rf_model.joblib")
print("Saved model to output/storeguard_rf_model.joblib")