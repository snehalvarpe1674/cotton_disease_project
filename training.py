import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler

# ML Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

print("\n==============================")
print("   TRAINING STARTED")
print("==============================\n")

# 1️⃣ Load Dataset
df = pd.read_csv("cotton_data.csv")
print("Columns:", df.columns.tolist())

# 2️⃣ Encode Categorical Columns
encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# 3️⃣ Features & Target
X = df[['Crop', 'Crop Stage']]
y = df['Disease']

# 4️⃣ Balance Dataset
ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_resample(X, y)

# 5️⃣ Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# 6️⃣ Train Models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42)
}

best_model = None
best_accuracy = 0
best_model_name = ""
model_accuracy = {}

print("\n📌 Training Models...\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    model_accuracy[name] = acc
    print(f"➡ {name} Accuracy: {acc*100:.2f}%")
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

# 7️⃣ Save Best Model & Encoders
pickle.dump(best_model, open("best_model.pkl", "wb"))
pickle.dump(encoders, open("encoders.pkl", "wb"))

print("\n==============================")
print("   TRAINING COMPLETED")
print("==============================")
print(f"✔ Best Model: {best_model_name} with Accuracy: {best_accuracy*100:.2f}%")
print("🎉 Files Saved: best_model.pkl, encoders.pkl\n")











