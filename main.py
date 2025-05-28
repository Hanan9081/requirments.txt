import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

#to split train and test model 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#for logistics learning and knn 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load the dataset
df = pd.read_csv("osteoporosis.csv")

# 1. Look at the first few rows
print("🔎 First 5 rows of the dataset:")
print(df.head())

# 2. Check shape of data
print(f"\n🧾 Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# 3. See all column names
print("\n📋 Column Names:")
print(df.columns.tolist())

# 4. Check for missing values
print("\n🚫 Missing values:")
print(df.isnull().sum())

# 5. Data types
print("\n🔠 Data Types:")
print(df.dtypes)

# 6. Drop 'Id' column (not useful)
df.drop('Id', axis=1, inplace=True)

# 7. Value counts for each column (basic stats)
for col in df.columns:
    print(f"\n📊 Value counts in '{col}':")
    print(df[col].value_counts())

# 8. Visualize counts for categorical columns
categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x=col)
    plt.xticks(rotation=45)
    plt.title(f"Count of {col}")
    plt.tight_layout()
    plt.show()

# 9. Correlation Heatmap (after encoding)
df_encoded = df.copy()
le = LabelEncoder()
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        df_encoded[col] = le.fit_transform(df_encoded[col])

plt.figure(figsize=(12, 8))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 10. Save preprocessed data (optional)
df_encoded.to_csv("preprocessed_data.csv", index=False)
print("\n✅ Preprocessed data saved as 'preprocessed_data.csv'")


# Load the preprocessed data
df = pd.read_csv("preprocessed_data.csv")

# Separate features (X) and target (y)
X = df.drop("Osteoporosis", axis=1)
y = df["Osteoporosis"]

# Step 1: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Choose and train a model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 3: Predict on test data
y_pred = model.predict(X_test)

# Step 4: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.2f}\n")

print("📋 Classification Report:")
print(classification_report(y_test, y_pred))

print("🔍 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Step 5: Feature Importance
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values(ascending=True).plot(kind='barh', figsize=(10,6), title="Feature Importances")
plt.tight_layout()
plt.show()


#logistics learning and knn 


# Data already split earlier:
# X_train, X_test, y_train, y_test

# 1️⃣ Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

print("🔵 Logistic Regression Results")
print(f"Accuracy: {accuracy_score(y_test, log_pred):.2f}")
print(classification_report(y_test, log_pred))

# Confusion Matrix
cm_log = confusion_matrix(y_test, log_pred)
sns.heatmap(cm_log, annot=True, fmt="d", cmap="Purples", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 2️⃣ K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)

print("🟢 K-Nearest Neighbors (KNN) Results")
print(f"Accuracy: {accuracy_score(y_test, knn_pred):.2f}")
print(classification_report(y_test, knn_pred))

# Confusion Matrix
cm_knn = confusion_matrix(y_test, knn_pred)
sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Greens", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.title("KNN - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# Simulated input (correct length: 14 values)
new_input = [[45, 1, 0, 1, 0, 2, 1, 1, 0, 0, 0, 1, 1, 0]]

# Turn it into a DataFrame with same columns as training
new_input_df = pd.DataFrame(new_input, columns=X.columns)

# Predictions
pred_rf = model.predict(new_input_df)
print("🔮 Random Forest:", "Osteoporosis" if pred_rf[0] == 1 else "No Osteoporosis")

