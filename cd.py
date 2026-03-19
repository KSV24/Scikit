import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("dataset.csv")

print(data.head())

# Features (X)
x = data[['Study Hours', 'Attendance', 'Sleep']]

# Linear Regression Target
y_reg = data['Marks']

# Logistic Target (Pass/Fail based on Marks)
y_clf = (data['Marks'] >= 60).astype(int)

# Split dataset
x_train, x_test, y_reg_train, y_reg_test = train_test_split(
    x, y_reg, test_size=0.2, random_state=42
)

_, _, y_clf_train, y_clf_test = train_test_split(
    x, y_clf, test_size=0.2, random_state=42
)

# -----------------------------
# 🔵 Linear Regression
# -----------------------------
lin_model = LinearRegression()
lin_model.fit(x_train, y_reg_train)

y_reg_pred = lin_model.predict(x_test)

print("\n🔵 Linear Regression Predictions:")
print(y_reg_pred[:5])

mse_reg = mean_squared_error(y_reg_test, y_reg_pred)
r2_reg = r2_score(y_reg_test, y_reg_pred)

print("Linear Regression MSE:", mse_reg)
print("Linear Regression R2 Score:", r2_reg)

# -----------------------------
# 🟢 Logistic Regression
# -----------------------------
log_model = LogisticRegression()
log_model.fit(x_train, y_clf_train)

y_clf_pred = log_model.predict(x_test)

print("\n🟢 Logistic Regression Predictions:")
print(y_clf_pred[:5])

mse_clf = mean_squared_error(y_clf_test, y_clf_pred)
r2_clf = r2_score(y_clf_test, y_clf_pred)

print("Logistic Regression MSE:", mse_clf)
print("Logistic Regression R2 Score:", r2_clf)