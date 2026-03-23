import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
data = pd.read_csv("dataset.csv")
print(data.head())
x = data[['Study Hours', 'Attendance', 'Sleep']]
y_reg = data['Marks']
y_clf = (data['Marks'] >= 60).astype(int)
imputer = SimpleImputer(strategy='mean')
x = imputer.fit_transform(x)
scaler = StandardScaler()
x = scaler.fit_transform(x)
x_train, x_test, y_reg_train, y_reg_test = train_test_split(
    x, y_reg, test_size=0.2, random_state=42
)

_, _, y_clf_train, y_clf_test = train_test_split(
    x, y_clf, test_size=0.2, random_state=42
)
lin_model = LinearRegression()
lin_model.fit(x_train, y_reg_train)
y_reg_pred = lin_model.predict(x_test)
print("\n🔵 Linear Regression")
print("Predictions:", y_reg_pred[:5])
print("MSE:", mean_squared_error(y_reg_test, y_reg_pred))
print("R2 Score:", r2_score(y_reg_test, y_reg_pred))
log_model = LogisticRegression(max_iter=1000)
log_model.fit(x_train, y_clf_train)
y_clf_pred = log_model.predict(x_test)
print("\n🟢 Logistic Regression")
print("Predictions:", y_clf_pred[:5])
print("Confusion Matrix:\n", confusion_matrix(y_clf_test, y_clf_pred))
print("Classification Report:\n", classification_report(y_clf_test, y_clf_pred))
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_clf_train)
y_knn_pred = knn.predict(x_test)
print("\n🟣 KNN Classifier")
print("Predictions:", y_knn_pred[:5])
print("Confusion Matrix:\n", confusion_matrix(y_clf_test, y_knn_pred))
print("Classification Report:\n", classification_report(y_clf_test, y_knn_pred))
cv_scores = cross_val_score(knn, x, y_clf, cv=5)
print("\n🟡 Cross Validation Scores:", cv_scores)
print("Average CV Score:", cv_scores.mean())