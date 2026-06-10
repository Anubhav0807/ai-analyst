# Problem Statement: Using the pass/fail prediction model built from student data,
# evaluate the model using a confusion matrix.

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("data1.csv")
x = df[["Study_Hours", "Attendance_Percentage", "Assignment_Score"]]
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
).plot()

plt.title("Pass/Fail Confusion Matrix")
plt.show()
