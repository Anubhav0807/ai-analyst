from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

correct = 0
for i in range(len(x_test)):
  prediction = model.predict([x_test[i]])
  if prediction[0] == y_test[i]:
    correct += 1

  predicted_label = iris.target_names[prediction[0]]
  actual_label = iris.target_names[y_test[i]]
  print(x_test[i], predicted_label, actual_label)

accuracy = correct / len(x_test)
print("Accuracy:", accuracy)

# print("Accuracy:", model.score(x_test, y_test))