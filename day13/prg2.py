from sklearn.neural_network import MLPClassifier

x = [
    [2, 3],
    [1, 1],
    [4, 5],
    [6, 7]
]

y = [0, 0, 1, 1]

model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000)
model.fit(x, y)
print(model.predict([[5, 6]])[0])
