from sklearn.linear_model import Perceptron

x = [
    [2, 3],
    [1, 1],
    [4, 5],
    [6, 7]
]

y = [0, 0, 1, 1]

model = Perceptron()
model.fit(x, y)
print(model.predict([[5, 6]])[0])
