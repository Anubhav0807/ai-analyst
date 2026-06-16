from sklearn.linear_model import Perceptron
import networkx as nx
import matplotlib.pyplot as plt

x = [[2, 3],
     [1, 1],
     [4, 5],
     [6, 7]]

y = [0, 0, 1, 1]

model = Perceptron()
model.fit(x, y)
w1, w2 = model.coef_[0]

G = nx.DiGraph()
G.add_edge("x1", "Output", weight = round(w1, 2))
G.add_edge("x2", "Output", weight = round(w2, 2))

pos = {
    "x1": (0, 1),
    "x2": (0, -1),
    "Output": (2, 0)
}

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=2000
)

labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=labels
)

plt.title("Perceptron")
plt.show()
