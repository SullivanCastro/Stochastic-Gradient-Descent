import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### Read data set Breast Cancer ###

def read_data_breast_cancer():
    with open(f"breast+cancer+wisconsin+diagnostic/wdbc.data") as f:
        data = f.read().splitlines()
        for idx in range(len(data)):
            data[idx] = data[idx].split(",")

    features = ["Radius", "Texture", "Perimeter", "Area", "Smoothness", "Compactness", "Concavity", "Concave Points", "Symmetry", "FractalDim"]
    colNames = []
    for type in ("_Mean", "_SE", "_Worst"):
        for feature in features:
            colNames.append(feature + type)

    data = pd.DataFrame(data, columns=["ID", "Diagnosis"] + colNames)
    data = data.drop(columns=["ID"])
    data["Diagnosis"] = data["Diagnosis"].map({"M": 1, "B": -1})

    # Center values
    X, y = data.drop(columns=["Diagnosis"]).astype(float), data["Diagnosis"]
    X -= X.mean()
    
    return X, y

### Data generation ###

def generate_data(n_points=10_000, w_barre=np.array([1.0, -1.0]), noisy=False):
    X = np.random.randn(n_points, 2)
    y = np.sign(X @ w_barre + np.random.normal(0, 0.5, (n_points)) * (noisy))
    return X, y


### Plot ###
def plot_result(X, y, w_barre=None, w_star=None):
    x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

    # Tracer les données et l'hyperplan appris
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='green', label='Classe +1')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='Classe -1')

    # Tracer l'hyperplan original
    if w_barre is not None:
        hyperplane_original = -(w_barre[0] * x_vals) / w_barre[1]
        plt.plot(x_vals, hyperplane_original, linestyle='--', color='black', label=r"Hyperplan original")

    # Tracer l'hyperplan appris par SGD
    if w_star is not None:
        hyperplane_sgd = -(w_star[0] * x_vals) / w_star[1]
        plt.plot(x_vals, hyperplane_sgd, linestyle='-', color='blue', label="Hyperplan appris")

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title("Résultats de SGD avec les classes et les hyperplans")
    plt.show()