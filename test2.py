import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

# 1. Preparação dos dados
X = np.array([
    [500, 80, 0.1], [1200, 80, 0.5], [64, 22, 0.02], [64000, 4444, 10.0], [45000, 8080, 15.0],
    [60000, 31337, 20.0], [800, 443, 0.3], [300, 53, 0.05], [55000, 9999, 18.0], [200, 25, 0.2],
    [700, 443, 0.2], [52000, 6666, 17.0], [150, 53, 0.03], [58000, 4444, 19.0], [900, 80, 0.4],
])
y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0])

# 2. Torneio de Algoritmos
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

results = {}

for name, model in models.items():
    # Criamos um pipeline para cada modelo para garantir a normalização correta
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", model)
    ])
    # Calculamos a média da acurácia com 5 folds
    score = cross_val_score(pipeline, X, y, cv=5).mean()
    results[name] = score
    print(f"{name}: {score:.2f}")

# 3. Declaração do Vencedor
winner_name = max(results, key=results.get)
print(f"Vencedor: {winner_name}")

# 4. Otimização do Vencedor com GridSearch
# Definimos grades de parâmetros diferentes dependendo de quem venceu
param_grids = {
    "RandomForest": {
        "clf__n_estimators": [10, 50, 100],
        "clf__max_depth": [None, 5, 10]
    },
    "SVM": {
        "clf__C": [0.1, 1, 10],
        "clf__kernel": ["linear", "rbf"]
    },
    "KNN": {
        "clf__n_neighbors": [3, 5, 7]
    }
}

# Montamos o pipeline do vencedor para o GridSearch
best_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", models[winner_name])
])

grid = GridSearchCV(best_pipeline, param_grids[winner_name], cv=5)
grid.fit(X, y)

print(f"Melhores parâmetros: {grid.best_params_}")
