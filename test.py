import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# 1. Preparação dos dados
X = np.array([
    [500, 80, 0.1], [1200, 80, 0.5], [64, 22, 0.02], [64000, 4444, 10.0], [45000, 8080, 15.0],
    [60000, 31337, 20.0], [800, 443, 0.3], [300, 53, 0.05], [55000, 9999, 18.0], [200, 25, 0.2],
    [700, 443, 0.2], [52000, 6666, 17.0], [150, 53, 0.03], [58000, 4444, 19.0], [900, 80, 0.4],
])

y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0])

# 2. Montagem do Pipeline
# O Pipeline garante que o StandardScaler seja aplicado aos dados de treino
# de cada fold antes de passá-los para o classificador, evitando data leakage.
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(random_state=42))
])

# 3. Avaliação com Validação Cruzada (5 folds)
# cross_val_score retorna a acurácia para cada um dos 5 folds
scores = cross_val_score(pipeline, X, y, cv=5)

# 4. Reporte de Média e Desvio Padrão
media = scores.mean()
desvio = scores.std()

print(f"Acurácia: {media:.2f} (+/- {desvio:.2f})")
