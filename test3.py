import numpy as np
import joblib
import hashlib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 1. Preparação e Treino (Usando o vencedor do exercício anterior)
X = np.array([
    [500, 80, 0.1], [1200, 80, 0.5], [64, 22, 0.02], [64000, 4444, 10.0], [45000, 8080, 15.0],
    [60000, 31337, 20.0], [800, 443, 0.3], [300, 53, 0.05], [55000, 9999, 18.0], [200, 25, 0.2],
    [700, 443, 0.2], [52000, 6666, 17.0], [150, 53, 0.03], [58000, 4444, 19.0], [900, 80, 0.4],
])
y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0])

# Criando e treinando o modelo final
modelo_final = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(random_state=42))
])
modelo_final.fit(X, y)


# 2. Função para calcular o Hash SHA-256 de um arquivo
def calcular_hash_arquivo(nome_arquivo):
    sha256_hash = hashlib.sha256()
    with open(nome_arquivo, "rb") as f:
        # Lê o arquivo em blocos para evitar estouro de memória em arquivos grandes
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


# 3. Persistência e Hash Original
nome_arquivo = "modelo_seguranca.pkl"
joblib.dump(modelo_final, nome_arquivo)
hash_original = calcular_hash_arquivo(nome_arquivo)

print(f"Modelo salvo: {nome_arquivo}")
print(f"Hash SHA-256: {hash_original}")

# 4. Verificação de Integridade
print("\nRecarregando e verificando integridade...")
hash_atual = calcular_hash_arquivo(nome_arquivo)

if hash_atual == hash_original:
    # Recarrega o modelo apenas se a integridade for confirmada
    modelo_carregado = joblib.load(nome_arquivo)

    # Teste de previsão com um dado fictício (tráfego suspeito: alta carga)
    teste = np.array([[60000, 4444, 15.0]])
    previsao = modelo_carregado.predict(teste)

    print(f"Hash confere: modelo íntegro. Previsão de teste: {previsao}")
else:
    print("ALERTA: arquivo do modelo foi modificado!")
