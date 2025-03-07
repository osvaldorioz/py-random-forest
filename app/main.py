from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
from typing import List
import matplotlib
import matplotlib.pyplot as plt
from rfe_module import RandomForest
import json

matplotlib.use('Agg')  # Usar backend no interactivo
app = FastAPI()

# Definir el modelo para el vector
class VectorF(BaseModel):
    vector: List[float]
    
@app.post("/random-forest-ensemble")
def calculo(samples: int, n_trees: int):
    output_file = 'random_forest_ensemble.png'
    
    # Generar datos de ejemplo
    np.random.seed(42)
    X = np.linspace(-3, 3, samples).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

    # Entrenar el modelo
    #n_trees = 50
    rf = RandomForest(n_trees)
    rf.fit(X, y)
    y_pred = rf.predict(X)

    # Graficar resultados
    plt.figure(figsize=(10, 4))

    # Gráfica de dispersión
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, label="Datos reales", color='blue', alpha=0.6)
    plt.plot(X, y_pred, label="Predicción", color='red')
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Regresión con Random Forest")
    plt.legend()

    # Gráfica de flujo de datos (residuals)
    residuals = y - y_pred
    plt.subplot(1, 2, 2)
    plt.scatter(X, residuals, color='purple', alpha=0.6)
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel("X")
    plt.ylabel("Residuals")
    plt.title("Flujo de datos (Residuos)")

    plt.tight_layout()
    #plt.show()

    plt.savefig(output_file)
    plt.close()
    
    j1 = {
        "Grafica": output_file
    }
    jj = json.dumps(str(j1))

    return jj

@app.get("/random-forest-ensemble-graph")
def getGraph(output_file: str):
    return FileResponse(output_file, media_type="image/png", filename=output_file)
