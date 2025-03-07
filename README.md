
**Random Forest** es un método de ensamble basado en árboles de decisión. Funciona creando múltiples árboles de decisión y combinando sus predicciones para mejorar la precisión y reducir el sobreajuste. En problemas de regresión, el resultado final es el promedio de las predicciones de todos los árboles.

**Implementación en este programa:**
1. Se generan datos de ejemplo usando una función seno con ruido aleatorio.
2. Se entrena un modelo de **Random Forest** con un número definido por el usuario de árboles (`n_trees = n`).
3. Se predicen los valores de `y` a partir de `X`.
4. Se generan dos gráficas:
   - **Gráfica de dispersión:** Muestra los datos reales y la predicción del modelo.
   - **Gráfica de flujo de datos (residuos):** Representa los errores de predicción en función de `X`.

Este enfoque permite visualizar cómo el modelo aprende la relación entre las variables y qué tan bien se ajusta a los datos.
