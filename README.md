# Predicción de demanda y stock con LSTM

Práctica 1 de Aprendizaje Automático: aprendizaje profundo y series temporales.

El [notebook](notebooks/Practica_Inventario_LSTM.ipynb) contiene la generación y validación de datos, el análisis exploratorio, la preparación de secuencias, el entrenamiento, las métricas y las funciones de predicción. Conserva sus gráficas, tablas y resultados guardados.

## Archivos necesarios

```text
data/
  inventory.csv
  sales_history.csv
notebooks/
  Practica_Inventario_LSTM.ipynb
README.md
requirements.txt
```

`.gitignore` excluye el entorno de Python y los archivos temporales. Las carpetas `.git/` y `.venv/` se conservan para mantener el historial y el entorno de trabajo local.

## Datos

Los CSV contienen datos sintéticos de diez productos durante 365 días, del 01/10/2025 al 30/09/2026: diez registros de inventario y 3.650 registros de ventas. El caso está ambientado en La Favorita; no utiliza registros oficiales de la empresa.

- `inventory.csv`: producto, nombre, categoría, unidad comercial, stock y expiración.
- `sales_history.csv`: producto, nombre, fecha, cantidad vendida y stock disponible.

## Ejecución

1. Usar Python 3.12 e instalar las dependencias con `python -m pip install -r requirements.txt`, o utilizar el entorno `.venv` existente.
2. Abrir el notebook desde la carpeta `notebooks/`, porque sus rutas son relativas a ella.
3. Leer las salidas guardadas o ejecutar las celdas en orden para reproducir la práctica.

Ejecutar todo regenera los CSV, reentrena los modelos y crea las carpetas `models/`, `figures/` y `tables/`. Estos archivos externos son resultados generados y se han retirado de la estructura mínima. Para hacer nuevas predicciones después de reiniciar el kernel, hay que ejecutar las celdas anteriores, incluido el entrenamiento.

## Modelo y resultados guardados

La red elegida usa una ventana de siete días y 28 variables: `Input((7,28)) -> LSTM(50) -> Dense(1)`. Se entrena con MAE y Adam; la selección entre configuraciones usa el RMSE de validación.

| Modelo | RMSE de prueba | MAE de prueba |
|---|---:|---:|
| LSTM con variables temporales, siete días | 4,400 | 3,103 |
| Base del mismo día de la semana anterior | 4,391 | 3,294 |

La base semanal obtiene un RMSE ligeramente menor; la LSTM obtiene menor MAE. La prueba cubre septiembre de 2026 con predicciones de un día e historial observado. Estas métricas mezclan unidades y kg y no evalúan el pronóstico recursivo de hasta 60 días ni constituyen una prueba independiente del modelo final reentrenado.

La red estima la demanda; el stock al cierre se calcula después mediante la política de reposición. El notebook admite fechas hasta el 29/11/2026. La entrega permanece en local.
