# Predicción de demanda y stock con LSTM — Supermercados La Favorita

Práctica 2 de Aprendizaje Automático: Aprendizaje profundo y series temporales.

Una red LSTM predice la **demanda diaria** de 10 productos. El **stock al cierre** se calcula a partir de esa demanda con la regla de reposición de la tienda:

```
LSTM → demanda(t) → stock_cierre(t) = stock_inicio(t) − demanda(t)
```

## Estructura

```text
data/          inventory.csv, sales_history.csv (los genera el notebook)
notebooks/     Practica_Inventario_LSTM.ipynb y su exportación .html
models/        demanda_lstm.keras, escaladores.joblib, configuracion.json
figures/       gráficas exportadas por el notebook
tables/        métricas y predicciones exportadas
requirements.txt
```

## Dataset

365 días (01/10/2025 – 30/09/2026), 3.650 registros, 10 productos en 6 categorías. Los datos son sintéticos y se generan en la sección 1 del notebook, con semilla 42:

- Demanda con patrón semanal por categoría, feriados oficiales de Ecuador (librería `holidays`) y sus vísperas, días de pago (quincena y fin de mes), temporada navideña, tendencia leve y ruido.
- Reposición con regla (s, S): si el cierre baja del punto de reorden, al día siguiente se repone hasta el nivel máximo.
- Productos empacados en `unidad` (enteros) y productos a granel en `kg` (decimales).

## Resultados (prueba: 01/09/2026 – 30/09/2026)

| Modelo | RMSE demanda | MAE demanda |
|---|---:|---:|
| LSTM sin FE, 7 días | 6,72 | 5,03 |
| LSTM con FE, 3 días | 4,80 | 3,64 |
| **LSTM con FE, 7 días (elegido)** | **4,40** | **3,10** |
| LSTM con FE, 14 días | 6,36 | 4,33 |
| Base: persistencia (venta de ayer) | 6,47 | 4,51 |
| Base: promedio móvil 7 días | 5,64 | 4,11 |
| Base: mismo día de la semana anterior | 4,39 | 3,29 |

El modelo elegido se selecciona con el conjunto de validación (agosto de 2026).

## Funciones de predicción

```python
predecir_demanda("114674", "2026-10-01")   # 33.23  (kg -> float)
predecir_demanda_todos("2026-10-01")       # lista en el orden de CODIGOS
predecir_stock("103665", "2026-10-01")     # 63     (unidad -> int)
predecir_stock_todos("2026-10-01")
```

Las fechas deben estar entre el día siguiente al último dato y **60 días después** (hasta el 29/11/2026). Más allá de un día la predicción es recursiva y el error se acumula.

## Ejecución

1. Python 3.12 y `python -m pip install -r requirements.txt`
2. Abrir `notebooks/Practica_Inventario_LSTM.ipynb` **desde la carpeta `notebooks/`**, porque las rutas son relativas (`../data`, `../models`, ...).
3. Ejecutar todas las celdas en orden (Restart & Run All). Tarda alrededor de 1 minuto en CPU.
4. Para regenerar el HTML: `jupyter nbconvert --to html Practica_Inventario_LSTM.ipynb`
