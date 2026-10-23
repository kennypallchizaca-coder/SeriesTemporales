# Predicción de demanda y stock con LSTM

Práctica 2 de Aprendizaje Automático: Aprendizaje profundo y series temporales.

Caso académico ambientado en Supermercados La Favorita, con **datos sintéticos propios**. Las ventas y el inventario se generan mediante código; no son registros oficiales de la empresa ni datos descargados de FreshRetailNet-50K.

Una red LSTM predice la **demanda diaria** de 10 productos. El **stock al cierre** se calcula a partir de esa demanda con la regla de reposición de la tienda:

```text
LSTM → demanda(t) → stock_cierre(t) = max(stock_inicio(t) − demanda(t), 0)
```

Las actividades técnicas de la guía están cubiertas: dataset sintético, análisis, preparación de secuencias, entrenamiento, ajuste, evaluación y funciones de predicción individual y por lote. La ampliación del historial a 365 días se explica más abajo.

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

Los archivos mantienen las columnas solicitadas:

```text
data/inventory.csv
  product_id, nombre, categoria, unidad, stock_actual, fecha_expiracion

data/sales_history.csv
  product_id, nombre, fecha_venta, cantidad_vendida, stock_en_momento
```

La guía propone aproximadamente 10 productos por 60 días. Se conserva el número de productos y se amplía el historial a 365 días para incluir el calendario anual; esta adaptación está justificada en la sección 1 del notebook. Los 60 días son también el horizonte máximo de predicción, un parámetro distinto de la duración del historial.

**Al ejecutar la sección 1 se sobrescriben ambos CSV.** Guardar una copia antes de sustituirlos por otra base de datos.

## Revisión de requisitos

Contraste con la guía «Aprendizaje profundo y series temporales», realizado el 23/10/2026.

| Requisito | Estado | Evidencia en el notebook |
|---|---|---|
| Dataset sintético y dos CSV con las columnas indicadas | Cumple con adaptación del historial | Secciones 1 y 1.1: 10 productos, 3.650 ventas y validaciones de integridad; 365 días frente a los ~60 propuestos. |
| Carga, exploración y preparación de datos | Cumple | Secciones 1.1 y 2A: tipos, nulos, duplicados, continuidad diaria, coherencia de stock, distribuciones y patrones temporales. |
| Feature Engineering temporal | Cumple | Sección 2B: día de semana, mes, semana del año, fin de semana, feriados, vísperas, días de pago y Navidad; también rezagos y media móvil. |
| Secuencias por producto o categoría | Cumple | Sección 2C: ventanas por producto, tensores 3D, partición por fechas y escaladores ajustados con entrenamiento durante la evaluación. |
| Diseño y entrenamiento de una RNN | Cumple | Sección 3: TensorFlow/Keras, LSTM(50) y Dense(1), pérdida MAE, Adam y EarlyStopping. |
| Evaluación con RMSE | Cumple | Sección 3.1: RMSE y MAE en escala original, comparación con líneas base y gráficas de predicción frente al valor real. |
| Fine-tuning | Cumple con ajuste básico | Sección 3: comparación con y sin FE y ventanas de 3, 7 y 14 días; selección por RMSE de validación y parada temprana. |
| Función que reciba product_id y fecha | Cumple | Sección 4: `predecir_demanda(product_id, fecha)`, con validación de producto y fecha. |
| Función que reciba un batch de productos y una fecha | Cumple | Sección 4: `predecir_demanda_lote(product_ids, fecha)` recibe una lista elegida y devuelve sus demandas en el mismo orden; tiene pruebas de coherencia y entradas inválidas. |
| Código documentado y entrega en ipynb y HTML | Archivos preparados | Hay explicaciones, comentarios, docstrings y ambos formatos con el mismo código. La subida al AVAC no se verificó. |

La función por lote reutiliza la misma preparación y el mismo modelo que las consultas individuales. El notebook incluye las explicaciones de cada fase y su HTML se regeneró después de añadir la función y ejecutar sus pruebas.

## Entrenamiento y evaluación

- Entrenamiento: 01/10/2025–01/08/2026, descontando las filas iniciales necesarias para rezagos y ventanas.
- Validación: 02/08/2026–31/08/2026.
- Prueba: 01/09/2026–30/09/2026.

Cada secuencia contiene un solo producto. La evaluación mantiene el orden temporal y usa información histórica anterior al día objetivo. El calendario del día a predecir se conoce de antemano.

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

El modelo elegido tiene el menor RMSE de validación (4,76). Supera a la persistencia y al promedio móvil en prueba. La línea base semanal obtiene un RMSE ligeramente menor (4,39 frente a 4,40); la LSTM obtiene menor MAE (3,10 frente a 3,29).

Estas métricas corresponden a predicciones de un día con historial observado. No son una evaluación del pronóstico recursivo a 60 días. Además, el promedio conjunto mezcla productos medidos en unidades y kg, por lo que no representa un error porcentual ni monetario.

Después de evaluar las configuraciones, se reentrena el modelo final con todo el historial durante 19 épocas y se guardan el modelo, los escaladores y la configuración. Las métricas de prueba anteriores corresponden al experimento separado, no a una nueva prueba del modelo final reentrenado.

## Funciones de predicción

```python
# Ejecutar primero las celdas anteriores del notebook.
predecir_demanda("114674", "2026-10-01")   # 33.23  (kg -> float)
predecir_demanda_lote(["114674", "103665"], "2026-10-01")  # [33.23, 39]
predecir_demanda_todos("2026-10-01")       # lista en el orden de CODIGOS
predecir_stock("103665", "2026-10-01")     # 63     (unidad -> int)
predecir_stock_todos("2026-10-01")
```

Las fechas deben estar entre el día siguiente al último dato y **60 días después** (hasta el 29/11/2026). Más allá de un día la predicción es recursiva y el error se acumula.

`predecir_demanda_lote` recibe una lista de códigos de texto, conserva su orden y sus repeticiones, y devuelve `[]` para una lista vacía. Rechaza códigos inexistentes, elementos que no sean códigos de texto y entradas que no sean listas. Las consultas de todo el catálogo mantienen el orden de `CODIGOS`.

Las funciones viven en el notebook y dependen de su estado en memoria; todavía no hay un módulo o servicio independiente de despliegue. Los artefactos guardados permiten recuperar el modelo, pero también se necesitan el historial y la preparación de variables.

## Comprobaciones de la revisión

- CSV: columnas correctas, 10 productos con 365 días cada uno, sin nulos ni duplicados, sin saltos de fechas y ventas entre cero y el stock disponible.
- Notebook: las 17 celdas de código tienen ejecución guardada y no contienen salidas de error. El código y la tabla de métricas del HTML coinciden con el notebook y el CSV de métricas.
- Artefactos: se cargaron el modelo y los escaladores guardados. Las funciones individuales y del catálogo completo coincidieron para 01/10/2026, 30/10/2026 y 29/11/2026, con resultados no negativos y los tipos numéricos esperados.
- Entradas inválidas: se rechazaron un producto inexistente, una fecha imposible, un formato incorrecto y fechas fuera del intervalo admitido.
- Lotes: se comprobó un subconjunto en distinto orden y con productos repetidos para el primer día, el día 30 y el día 60. Los resultados coinciden con las consultas individuales; la lista vacía devuelve `[]`. Se rechazaron siete lotes inválidos y cuatro fechas inválidas.
- Predicciones exportadas: las 124 filas de `tables/predicciones_personalizadas.csv` coincidieron con las consultas al modelo guardado para demanda y stock, con tolerancia de 0,011 por redondeo.

Se reejecutó la celda de pruebas de predicción con el modelo guardado y se conservaron sus resultados en el notebook. No se reejecutó el entrenamiento completo ni se regeneraron los datos. Los patrones del EDA describen el escenario sintético definido por el generador; no constituyen evidencia sobre ventas reales de la empresa.

## Ejecución

1. Python 3.12 y `python -m pip install -r requirements.txt`
2. Abrir `notebooks/Practica_Inventario_LSTM.ipynb` **desde la carpeta `notebooks/`**, porque las rutas son relativas (`../data`, `../models`, ...).
3. Ejecutar todas las celdas en orden (Restart & Run All). Esto regenera los datos, reentrena los modelos y sobrescribe los artefactos de salida. El tiempo depende del equipo.
4. Para regenerar el HTML: `jupyter nbconvert --to html Practica_Inventario_LSTM.ipynb`

La guía recomienda Google Colab con GPU; no lo establece como requisito obligatorio. La ejecución guardada se realizó en CPU.

Para entregar en AVAC, subir `notebooks/Practica_Inventario_LSTM.ipynb` y `notebooks/Practica_Inventario_LSTM.html`, verificando que el HTML se exporte después del último cambio del notebook.
