# Predicción de demanda y stock con LSTM

Práctica 1 de Aprendizaje Automático: aprendizaje profundo y series temporales.

Evaluación de conocimientos: **lunes 9 de noviembre de 2026**. Este README reúne la práctica y el material de repaso para consultarlos desde GitHub.

- [Archivos necesarios](#archivos-necesarios)
- [Ejecución](#ejecución)
- [Modelo y resultados](#modelo-y-resultados-guardados)
- [Formulario de estudio](#formulario-de-estudio)
- [Preparación para la evaluación](#preparación-para-la-evaluación)
- [Práctica de live coding](#práctica-de-live-coding)
- [MLflow y Airflow](#mlflow-y-airflow)

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

`.gitignore` excluye el entorno de Python, los archivos temporales y la carpeta local `estudio/`. El material de estudio publicado se encuentra en este README; no necesita archivos de esa carpeta. Las carpetas `.git/` y `.venv/` se conservan para mantener el historial y el entorno de trabajo local.

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

La red estima la demanda; el stock al cierre se calcula después mediante la política de reposición. El notebook admite fechas hasta el 29/11/2026. El notebook, los CSV y esta documentación forman la entrega versionada.

## Formulario de estudio

Comandos, fórmulas y conceptos para repasar. El docente determina si permite consultarlos durante la evaluación.

### Pandas

Ejecutar desde la raíz del repositorio, en una sesión Python o un notebook de prueba. Los identificadores se conservan como texto.

```python
import numpy as np
import pandas as pd

v = pd.read_csv('data/sales_history.csv',
                dtype={'product_id': str}, parse_dates=['fecha_venta'])
i = pd.read_csv('data/inventory.csv', dtype={'product_id': str})
v = v.sort_values(['product_id', 'fecha_venta']).reset_index(drop=True)
```

| Necesidad | Comando |
|---|---|
| Inspeccionar | `v.head(); v.shape; v.info(); v.describe()` |
| Nulos por columna | `v.isna().sum()` |
| Duplicados de la clave | `v.duplicated(['product_id', 'fecha_venta']).sum()` |
| Seleccionar columnas | `v[['product_id', 'cantidad_vendida']]` |
| Filtrar por etiquetas | `v.loc[v['product_id'].eq('103665'), ['fecha_venta', 'cantidad_vendida']]` |
| Seleccionar por posición | `v.iloc[:5, :3]` |
| Combinar filtros | `v.loc[(v['cantidad_vendida'] > 30) & v['product_id'].eq('103665')]` |
| Ordenar | `v.sort_values(['product_id', 'fecha_venta'])` |
| Unir categoría | `v.merge(i[['product_id', 'categoria']], on='product_id', validate='many_to_one')` |
| Sumar por producto | `v.groupby('product_id')['cantidad_vendida'].sum()` |
| Varias estadísticas | `v.groupby('product_id')['cantidad_vendida'].agg(['mean', 'std', 'max'])` |
| Media alineada con cada fila | `v.groupby('product_id')['cantidad_vendida'].transform('mean')` |
| Calendario | `v['fecha_venta'].dt.dayofweek` (lunes = 0), `.dt.month` |
| Tabla por fecha y producto | `v.pivot(index='fecha_venta', columns='product_id', values='cantidad_vendida')` |
| Exportar sin índice | `v.to_csv('resultado.csv', index=False)` |

`agg` resume los grupos; `transform` conserva una salida por fila. `merge` combina por claves; `concat` apila filas o columnas. `pivot` necesita claves únicas; `pivot_table` permite agregar. No sumar kg y unidades como una misma magnitud.

### Series temporales y escalado

Ordenar por producto y fecha antes de crear rezagos. En este ejemplo cada fila representa el día objetivo `t`:

```python
g = v.groupby('product_id')['cantidad_vendida']
v['lag_1'] = g.shift(1)
v['lag_7'] = g.shift(7)
v['media_7'] = g.transform(lambda s: s.shift(1).rolling(7).mean())
```

`shift(1)` excluye la venta objetivo del promedio. Eliminar los nulos iniciales después de crear las variables, sin rellenarlos con información futura. Si faltan días, siete filas no equivalen necesariamente a siete días.

| Fórmula o regla | Interpretación |
|---|---|
| Media: `sum(x)/n` | Nivel promedio |
| Desviación muestral: `sqrt(sum((x-media)**2)/(n-1))` | Pandas `std()` usa `ddof=1` por defecto |
| MinMax: `(x-min_train)/(max_train-min_train)` | Ajustar con train y transformar val/test |
| Ventana: `[t-7,...,t-1] -> y[t]` | Siete días históricos predicen un día |
| Entrada LSTM: `(muestras, pasos, variables)` | En la práctica: `(2920,7,28)` para train |
| Persistencia: `y_pred[t] = y[t-1]` | Repite la última venta observada |
| Base semanal: `y_pred[t] = y[t-7]` | Repite el mismo día de la semana anterior |
| Stock: `max(stock_inicio-demanda_pred, 0)` | Cálculo posterior a la red |

Train: hasta 01/08/2026. Validación: 02/08 a 31/08. Prueba: 01/09 a 30/09. Elegir hiperparámetros con validación. Los datos nuevos pueden quedar fuera de `[0,1]` después de MinMax.

En el notebook principal cada fila de entrada describe el día `d` y el calendario de `d+1`: allí `shift(6)` representa la venta del mismo día semanal que ese objetivo. Cuando la fila representa directamente `t`, se usa `shift(7)`.

### Métricas

Con `e = y_real-y_pred` y `y_media` igual a la media real:

| Métrica | Fórmula | Lectura |
|---|---|---|
| MAE | `mean(abs(e))` | Error absoluto medio, en la unidad del objetivo |
| MSE | `mean(e**2)` | Penaliza errores grandes; unidad al cuadrado |
| RMSE | `sqrt(mean(e**2))` | Penaliza errores grandes; unidad del objetivo |
| R² | `1-sum(e**2)/sum((y_real-y_media)**2)` | Puede ser negativo; objetivo constante es un caso especial |
| MAPE (%) | `100*mean(abs(e/y_real))` | Indefinido cuando un valor real es cero |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
y_real = [10, 20, 30]
y_pred = [12, 18, 33]
mae = mean_absolute_error(y_real, y_pred)
rmse = np.sqrt(mean_squared_error(y_real, y_pred))
```

Ejemplo: real `[10,20,30]`, predicción `[12,18,33]`: MAE = 2,333; MSE = 5,667; RMSE = 2,380. Medir después de invertir el escalado. RMSE 4,40 no significa 4,40 % ni exactitud de 95,60 %.

### Redes neuronales

| Concepto | Qué recordar |
|---|---|
| Neurona | `z = W*x+b`; `a = activacion(z)` |
| Activaciones | ReLU: `max(0,z)`; sigmoide: `1/(1+exp(-z))`; tanh: rango `[-1,1]` |
| Regresión | `Dense(1)` lineal produce un valor continuo |
| Clasificación | Sigmoide para binaria; softmax para clases excluyentes; pérdida acorde |
| Parámetros | Pesos y sesgos aprendidos |
| Hiperparámetros | Ventana, neuronas, tasa de aprendizaje, batch y máximo de épocas |
| Época | Recorrido del conjunto de entrenamiento |
| Batch | Grupo de muestras usado para actualizar los pesos |
| Gradiente | `W <- W-learning_rate*gradiente`; Adam adapta las actualizaciones |
| Backpropagation | Calcula derivadas de la pérdida; en redes recurrentes recorre el tiempo |
| LSTM | Estado de celda y compuertas de olvido, entrada y salida |
| Sobreajuste | Train mejora mientras validación empeora de forma sostenida |
| EarlyStopping | Detiene por falta de mejora; `restore_best_weights=True` recupera los mejores pesos |

Práctica: `Input((7,28)) -> LSTM(50) -> Dense(1)`. Pérdida MAE, Adam, batch 64, máximo 100 épocas y `shuffle=False`. EarlyStopping monitorea MAE de validación (`val_loss`) con paciencia 10. Se elige la configuración por RMSE de validación. El modelo final se reentrena durante 19 épocas con todo el historial.

### Resultados y errores que debes reconocer

Datos sintéticos: diez productos, 365 días y 3.650 ventas. LSTM con FE y siete días: RMSE val 4,760; RMSE test 4,400; MAE test 3,103. Base semanal: RMSE 4,391; MAE 3,294. La red mejora el MAE de esa base, pero no su RMSE. El agregado mezcla unidades y kg.

Evitar ajustar el escalador con test, mezclar productos en una ventana, usar el cierre del día objetivo como entrada o elegir hiperparámetros mirando prueba. Las métricas del experimento no son una prueba independiente del modelo reentrenado. Evaluar un día con historia observada no evalúa automáticamente el pronóstico recursivo a 60 días.

### Fuentes

- [Tutoriales de Pandas](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html).
- [Métricas de scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics).
- [LSTM de Keras](https://keras.io/api/layers/recurrent_layers/lstm/) y [EarlyStopping](https://keras.io/api/callbacks/early_stopping/).
- Resumen y salidas guardadas de [notebook de la práctica](notebooks/Practica_Inventario_LSTM.ipynb).

## Preparación para la evaluación

Plan y preguntas basados en la práctica LSTM. Son material de preparación propuesto, no preguntas oficiales del docente. Si empiezas después del 2 de noviembre, retoma los temas pendientes y ajusta los tiempos.

### Plan diario propuesto

| Día | Trabajo | Resultado que debes conseguir |
|---|---|---|
| Lunes 2 de noviembre | 60-75 min: resumen y fases del notebook | Explicar qué entra, qué estima la red y cómo se calcula stock |
| Martes 3 | 75-90 min: Pandas; ejercicios 1 y 2 | Cargar, validar, filtrar, agrupar y unir sin copiar |
| Miércoles 4 | 75-90 min: series y métricas; ejercicios 3 a 6 | Crear rezagos sin fuga y medir MAE/RMSE |
| Jueves 5 | 75-90 min: redes; ejercicio 7 | Explicar arquitectura, épocas, batch, pérdida, Adam y EarlyStopping |
| Viernes 6 | 60-75 min: MLflow y Airflow | Explicar runs, DAG, dependencias y reintentos |
| Sábado 7 | 90 min: simulacro y corrección | Resolver el caso y anotar los errores |
| Domingo 8 | 45-60 min: repetir lo fallado | Cerrar dudas y practicar los pasos difíciles |
| Lunes 9 | 15-20 min: repaso del formulario | Recordar fórmulas y formas de tensores |

En cada sesión intenta responder de memoria antes de consultar. Después vuelve a escribir la parte que falló. Ajusta los tiempos a tu disponibilidad.

### Preguntas de teoría con respuestas

1. **¿Qué aprende la red?** Cantidad vendida diaria como aproximación a demanda. Si hay ventas limitadas por stock, puede existir demanda no observada.
2. **¿Los datos son reales?** Son sintéticos, con semilla 42; no son registros oficiales de La Favorita.
3. **¿Por qué separar por fecha?** Para entrenar con pasado y evaluar fechas posteriores.
4. **¿Qué es fuga?** Usar información que no estaría disponible al predecir, como ajustar el escalador con test o usar la venta objetivo en una media.
5. **¿Se puede usar calendario futuro?** El calendario conocido sí está disponible antes de observar las ventas. Hay que distinguirlo de eventos conocidos solo después.
6. **¿Qué forma tiene X?** `(muestras,pasos,variables)`; en la configuración elegida, train tiene `(2920,7,28)` y la salida `(2920,1)`.
7. **¿Qué significan las 50 unidades LSTM?** La dimensión del estado oculto; son distintas de los siete pasos y de las 28 variables.
8. **¿Pérdida y métrica son iguales?** La pérdida guía el aprendizaje; las métricas permiten evaluar. Aquí se optimiza MAE y se comparan configuraciones por RMSE de validación.
9. **¿Qué hace EarlyStopping?** Vigila `val_loss`, espera diez épocas sin mejora y recupera los mejores pesos.
10. **¿Cómo reconocer sobreajuste?** La pérdida de train mejora mientras la de validación empeora de manera sostenida.
11. **¿Por qué usar bases?** Permiten comprobar qué aporta la complejidad de la red. La base semanal tiene RMSE 4,391 frente a 4,400 de la LSTM.
12. **¿RMSE 4,40 es un porcentaje?** Es un error en escala del objetivo; el agregado de esta práctica mezcla unidades y kg.
13. **¿La red predice stock?** Predice demanda; el cierre se calcula después con disponibilidad y reposición.
14. **¿El error de stock siempre coincide con demanda?** Sin recorte, sus errores tienen igual magnitud. Al actuar `max(...,0)`, la igualdad puede cambiar.
15. **¿Qué cambia al reentrenar?** Se aprovecha todo el historial durante 19 épocas; no se obtiene una nueva prueba independiente con ello.
16. **¿Qué es pronóstico recursivo?** Incorporar predicciones a la historia para estimar días siguientes. El error puede acumularse.
17. **¿Qué conservar para reproducir?** Modelo, escaladores, orden de columnas, configuración, historial, política de stock, código y versiones.
18. **¿Cómo combinar MLflow y Airflow?** Airflow coordina las tareas; cada entrenamiento registra parámetros, métricas y artefactos en MLflow.

Las cifras proceden del resumen y las salidas del notebook. Para la API de redes, consulta [LSTM](https://keras.io/api/layers/recurrent_layers/lstm/) y [EarlyStopping](https://keras.io/api/callbacks/early_stopping/).

### Simulacro de 60 minutos

Resuelve los siete ejercicios de la siguiente sección en una sesión o un notebook de prueba, desde la raíz del repositorio. Intenta cada ejercicio antes de desplegar su solución.

| Minutos | Tarea | Puntos propuestos |
|---|---|---:|
| 0-8 | Cargar, ordenar y validar CSV | 15 |
| 8-16 | Agregar y unir catálogo; ordenar dentro de cada unidad comercial | 10 |
| 16-27 | Crear rezagos y media móvil sin fuga | 20 |
| 27-35 | Separar por fechas y escalar con train | 15 |
| 35-43 | Construir ventanas de siete días | 15 |
| 43-50 | Medir persistencia y base semanal en los mismos días | 15 |
| 50-60 | Definir LSTM y explicar entrenamiento | 10 |

La puntuación es para autoevaluación. Si una ventana mezcla productos o el cálculo usa el futuro, corrige esa parte aunque el código ejecute. La última solución solo entrena una tanda pequeña y no reproduce las métricas del proyecto.

### Antes de la evaluación

Explica el flujo completo en cinco minutos, calcula MAE y RMSE a mano en un ejemplo pequeño y escribe sin copiar los rezagos por grupo. Repite los ejercicios que necesitaron solución. Preparar estos archivos no sustituye el estudio personal.

## Práctica de live coding

Ejecuta desde la raíz del repositorio, usando las dependencias de `requirements.txt`. Las soluciones comparten variables: ejecutarlas en orden en una misma sesión. Solo leen los CSV y trabajan en memoria; no necesitan modelos, figuras o tablas externos.

Intenta cada enunciado antes de desplegar la solución. Las comprobaciones usan los datos sintéticos de esta práctica. El ejercicio 7 entrena una tanda pequeña para verificar la API de Keras y no reproduce las métricas del modelo completo.

### 1. Cargar y validar (8 min)

Carga y ordena los CSV. Conserva los ID como texto. Comprueba nulos, claves duplicadas, continuidad diaria, diez productos y coherencia entre ventas y disponibilidad.

<details>
<summary>Solución y comprobaciones</summary>

```python
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display
DATA=Path('data')
ventas=pd.read_csv(DATA/'sales_history.csv',dtype={'product_id':str},parse_dates=['fecha_venta'])
inventario=pd.read_csv(DATA/'inventory.csv',dtype={'product_id':str})
ventas=ventas.sort_values(['product_id','fecha_venta']).reset_index(drop=True)
assert ventas.shape==(3650,5)
assert inventario['product_id'].nunique()==10 and inventario['product_id'].is_unique
assert not ventas.isna().any().any() and not inventario.isna().any().any()
assert not ventas.duplicated(['product_id','fecha_venta']).any()
assert set(ventas['product_id'])==set(inventario['product_id'])
assert ventas['cantidad_vendida'].between(0,ventas['stock_en_momento']).all()
assert ventas.groupby('product_id').size().eq(365).all()
assert ventas.groupby('product_id')['fecha_venta'].diff().dropna().eq(pd.Timedelta(days=1)).all()
display(ventas.head())
print('Datos: OK')
```

</details>

### 2. Agrupar y unir (8 min)

Calcula suma y promedio por producto. Añade nombre, categoría y unidad con una unión validada. Presenta los tres productos de mayor cantidad dentro de cada unidad comercial.

<details>
<summary>Solución y comprobaciones</summary>

```python
resumen=(ventas.groupby('product_id')['cantidad_vendida']
         .agg(total='sum',promedio='mean').reset_index()
         .merge(inventario[['product_id','nombre','categoria','unidad']],
                on='product_id',validate='one_to_one'))
top=resumen.sort_values('total',ascending=False).groupby('unidad').head(3)
assert len(resumen)==10 and not resumen.isna().any().any()
assert np.isclose(resumen['total'].sum(),ventas['cantidad_vendida'].sum())
assert top.groupby('unidad').size().eq(3).all()
display(top[['product_id','nombre','unidad','total']])
```

</details>

### 3. Variables sin fuga (11 min)

Cada fila representa el día objetivo t. Crea lag_1, lag_7 y la media de los siete días anteriores por producto. Demuestra que modificar la venta objetivo no cambia esa media. En el notebook principal la fila describe d y el calendario de d+1; su shift(6) corresponde al rezago semanal de ese objetivo.

<details>
<summary>Solución y comprobaciones</summary>

```python
variables=ventas.copy()
g=variables.groupby('product_id')['cantidad_vendida']
variables['lag_1']=g.shift(1)
variables['lag_7']=g.shift(7)
variables['media_7']=g.transform(lambda s:s.shift(1).rolling(7).mean())
variables['dia_semana']=variables['fecha_venta'].dt.dayofweek
primeras=variables.groupby('product_id').head(7)
assert primeras['lag_7'].isna().all() and primeras['media_7'].isna().all()
p=variables.loc[variables['product_id'].eq('103665')].reset_index(drop=True)
assert p.loc[7,'lag_1']==p.loc[6,'cantidad_vendida']
assert p.loc[7,'lag_7']==p.loc[0,'cantidad_vendida']
assert np.isclose(p.loc[7,'media_7'],p.loc[:6,'cantidad_vendida'].mean())
alterada=p['cantidad_vendida'].copy()
alterada.loc[7]+=10000
assert np.isclose(alterada.shift(1).rolling(7).mean().iloc[7],p.loc[7,'media_7'])
display(p[['fecha_venta','cantidad_vendida','lag_1','lag_7','media_7']].head(9))
```

</details>

### 4. Separar y escalar (8 min)

Separa train antes del 02/08/2026, validación hasta el 31/08 y prueba desde el 01/09. Ajusta MinMaxScaler solo con train sobre las tres variables históricas. Todavía son entradas 2D, no ventanas LSTM.

<details>
<summary>Solución y comprobaciones</summary>

```python
from sklearn.preprocessing import MinMaxScaler
limpias=variables.dropna(subset=['lag_1','lag_7','media_7'])
train=limpias.loc[limpias['fecha_venta']<'2026-08-02']
val=limpias.loc[(limpias['fecha_venta']>='2026-08-02')&(limpias['fecha_venta']<'2026-09-01')]
test=limpias.loc[limpias['fecha_venta']>='2026-09-01']
cols=['lag_1','lag_7','media_7']
scaler=MinMaxScaler().fit(train[cols])
X_train_2d,X_val_2d,X_test_2d=[scaler.transform(df[cols]) for df in (train,val,test)]
assert train['fecha_venta'].max()<val['fecha_venta'].min()<test['fecha_venta'].min()
assert len(train)+len(val)+len(test)==len(limpias)
assert len(val)==len(test)==300
assert np.allclose(scaler.data_min_,train[cols].min())
assert np.allclose(scaler.data_max_,train[cols].max())
assert all(np.isfinite(x).all() for x in (X_train_2d,X_val_2d,X_test_2d))
print('Formas 2D:',X_train_2d.shape,X_val_2d.shape,X_test_2d.shape)
```

</details>

### 5. Ventanas de tiempo (8 min)

Implementa ventanas(serie,n_dias) para una serie numérica 1D. Devuelve X=(muestras,n_dias,1) e y=(muestras,1). Rechaza ventanas inválidas y series insuficientes. Comprueba [10,20,30,40,50] con ventana 3 y aplícalo a un solo producto de train.

<details>
<summary>Solución y comprobaciones</summary>

```python
def ventanas(serie,n_dias):
    serie=np.asarray(serie,dtype='float32')
    if isinstance(n_dias,bool) or not isinstance(n_dias,int) or n_dias<=0:
        raise ValueError('n_dias debe ser un entero positivo')
    if serie.ndim!=1 or len(serie)<=n_dias or not np.isfinite(serie).all():
        raise ValueError('Se requiere una serie 1D finita con más datos que la ventana')
    X=np.stack([serie[t-n_dias:t] for t in range(n_dias,len(serie))])
    return X[...,None],serie[n_dias:,None]
X_ej,y_ej=ventanas([10,20,30,40,50],3)
assert X_ej.shape==(2,3,1) and y_ej.shape==(2,1)
assert X_ej[0,:,0].tolist()==[10,20,30] and y_ej[:,0].tolist()==[40,50]
for serie,n_dias in [([1,2],2),([1,2],0),([1,2],1.5),([[1],[2]],1),([1,np.nan],1)]:
    try:
        ventanas(serie,n_dias)
    except ValueError:
        pass
    else:
        raise AssertionError('Entrada inválida aceptada')
serie_train=ventas.loc[ventas['product_id'].eq('103665')&(ventas['fecha_venta']<'2026-08-02'),['cantidad_vendida']]
escala_serie=MinMaxScaler().fit(serie_train)
X_lstm,y_lstm=ventanas(escala_serie.transform(serie_train).ravel(),7)
assert X_lstm.shape[1:]==(7,1) and len(X_lstm)==len(y_lstm)
print('Ventanas de un producto:',X_lstm.shape,y_lstm.shape)
```

</details>

### 6. Métricas y bases (7 min)

Comprueba MAE y RMSE para real [10,20,30] y predicción [12,18,33]. Evalúa persistencia y base semanal sobre las mismas 300 filas de septiembre. Contrasta con los resultados guardados de la práctica, sin depender de un CSV de métricas externo.

<details>
<summary>Solución y comprobaciones</summary>

```python
from sklearn.metrics import mean_absolute_error,mean_squared_error
real,pred=np.array([10,20,30]),np.array([12,18,33])
assert np.isclose(mean_absolute_error(real,pred),7/3)
assert np.isclose(np.sqrt(mean_squared_error(real,pred)),np.sqrt(17/3))
prueba=variables.loc[variables['fecha_venta']>='2026-09-01'].dropna(subset=['lag_1','lag_7'])
assert len(prueba)==300
metricas=pd.DataFrame([
    {'modelo':col,'RMSE':np.sqrt(mean_squared_error(prueba['cantidad_vendida'],prueba[col])),
     'MAE':mean_absolute_error(prueba['cantidad_vendida'],prueba[col])}
    for col in ['lag_1','lag_7']])
assert np.allclose(metricas['RMSE'],[6.467266295223457,4.391233691496426])
assert np.allclose(metricas['MAE'],[4.505333333333333,3.294])
display(metricas)
print('Las líneas base coinciden con las métricas guardadas: OK')
```

</details>

### 7. Definir una LSTM (10 min)

Construye Input((7,1)), LSTM(50), Dense(1), MAE y Adam. Comprueba la forma de salida y entrena una tanda pequeña de train. Prepara EarlyStopping y explica cómo usarías validation_data sin tocar test. La entrega principal utiliza 28 variables.

<details>
<summary>Solución y comprobaciones</summary>

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input,LSTM,Dense
from tensorflow.keras.callbacks import EarlyStopping
tf.keras.utils.set_random_seed(42)
modelo_demo=Sequential([Input(shape=(7,1)),LSTM(50),Dense(1)])
modelo_demo.compile(optimizer='adam',loss='mae')
parada=EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
assert modelo_demo(X_lstm[:2]).shape==(2,1)
perdida=modelo_demo.train_on_batch(X_lstm[:16],y_lstm[:16])
assert np.isfinite(perdida)
print('Tanda de entrenamiento comprobada; MAE normalizada:',float(perdida))
print('Para un entrenamiento completo se usaría fit con validation_data,')
print('epochs=100, batch_size=64, shuffle=False y callbacks=[parada].')
print('Esta comprobación no mide generalización.')
```

</details>

## MLflow y Airflow

Documentación oficial consultada el 2 de noviembre de 2026. Los ejemplos son didácticos: los servicios no están instalados ni se han ejecutado en este proyecto. El ejemplo de DAG usa Airflow 3.

### Qué gestiona cada herramienta

| Pregunta | MLflow | Airflow |
|---|---|---|
| ¿Qué organiza? | Experimentos, runs, métricas y artefactos | Flujos de tareas y dependencias |
| ¿Qué permite revisar? | Configuración y resultados de un modelo | Estado, logs, programación y reintentos de tareas |
| Uso en la práctica | Comparar las cuatro configuraciones LSTM | Coordinar carga, validación, entrenamiento y evaluación |

MLflow Tracking agrupa runs dentro de experimentos. Un parámetro describe una configuración, una métrica cuantifica un resultado y un artefacto es un archivo producido durante la ejecución. [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/).

Airflow representa un flujo mediante un DAG, un grafo dirigido sin ciclos. El scheduler programa tareas; el executor determina cómo ejecutarlas. El procesador interpreta los DAG; el servidor API y su interfaz permiten inspeccionarlos, y la base de metadatos conserva sus estados. [Arquitectura de Airflow](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/overview.html).

### MLflow: cómo se utiliza

Crear un experimento y abrir un run por configuración. Registrar ventana, FE, semilla y métricas de validación. Conservar modelo, escaladores y configuración como artefactos correspondientes a ese entrenamiento. Comparar runs en la interfaz y elegir con validación antes de revisar prueba.

Ejemplo independiente de la práctica: registra las métricas de validación guardadas en el notebook, sin volver a entrenar ni necesitar la carpeta `tables/`.

En un entorno aparte con MLflow instalado, iniciar un servidor local:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Desde otra terminal de ese entorno, ejecutar el siguiente Python. Los archivos del servidor se crean en el directorio desde el que se inicia; usar un directorio de pruebas fuera de la entrega.

```python
import mlflow

mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('Inventario_LSTM_estudio')
resultados = [
    ('LSTM sin FE, 7 dias', 7, False, 7.805),
    ('LSTM con FE, 3 dias', 3, True, 4.931),
    ('LSTM con FE, 7 dias', 7, True, 4.760),
    ('LSTM con FE, 14 dias', 14, True, 7.095),
]
assert len(resultados) == 4
for nombre, ventana, fe, rmse in resultados:
    with mlflow.start_run(run_name=nombre):
        mlflow.log_params({'ventana': ventana, 'feature_engineering': fe,
                           'semilla': 42})
        mlflow.log_metric('rmse_val', rmse)
        mlflow.set_tag('origen', 'resultados_guardados_redondeados')
```

Abrir `http://127.0.0.1:5000` y comparar los cuatro runs. La configuración de siete días con FE tiene menor RMSE de validación. Este ejemplo importa resultados redondeados y no crea modelos entrenados. [Inicio rápido de MLflow](https://mlflow.org/docs/latest/ml/tracking/quickstart/).

En un entrenamiento nuevo con Keras 3, `mlflow.keras.autolog()` puede registrar información de `model.fit()`. Para conservar y recuperar el modelo con su formato de MLflow, se usan `mlflow.keras.log_model(...)` y `mlflow.keras.load_model(...)`. `log_artifact` conserva un archivo, pero no crea por sí solo ese formato de modelo. [API para Keras](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.keras.html).

Los pesos del reentrenamiento final con todo el historial no corresponden a los modelos con los que se calcularon las métricas de prueba. En una integración futura, conservar un modelo y la versión de sus datos por run.

### Airflow: cómo se utiliza

TaskFlow define tareas con `@task` y un flujo con `@dag`. Las llamadas establecen dependencias. `retries` permite reintentar; `schedule=None` deja la ejecución manual. [Tutorial TaskFlow](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/taskflow.html).

Ejemplo pequeño de cuatro tareas. Usa datos incluidos en el código y calcula una base de promedio; no automatiza la LSTM:

```python
from datetime import datetime, timezone
from airflow.sdk import dag, task

@dag(schedule=None,
     start_date=datetime(2026, 11, 2, tzinfo=timezone.utc),
     catchup=False, tags=['estudio'])
def practica_demanda_demo():
    @task(retries=1)
    def cargar():
        return [10.0, 12.0, 11.0, 13.0]

    @task
    def validar(valores):
        if not valores or any(x < 0 for x in valores):
            raise ValueError('Ventas vacias o negativas')
        return valores

    @task
    def calcular_base(valores):
        return sum(valores) / len(valores)

    @task
    def reportar(promedio):
        print(f'Promedio de la muestra: {promedio:.2f}')

    reportar(calcular_base(validar(cargar())))

practica_demanda_demo()
```

Guardar como `practica_demanda_demo.py` en la carpeta de DAG de una instalación de Airflow 3. Revisar errores de importación y lanzar manualmente desde la interfaz. El resultado esperado es `11.50`; queda pendiente comprobarlo con Airflow real.

En Windows, el desarrollo con Airflow requiere WSL2 o contenedores Linux. Utilizar un entorno separado del de TensorFlow. [Requisitos oficiales](https://airflow.apache.org/docs/apache-airflow/stable/installation/prerequisites.html).

Las tareas pueden ejecutarse en equipos distintos. XCom está pensado para cantidades pequeñas de datos. En un flujo real, guardar ventas y modelos en almacenamiento compartido y pasar rutas o identificadores, en lugar de DataFrames completos. [XComs](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/xcoms.html).

### Cómo se combinarían

```text
Airflow: cargar -> validar -> preparar -> entrenar -> evaluar -> guardar
                                        |
                           MLflow: parámetros, métricas y artefactos
```

Esta es una propuesta para una evolución del proyecto. Primero extraer las fases del notebook en funciones que no dependan del estado de celdas anteriores. Abrir un run de MLflow dentro del entrenamiento y pasar su identificador a las tareas siguientes. Conservar trazabilidad de los datos, evitar duplicaciones al reintentar y evaluar antes de promover un modelo.

### Repaso oral

- Parámetro, métrica y artefacto: ventana de siete días, RMSE y archivo del modelo.
- Falló una descarga: revisar tarea, logs y política de reintentos en Airflow.
- Comparar ventanas: un run por configuración y la misma partición de validación.
- Compartir entre tareas: rutas, identificadores y metadatos pequeños.
- Automatizar nuestra red: el DAG de ejemplo solo enseña dependencias; la integración LSTM requiere conectar sus fases.
