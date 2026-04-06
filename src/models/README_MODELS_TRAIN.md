# Implementación — fase `src/models/train.py`

## Archivos incluidos

```text
src/models/train.py
src/models/__init__.py
notebooks/legacy/10_logistic_regression.py
notebooks/legacy/11_decision_tree.py
notebooks/legacy/12_random_forest.py
notebooks/legacy/13_xgboost.py
```

## Qué se implementó

- Nuevo módulo `src/models/train.py` con helpers públicos:
  - `train_logistic_regression()`
  - `train_decision_tree()`
  - `train_random_forest()`
  - `train_xgboost()`
- Actualización de `src/models/__init__.py` para exportar la nueva API.
- Reemplazo de los bloques repetidos `ModelClass(...) + fit(...)` en los notebooks legacy `10–13`.

## Qué no se tocó

- `splitting.py`
- `evaluation.py`
- `interpretation.py`
- tuning / búsqueda de hiperparámetros
- plotting específico de cada modelo

## Patrón aplicado

Antes:

```python
model = SomeClassifier(...)
model.fit(X_train, y_train)
```

Después:

```python
model = train_some_model(X_train, y_train, ...)
```

## Nota

Este paquete se generó fuera del repo remoto, usando como referencia el estado actual de la rama `migration/rename-thesis-naming` que compartiste en el handoff.
