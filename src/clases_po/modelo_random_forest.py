"""EA3 - Random Forest para predecir si un caballo termina en posición de pago."""
from __future__ import annotations

import itertools
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from .clasificacion import ClasificadorBinario


class ModeloPodioRandomForest:
    """Random Forest binario con selección de hiperparámetros por validación cruzada.

    Parámetros principales del bosque:

    - ``n_estimators``: número de árboles; más árboles estabilizan la
      predicción pero cuestan más tiempo.
    - ``max_depth``: profundidad máxima de cada árbol; limita el sobreajuste.
    - ``min_samples_leaf``: mínimo de filas por hoja; suaviza las probabilidades.
    - ``class_weight``: pondera la clase minoritaria (los que sí hacen podio).
    - ``random_state``: semilla, para repetir exactamente el resultado.
    """

    def __init__(
        self,
        umbral: float = 0.5,
        class_weight: str | dict | None = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        if not 0 < umbral < 1:
            raise ValueError("umbral debe estar entre 0 y 1, sin incluirlos.")
        self.umbral = umbral
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.parametros: dict[str, Any] = {
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_leaf": 50,
        }
        self.modelo: RandomForestClassifier | None = None
        self.ruta_modelo: Path | None = None
        self.fecha_entrenamiento: str | None = None

    # ------------------------------------------------------------------ #
    # Construcción del bosque
    # ------------------------------------------------------------------ #
    def _crear_bosque(self, **parametros: Any) -> RandomForestClassifier:
        config = {**self.parametros, **parametros}
        return RandomForestClassifier(
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            **config,
        )

    # ------------------------------------------------------------------ #
    # Selección de hiperparámetros
    # ------------------------------------------------------------------ #
    def seleccionar_hiperparametros(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        grupos_train: pd.Series,
        profundidades: tuple[int, ...] = (4, 8, 12),
        hojas_minimas: tuple[int, ...] = (20, 100),
        arboles: tuple[int, ...] = (50, 150, 300),
        n_splits: int = 5,
        n_arboles_busqueda: int = 100,
    ) -> dict[str, pd.DataFrame]:
        """Elige los hiperparámetros con validación cruzada por carrera.

        Usa **solo el entrenamiento** (``GroupKFold``: una carrera nunca queda
        repartida entre validación y entrenamiento). Criterio de elección:
        PR-AUC media, porque no depende del umbral y es exigente con la clase
        positiva. Dos etapas: primero profundidad y tamaño de hoja con un
        número fijo de árboles, luego el número de árboles con la mejor pareja.

        Devuelve ``{"profundidad_hoja": DataFrame, "n_arboles": DataFrame}`` y
        deja la mejor configuración en ``self.parametros``.
        """
        if not (len(X_train) == len(y_train) == len(grupos_train)):
            raise ValueError("X_train, y_train y grupos_train deben tener la misma longitud.")
        pliegues = list(GroupKFold(n_splits=n_splits).split(X_train, y_train, grupos_train))

        etapa1 = [
            self._validar_configuracion(
                X_train, y_train, pliegues,
                {"max_depth": d, "min_samples_leaf": h, "n_estimators": n_arboles_busqueda},
            )
            for d, h in itertools.product(profundidades, hojas_minimas)
        ]
        tabla1 = pd.DataFrame(etapa1).sort_values("pr_auc_cv", ascending=False).reset_index(drop=True)
        mejor1 = tabla1.iloc[0]
        profundidad, hoja = int(mejor1["max_depth"]), int(mejor1["min_samples_leaf"])

        etapa2 = [
            self._validar_configuracion(
                X_train, y_train, pliegues,
                {"max_depth": profundidad, "min_samples_leaf": hoja, "n_estimators": n},
            )
            for n in arboles
        ]
        tabla2 = pd.DataFrame(etapa2).sort_values("n_estimators").reset_index(drop=True)
        # Menor número de árboles cuya PR-AUC está a menos de 0,001 de la mejor.
        mejor_pr = tabla2["pr_auc_cv"].max()
        elegido = tabla2[tabla2["pr_auc_cv"] >= mejor_pr - 0.001].iloc[0]
        self.parametros = {
            "n_estimators": int(elegido["n_estimators"]),
            "max_depth": profundidad,
            "min_samples_leaf": hoja,
        }
        print(f"[seleccionar_hiperparametros] mejor configuración: {self.parametros}")
        return {"profundidad_hoja": tabla1, "n_arboles": tabla2}

    def _validar_configuracion(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        pliegues: list[tuple[np.ndarray, np.ndarray]],
        parametros: dict[str, Any],
    ) -> dict[str, Any]:
        pr, roc, pr_entrenamiento = [], [], []
        for idx_tr, idx_va in pliegues:
            bosque = self._crear_bosque(**parametros).fit(X.iloc[idx_tr], y.iloc[idx_tr])
            p_va = bosque.predict_proba(X.iloc[idx_va])[:, 1]
            pr.append(average_precision_score(y.iloc[idx_va], p_va))
            roc.append(roc_auc_score(y.iloc[idx_va], p_va))
            p_tr = bosque.predict_proba(X.iloc[idx_tr])[:, 1]
            pr_entrenamiento.append(average_precision_score(y.iloc[idx_tr], p_tr))
        return {
            **parametros,
            "pr_auc_cv": float(np.mean(pr)),
            "pr_auc_cv_desv": float(np.std(pr)),
            "roc_auc_cv": float(np.mean(roc)),
            "pr_auc_entrenamiento": float(np.mean(pr_entrenamiento)),
        }

    # ------------------------------------------------------------------ #
    # Entrenamiento y persistencia
    # ------------------------------------------------------------------ #
    def entrenar(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        directorio_entrenamiento: str | Path | None = "entrenamiento_ea3",
    ) -> RandomForestClassifier:
        """Entrena el bosque con los parámetros vigentes y, si se pide, lo guarda."""
        self.modelo = self._crear_bosque().fit(X_train, y_train)
        self.fecha_entrenamiento = datetime.now().isoformat(timespec="seconds")
        if directorio_entrenamiento is not None:
            carpeta = Path(directorio_entrenamiento)
            carpeta.mkdir(parents=True, exist_ok=True)
            marca = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.ruta_modelo = carpeta / f"{marca}_random_forest_podio.joblib"
            joblib.dump(self.modelo, self.ruta_modelo, compress=3)
        return self.modelo

    def hiperparametros_completos(self) -> dict[str, str]:
        """Todos los parámetros del bosque (incluye los valores por defecto)."""
        if self.modelo is None:
            raise RuntimeError("Entrena el modelo primero.")
        return {k: str(v) for k, v in sorted(self.modelo.get_params().items())}

    def predecir_probabilidad(self, X: pd.DataFrame) -> np.ndarray:
        if self.modelo is None:
            raise RuntimeError("Entrena el modelo primero.")
        return self.modelo.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------ #
    # Evaluación
    # ------------------------------------------------------------------ #
    def evaluar(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        grupos: pd.Series,
        probabilidades: np.ndarray | None = None,
        umbral: float | None = None,
    ) -> dict[str, Any]:
        """Métricas de clasificación y del uso real (ranking dentro de cada carrera)."""
        corte = self.umbral if umbral is None else umbral
        prob = self.predecir_probabilidad(X) if probabilidades is None else probabilidades
        reales = np.asarray(y, dtype=int)
        pred = (prob >= corte).astype(int)
        matriz = confusion_matrix(reales, pred, labels=[0, 1])
        tn, fp, fn, tp = (int(v) for v in matriz.ravel())
        tasa = float(reales.mean())
        metricas: dict[str, Any] = {
            "umbral": float(corte),
            "filas": int(len(reales)),
            "tasa_positiva": tasa,
            "exactitud": float((tn + tp) / len(reales)),
            "exactitud_baseline": float(1 - tasa),
            "exactitud_balanceada": float(balanced_accuracy_score(reales, pred)),
            "precision": float(precision_score(reales, pred, zero_division=0)),
            "recall": float(recall_score(reales, pred, zero_division=0)),
            "f1": float(f1_score(reales, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(reales, prob)),
            "pr_auc": float(average_precision_score(reales, prob)),
            "pr_auc_azar": tasa,
            "matriz_confusion": matriz.tolist(),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        }
        metricas.update(self._acierto_por_carrera(prob, reales, np.asarray(grupos), np.asarray(X["decimalPrice"])))
        return metricas

    @staticmethod
    def _acierto_por_carrera(
        prob: np.ndarray, reales: np.ndarray, grupos: np.ndarray, precio: np.ndarray, k: int = 3
    ) -> dict[str, float]:
        """De los ``k`` caballos mejor puntuados por carrera, ¿qué fracción hizo podio?

        Se compara con elegir al azar y con la regla simple "los ``k`` favoritos
        del mercado" (mayor ``decimalPrice``). Es la métrica más cercana a la
        decisión del analista: elegir a quién apostar dentro de cada carrera.
        """
        tabla = pd.DataFrame({"g": grupos, "p": prob, "y": reales, "precio": precio})
        tabla = tabla[tabla.groupby("g")["y"].transform("size") >= k]

        def precision_topk(columna: str) -> float:
            elegidos = tabla.sort_values(["g", columna], ascending=[True, False]).groupby("g").head(k)
            return float(elegidos["y"].mean())

        azar = float(tabla["y"].mean())
        return {
            "k_top": k,
            "carreras_top_k": int(tabla["g"].nunique()),
            "precision_top_k_modelo": precision_topk("p"),
            "precision_top_k_mercado": precision_topk("precio"),
            "precision_top_k_azar": azar,
        }

    def baseline_mercado(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Referencia: usar solo la probabilidad del mercado como puntaje."""
        precio = np.asarray(X["decimalPrice"])
        reales = np.asarray(y, dtype=int)
        return {
            "roc_auc": float(roc_auc_score(reales, precio)),
            "pr_auc": float(average_precision_score(reales, precio)),
        }

    def desempeno_por_campo(
        self, X: pd.DataFrame, y: pd.Series, probabilidades: np.ndarray
    ) -> pd.DataFrame:
        """ROC-AUC y PR-AUC según el tamaño del campo (límite de confiabilidad)."""
        rangos = pd.cut(X["runners"], bins=[0, 7, 11, 15, 100], labels=["≤7", "8-11", "12-15", "16+"])
        filas = []
        for etiqueta in rangos.cat.categories:
            mascara = (rangos == etiqueta).to_numpy()
            if mascara.sum() < 50 or len(np.unique(np.asarray(y)[mascara])) < 2:
                continue
            filas.append(
                {
                    "corredores": etiqueta,
                    "filas": int(mascara.sum()),
                    "tasa_podio": float(np.asarray(y)[mascara].mean()),
                    "roc_auc": float(roc_auc_score(np.asarray(y)[mascara], probabilidades[mascara])),
                    "pr_auc": float(average_precision_score(np.asarray(y)[mascara], probabilidades[mascara])),
                }
            )
        return pd.DataFrame(filas).set_index("corredores")

    def calibracion(self, y: pd.Series, probabilidades: np.ndarray, grupos_cuantiles: int = 10) -> pd.DataFrame:
        """Probabilidad media predicha vs tasa real de podio por decil de probabilidad."""
        tabla = pd.DataFrame({"p": probabilidades, "y": np.asarray(y, dtype=int)})
        tabla["decil"] = pd.qcut(tabla["p"].rank(method="first"), grupos_cuantiles, labels=False) + 1
        resumen = tabla.groupby("decil").agg(
            filas=("y", "size"), prob_predicha=("p", "mean"), tasa_real=("y", "mean")
        )
        resumen["diferencia"] = resumen["prob_predicha"] - resumen["tasa_real"]
        return resumen

    def sensibilidad_umbral(self, y: pd.Series, probabilidades: np.ndarray) -> pd.DataFrame:
        """Precisión, recall y F1 para varios umbrales (reutiliza ``ClasificadorBinario``)."""
        return ClasificadorBinario.sensibilidad_umbral(
            y, probabilidades, umbrales=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
        )

    # ------------------------------------------------------------------ #
    # Importancia de variables
    # ------------------------------------------------------------------ #
    def importancia_variables(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_repeticiones: int = 5,
        max_filas: int = 30000,
    ) -> pd.DataFrame:
        """Importancia por impureza (entrenamiento) y por permutación (prueba).

        La permutación mide cuánto empeora la PR-AUC al desordenar una variable
        en datos no vistos; es más fiable que la impureza, que favorece a las
        variables con muchos valores distintos.
        """
        if self.modelo is None:
            raise RuntimeError("Entrena el modelo primero.")
        if len(X_test) > max_filas:
            muestra = X_test.sample(max_filas, random_state=self.random_state).index
            X_perm, y_perm = X_test.loc[muestra], y_test.loc[muestra]
        else:
            X_perm, y_perm = X_test, y_test
        permutacion = permutation_importance(
            self.modelo, X_perm, y_perm, scoring="average_precision",
            n_repeats=n_repeticiones, random_state=self.random_state, n_jobs=self.n_jobs,
        )
        tabla = pd.DataFrame(
            {
                "importancia_impureza": self.modelo.feature_importances_,
                "importancia_permutacion": permutacion.importances_mean,
                "permutacion_desv": permutacion.importances_std,
            },
            index=X_test.columns,
        ).sort_values("importancia_permutacion", ascending=False)
        positivos = tabla["importancia_permutacion"].clip(lower=0)
        tabla["peso_relativo_pct"] = positivos / positivos.sum() * 100 if positivos.sum() > 0 else 0.0
        return tabla
