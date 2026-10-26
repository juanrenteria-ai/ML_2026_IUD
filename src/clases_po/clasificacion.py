"""Flujo reutilizable para entrenar y evaluar un clasificador binario."""
from __future__ import annotations
 
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any
 
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, cross_validate
 
 
class ClasificadorBinario:
    """Regresión logística para un objetivo binario con clases desbalanceadas.
 
    ``class_weight="balanced"`` hace que cada error sobre la clase minoritaria
    pese más durante el entrenamiento, en proporción inversa a su frecuencia.
    Sin esto, con 10 % de positivos el modelo tiende a responder siempre
    "negativo" y acierta el 90 % sin detectar nada.
    """
 
    def __init__(
        self,
        umbral: float = 0.5,
        class_weight: str | dict | None = "balanced",
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        if not 0 < umbral < 1:
            raise ValueError("umbral debe estar entre 0 y 1, sin incluirlos.")
        self.umbral = umbral
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.modelo: LogisticRegression | None = None
        self.ruta_modelo: Path | None = None
 
    # ------------------------------------------------------------------ #
    # Entrenamiento y persistencia
    # ------------------------------------------------------------------ #
    def entrenar(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        directorio_entrenamiento: str | Path = "entrenamiento",
    ) -> dict[str, Any]:
        """Ajusta la regresión logística y la guarda como ``.pkl`` con fecha y hora."""
        self._validar_pares(X_train, y_train)
        if set(np.unique(y_train)) != {0, 1}:
            raise ValueError("y_train debe contener las dos clases, 0 y 1.")
 
        modelo = LogisticRegression(
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        modelo.fit(X_train, y_train)
        self.modelo = modelo
 
        carpeta = Path(directorio_entrenamiento)
        carpeta.mkdir(parents=True, exist_ok=True)
        fecha = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        ruta = carpeta / f"{fecha}_clasificador.pkl"
        with ruta.open("wb") as archivo:
            pickle.dump(modelo, archivo)
        self.ruta_modelo = ruta
        print(
            f"[entrenar] clasificador ajustado con {len(X_train)} filas y "
            f"{X_train.shape[1]} características; guardado en: {ruta}"
        )
        return {"modelo": modelo, "ruta_modelo": ruta, "fecha_modelo": fecha}
 
    @staticmethod
    def cargar(ruta_modelo: str | Path) -> LogisticRegression:
        """Carga un clasificador guardado por ``entrenar``."""
        ruta = Path(ruta_modelo)
        if not ruta.is_file():
            raise FileNotFoundError(f"No existe el modelo: {ruta}.")
        with ruta.open("rb") as archivo:
            modelo = pickle.load(archivo)
        if not isinstance(modelo, LogisticRegression):
            raise TypeError("El archivo no contiene un modelo LogisticRegression.")
        return modelo
 
    # ------------------------------------------------------------------ #
    # Predicción y evaluación
    # ------------------------------------------------------------------ #
    def predecir_probabilidad(self, ruta_modelo: str | Path, X: pd.DataFrame) -> np.ndarray:
        """Probabilidad estimada de la clase positiva para cada fila."""
        modelo = self.cargar(ruta_modelo)
        if not isinstance(X, pd.DataFrame) or X.empty:
            raise ValueError("X debe ser un DataFrame no vacío.")
        return modelo.predict_proba(X)[:, 1]
 
    def predecir(
        self, ruta_modelo: str | Path, X: pd.DataFrame, umbral: float | None = None
    ) -> np.ndarray:
        """Clase predicha (1 si la probabilidad alcanza el umbral)."""
        corte = self.umbral if umbral is None else umbral
        return (self.predecir_probabilidad(ruta_modelo, X) >= corte).astype(int)
 
    def obtener_parametros(
        self, ruta_modelo: str | Path, columnas: list[str], escalador: Any
    ) -> tuple[pd.DataFrame, float]:
        """Tabla de coeficientes y el intercepto en escala original.
 
        Los coeficientes de una regresión logística están en log-odds. El
        ``odds_ratio_por_desv`` indica cuánto se multiplican las probabilidades
        relativas de ganar (odds) por cada desviación estándar adicional de la
        variable, con las demás constantes: mayor que 1 favorece ganar.
        """
        modelo = self.cargar(ruta_modelo)
        coef_std = modelo.coef_.ravel().astype(float)
        if len(coef_std) != len(columnas):
            raise ValueError("La cantidad de columnas no coincide con los coeficientes.")
        coef_orig = coef_std / escalador.scale_
        intercepto = float(
            modelo.intercept_[0] - np.sum(coef_std * escalador.mean_ / escalador.scale_)
        )
        absolutos = np.abs(coef_std)
        tabla = pd.DataFrame(
            {
                "coef_estandarizado": coef_std,
                "odds_ratio_por_desv": np.exp(coef_std),
                "desv_variable": escalador.scale_,
                "coef_unidad_original": coef_orig,
                "peso_relativo_pct": absolutos / absolutos.sum() * 100,
            },
            index=columnas,
        ).sort_values("peso_relativo_pct", ascending=False)
        return tabla, intercepto
 
    def evaluar(
        self,
        ruta_modelo: str | Path,
        X: pd.DataFrame,
        y: pd.Series,
        grupos: pd.Series | None = None,
        umbral: float | None = None,
    ) -> dict[str, Any]:
        """Calcula métricas de clasificación sobre ``X`` e ``y``.
 
        ``grupos`` (opcional) es el identificador de carrera de cada fila; con
        él se calcula además cuántas veces el caballo con mayor probabilidad de
        cada carrera fue realmente el ganador.
        """
        corte = self.umbral if umbral is None else umbral
        prob = self.predecir_probabilidad(ruta_modelo, X)
        pred = (prob >= corte).astype(int)
        reales = np.asarray(y, dtype=int)
        if len(reales) != len(pred):
            raise ValueError("y y X deben tener la misma longitud.")
        matriz = confusion_matrix(reales, pred, labels=[0, 1])
        tn, fp, fn, tp = (int(v) for v in matriz.ravel())
        tasa_positiva = float(reales.mean())
        resultados: dict[str, Any] = {
            "umbral": float(corte),
            "filas": int(len(reales)),
            "tasa_positiva": tasa_positiva,
            "exactitud": float((tn + tp) / len(reales)),
            "exactitud_baseline": float(1 - tasa_positiva),
            "exactitud_balanceada": float(balanced_accuracy_score(reales, pred)),
            "precision": float(precision_score(reales, pred, zero_division=0)),
            "recall": float(recall_score(reales, pred, zero_division=0)),
            "f1": float(f1_score(reales, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(reales, prob)),
            "pr_auc": float(average_precision_score(reales, prob)),
            "pr_auc_azar": tasa_positiva,
            "matriz_confusion": matriz.tolist(),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }
        if grupos is not None:
            resultados.update(self._aciertos_por_carrera(prob, reales, np.asarray(grupos)))
        return resultados
 
    @staticmethod
    def _aciertos_por_carrera(
        prob: np.ndarray, reales: np.ndarray, grupos: np.ndarray
    ) -> dict[str, float]:
        """Porcentaje de carreras donde el caballo con mayor probabilidad ganó."""
        tabla = pd.DataFrame({"g": grupos, "p": prob, "y": reales})
        mejor = tabla.loc[tabla.groupby("g")["p"].idxmax()]
        azar = (1 / tabla.groupby("g").size()).mean()
        return {
            "carreras": int(tabla["g"].nunique()),
            "aciertos_top1_pct": float(mejor["y"].mean() * 100),
            "aciertos_top1_azar_pct": float(azar * 100),
        }
 
    @staticmethod
    def sensibilidad_umbral(
        y: pd.Series, prob: np.ndarray, umbrales: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)
    ) -> pd.DataFrame:
        """Precisión, recall y F1 para varios umbrales (solo informativo)."""
        reales = np.asarray(y, dtype=int)
        filas = []
        for corte in umbrales:
            pred = (prob >= corte).astype(int)
            filas.append(
                {
                    "umbral": corte,
                    "predichos_positivos": int(pred.sum()),
                    "precision": precision_score(reales, pred, zero_division=0),
                    "recall": recall_score(reales, pred, zero_division=0),
                    "f1": f1_score(reales, pred, zero_division=0),
                }
            )
        return pd.DataFrame(filas).set_index("umbral")
 
    # ------------------------------------------------------------------ #
    # Justificación del algoritmo
    # ------------------------------------------------------------------ #
    def comparar_algoritmos(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        grupos_train: pd.Series,
        n_splits: int = 5,
    ) -> pd.DataFrame:
        """Compara algoritmos con validación cruzada por carrera, solo con entrenamiento.
 
        La prueba no interviene, de modo que elegir el algoritmo con esta tabla
        no contamina la evaluación final.
        """
        modelos = {
            "Siempre la clase mayoritaria (referencia)": DummyClassifier(strategy="prior"),
            "Regresión logística": LogisticRegression(
                class_weight=self.class_weight, max_iter=self.max_iter, random_state=self.random_state
            ),
            "Gradient boosting (HistGradientBoosting)": HistGradientBoostingClassifier(
                class_weight=self.class_weight, max_iter=100, learning_rate=0.05,
                random_state=self.random_state,
            ),
            "Bosque aleatorio": RandomForestClassifier(
                n_estimators=100, min_samples_leaf=50, class_weight="balanced_subsample",
                n_jobs=-1, random_state=self.random_state,
            ),
        }
        puntajes = {
            "precision": make_scorer(precision_score, zero_division=0),
            "recall": make_scorer(recall_score, zero_division=0),
            "f1": make_scorer(f1_score, zero_division=0),
            "roc_auc": "roc_auc",
            "pr_auc": "average_precision",
        }
        validacion = GroupKFold(n_splits=n_splits)
        filas = {}
        for nombre, modelo in modelos.items():
            resultado = cross_validate(
                modelo, X_train, y_train, groups=grupos_train, cv=validacion,
                scoring=puntajes, error_score=0.0,
            )
            filas[nombre] = {m: float(np.mean(resultado[f"test_{m}"])) for m in puntajes}
        return pd.DataFrame(filas).T
 
    # ------------------------------------------------------------------ #
    # Validación
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validar_pares(X: pd.DataFrame, y: pd.Series) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X debe ser un pandas.DataFrame.")
        if not isinstance(y, pd.Series):
            raise TypeError("y debe ser una pandas.Series.")
        if X.empty or y.empty or len(X) != len(y):
            raise ValueError("X e y deben tener filas y longitudes compatibles.")
        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError("X contiene valores no numéricos o inválidos.")
 