"""EA3 - Pregunta de negocio y construcción automática del DataFrame."""
from __future__ import annotations

from typing import Any

import pandas as pd

from .hf_loader import HFDatasetLoader


class DatasetPodioCarreras:
    """Documenta y carga el dataset de carreras de caballos usado en EA3.

    Reúne en un solo lugar la pregunta de negocio, el origen del dataset y las
    funciones que muestran el DataFrame (tamaño, tipos, muestra) para que el
    notebook solo tenga que llamarlas.
    """

    DATASET_ID = "mayacheruty/Horse-Race-Prediction-EDA"
    ENLACE = "https://huggingface.co/datasets/mayacheruty/Horse-Race-Prediction-EDA"
    CONFIG = "default"
    SPLIT = "train"
    ARCHIVO_ORIGEN = "data/horse_racing_cleaned.csv"
    LICENCIA = "MIT (según la ficha oficial de Hugging Face)"

    PREGUNTA_NEGOCIO = (
        "¿Qué factores conocidos antes de la salida determinan que un caballo "
        "termine en posición de pago (podio), para que un analista de apuestas "
        "hípicas decida a qué caballos de cada carrera dar prioridad en sus "
        "apuestas de colocado?"
    )
    QUIEN_SE_BENEFICIA = (
        "Un analista o apostador de carreras de caballos que debe repartir un "
        "presupuesto limitado entre los caballos de cada carrera."
    )
    DECISION_QUE_APOYA = (
        "Elegir, antes de cada carrera, qué caballos merecen una apuesta de "
        "colocado y cuáles factores (mercado, rating, peso, edad, tamaño del "
        "campo) pesan más en esa elección."
    )
    OBJETIVO = "res_place"
    CLASES = {0: "Fuera del podio", 1: "En posición de pago"}

    # columna -> (rol en el proyecto, motivo)
    COLUMNAS_RELEVANTES: dict[str, tuple[str, str]] = {
        "rid": ("Agrupación", "Identifica la carrera; sirve para dividir train/test sin mezclar carreras"),
        "res_place": ("Objetivo", "1 si el caballo terminó en posición de pago, 0 si no"),
        "decimalPrice": ("Predictora", "Probabilidad implícita del mercado de apuestas, conocida antes de la salida"),
        "OR": ("Predictora", "Rating oficial previo a la carrera (con valores imputados por el autor)"),
        "age": ("Predictora", "Edad del caballo"),
        "weight": ("Predictora", "Peso que carga el caballo"),
        "runners": ("Predictora", "Número de corredores: define el tamaño del campo"),
        "saddle": ("Predictora", "Número de silla o puesto de salida (0 = dato faltante)"),
        "position": ("Excluida (fuga)", "Posición final: revela el objetivo"),
        "res_win": ("Excluida (fuga)", "Indica si ganó: es parte del resultado"),
        "margin": ("Excluida (fuga)", "Diferencia en la llegada: solo existe tras la carrera"),
        "positionL": ("Excluida (fuga)", "Distancia al ganador en la llegada"),
        "dist": ("Excluida (fuga)", "Distancia acumulada a la llegada"),
        "RPR": ("Excluida (fuga)", "Rating calculado después de la carrera"),
        "TR": ("Excluida (fuga)", "Rating calculado después de la carrera"),
        "isFav": ("Excluida", "Redundante: se deriva del precio"),
        "weightSt": ("Excluida", "Redundante con weight"),
        "weightLb": ("Excluida", "Redundante con weight"),
        "horseName": ("Excluida", "Texto de identificación"),
        "trainerName": ("Excluida", "Texto de alta cardinalidad"),
        "jockeyName": ("Excluida", "Texto de alta cardinalidad"),
        "father": ("Excluida", "Texto de alta cardinalidad"),
        "mother": ("Excluida", "Texto de alta cardinalidad"),
        "gfather": ("Excluida", "Texto de alta cardinalidad"),
    }

    def __init__(self, max_rows: int | None = None) -> None:
        self.max_rows = max_rows
        self.cargador: HFDatasetLoader | None = None
        self.df: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    # Carga
    # ------------------------------------------------------------------ #
    def cargar(self) -> pd.DataFrame:
        """Descarga el dataset desde Hugging Face y devuelve el DataFrame."""
        self.cargador = HFDatasetLoader(
            dataset_name=self.DATASET_ID,
            config=None,
            split=self.SPLIT,
            max_rows=self.max_rows,
        )
        self.df = self.cargador.load()
        return self.df

    def informacion(self) -> dict[str, Any]:
        """Identificación del dataset y del DataFrame obtenido."""
        if self.cargador is None:
            raise RuntimeError("Llama a cargar() primero.")
        return {
            **self.cargador.info(),
            "dataset": self.DATASET_ID,
            "config": self.CONFIG,
            "enlace": self.ENLACE,
            "archivo_origen": self.ARCHIVO_ORIGEN,
            "licencia": self.LICENCIA,
        }

    # ------------------------------------------------------------------ #
    # Evidencia del DataFrame
    # ------------------------------------------------------------------ #
    @staticmethod
    def resumen_dataframe(df: pd.DataFrame, filas_muestra: int = 8) -> dict[str, Any]:
        """Tamaño, tipos de datos y muestra del DataFrame."""
        tipos = pd.DataFrame(
            {
                "tipo": df.dtypes.astype(str),
                "nulos": df.isna().sum(),
                "valores_distintos": df.nunique(),
            }
        )
        return {
            "filas": int(df.shape[0]),
            "columnas": int(df.shape[1]),
            "lista_columnas": list(df.columns),
            "tipos": tipos,
            "muestra": df.head(filas_muestra),
        }

    @classmethod
    def tabla_columnas(cls) -> pd.DataFrame:
        """Rol de cada columna en el proyecto y el motivo de la decisión."""
        return pd.DataFrame(
            [(c, rol, motivo) for c, (rol, motivo) in cls.COLUMNAS_RELEVANTES.items()],
            columns=["columna", "rol", "motivo"],
        ).set_index("columna")

    @classmethod
    def verificar_objetivo(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Muestra qué posiciones finales corresponden a ``res_place``.

        Sirve como evidencia de que el objetivo representa "terminar en
        posición de pago" y de que ``position`` no puede ser predictora.
        """
        faltantes = {cls.OBJETIVO, "position", "runners"} - set(df.columns)
        if faltantes:
            raise KeyError(f"Faltan columnas: {sorted(faltantes)}.")
        tabla = (
            df.groupby(cls.OBJETIVO)["position"]
            .agg(filas="size", posicion_minima="min", posicion_mediana="median", posicion_maxima="max")
        )
        tabla.insert(0, "clase", [cls.CLASES.get(int(i), str(i)) for i in tabla.index])
        tabla["porcentaje"] = (tabla["filas"] / tabla["filas"].sum() * 100).round(2)
        return tabla
