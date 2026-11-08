"""EA3 - Preprocesamiento para predecir si un caballo termina en posición de pago."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from .preprocesamiento import PreprocesamientoCarreras


class PreprocesamientoPodio(PreprocesamientoCarreras):
    """Prepara ``X``, ``y`` y grupos para el Random Forest de EA3.

    Se reutiliza de ``PreprocesamientoCarreras`` la detección de ratings
    imputados, la eliminación de duplicados, la división por carrera y la
    bitácora de operaciones.

    - **Objetivo** ``res_place`` (binario). No se construye a partir de otra
      variable, pero ``position`` y demás columnas de resultado quedan
      prohibidas como predictoras (control de fuga de información).
    - **Enriquecimiento** con variables relativas dentro de la carrera, que un
      analista sí conoce antes de la salida: probabilidad de mercado
      normalizada, ranking de mercado, peso y rating respecto al promedio del
      campo.
    - **Imputación** con la mediana, ajustada solo con el entrenamiento.
    - **Sin escalado**: un Random Forest divide por umbrales y no cambia al
      reescalar las variables; escalar solo complicaría la interpretación.
    """

    PREDICTORAS_BASE = ("decimalPrice", "age", "weight", "runners", "saddle", "OR")
    PREDICTORAS_DERIVADAS = (
        "prob_mercado_norm",
        "rank_mercado",
        "weight_rel",
        "OR_rel",
        "OR_faltante",
    )

    def __init__(
        self,
        columna_grupo: str = "rid",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        super().__init__(columna_grupo=columna_grupo, test_size=test_size, random_state=random_state)
        self.imputador: SimpleImputer | None = None
        self.nulos_antes_imputar: dict[str, int] = {}

    @property
    def predictoras_finales(self) -> list[str]:
        """Predictoras base más derivadas, en el orden en que entran al modelo."""
        return list(self.PREDICTORAS_BASE) + list(self.PREDICTORAS_DERIVADAS)

    # ------------------------------------------------------------------ #
    # Preparación
    # ------------------------------------------------------------------ #
    def preparar_podio(
        self,
        df: pd.DataFrame,
        objetivo: str = "res_place",
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Limpia, enriquece y devuelve ``X``, ``y`` y los grupos (carreras).

        Pasos (cada uno queda en ``resumen_operaciones()``):
        eliminar duplicados, devolver a ``NaN`` los valores imputados por el
        autor (``OR``, ``saddle == 0``), crear variables relativas a la
        carrera, descartar filas sin objetivo o sin variables esenciales y
        convertir el objetivo a entero.

        Raises:
            ValueError: Si el objetivo no es binario o una predictora revela
                el resultado de la carrera.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df debe ser un pandas.DataFrame.")
        self._validar_sin_fuga(self.predictoras_finales, objetivo)
        self._validar_columnas(df, list(self.PREDICTORAS_BASE) + [objetivo, self.columna_grupo])

        self.bitacora = []
        datos = self.eliminar_duplicados(df)

        # 1) Valores imputados por el autor del dataset -> NaN
        datos = self.restaurar_nulos_imputados(datos, columnas=["OR"])
        datos = datos.copy()
        saddle_cero = int((datos["saddle"] == 0).sum())
        datos.loc[datos["saddle"] == 0, "saddle"] = np.nan
        self._registrar(
            "restaurar_saddle_faltante",
            "El autor usó 0 como marcador de 'sin dato' en saddle; no es un puesto real",
            len(datos),
            len(datos),
            f"saddle == 0 devuelto a NaN: {saddle_cero} filas",
        )

        # 2) Selección y tipos numéricos
        columnas_base = list(self.PREDICTORAS_BASE) + [objetivo, self.columna_grupo]
        sel = datos[columnas_base].copy()
        numericas = list(self.PREDICTORAS_BASE) + [objetivo]
        sel[numericas] = sel[numericas].apply(pd.to_numeric, errors="coerce")
        sel[numericas] = sel[numericas].where(np.isfinite(sel[numericas]), np.nan)
        self._registrar(
            "seleccionar_y_convertir",
            "Quitar identificadores, texto de alta cardinalidad y columnas de resultado; forzar tipos numéricos",
            len(sel),
            len(sel),
            f"{len(columnas_base)} de {len(datos.columns)} columnas conservadas",
        )

        # 3) Filas sin objetivo o sin variables esenciales (no se imputan)
        esenciales = [objetivo, "decimalPrice", "age", "weight", "runners"]
        antes = len(sel)
        nulos = sel[esenciales].isna().sum()
        sel = sel.dropna(subset=esenciales).reset_index(drop=True)
        detalle = ", ".join(f"{c}={int(n)}" for c, n in nulos.items() if n > 0) or "ninguno"
        self._registrar(
            "descartar_filas_sin_datos_esenciales",
            "El objetivo no debe imputarse y el precio/edad/peso/campo son esenciales",
            antes,
            len(sel),
            f"nulos por columna: {detalle}",
        )

        # 4) Objetivo binario entero
        valores = set(sel[objetivo].unique())
        if not valores <= {0, 1} or len(valores) < 2:
            raise ValueError(
                f"El objetivo debe ser binario (0 y 1); contiene {sorted(float(v) for v in valores)}."
            )
        sel[objetivo] = sel[objetivo].astype(int)
        self._registrar(
            "convertir_objetivo",
            "El objetivo llega como número (0 / 1); el clasificador necesita enteros",
            len(sel),
            len(sel),
            f"'{objetivo}' convertido a entero, clases {sorted(int(v) for v in valores)}",
        )

        # 5) Enriquecimiento: variables relativas a la carrera
        sel = self._agregar_variables_relativas(sel)
        self._registrar(
            "enriquecer_variables_relativas",
            "El precio, el peso y el rating solo informan al compararse con los rivales de la misma carrera",
            len(sel),
            len(sel),
            "creadas: " + ", ".join(self.PREDICTORAS_DERIVADAS),
        )

        self.objetivo = objetivo
        self.columnas_predictoras = self.predictoras_finales
        X = sel[self.columnas_predictoras]
        y = sel[objetivo]
        grupos = sel[self.columna_grupo]
        print(
            f"[preparar_podio] {len(self.columnas_predictoras)} predictoras; y='{objetivo}'; "
            f"{len(sel)} filas de {len(df)}; {grupos.nunique()} carreras."
        )
        return X, y, grupos

    def _agregar_variables_relativas(self, sel: pd.DataFrame) -> pd.DataFrame:
        """Crea las variables que comparan cada caballo con su carrera."""
        g = sel.groupby(self.columna_grupo)
        suma_precio = g["decimalPrice"].transform("sum")
        sel["prob_mercado_norm"] = np.where(suma_precio > 0, sel["decimalPrice"] / suma_precio, np.nan)
        sel["rank_mercado"] = g["decimalPrice"].rank(method="min", ascending=False)
        sel["weight_rel"] = sel["weight"] - g["weight"].transform("mean")
        sel["OR_rel"] = sel["OR"] - g["OR"].transform("mean")
        sel["OR_faltante"] = sel["OR"].isna().astype(int)
        return sel

    def _validar_sin_fuga(self, predictoras: list[str], objetivo: str) -> None:
        prohibidas = set(self.COLUMNAS_RESULTADO) | set(self.COLUMNAS_POSTERIORES)
        fuga = sorted(set(predictoras) & prohibidas)
        if fuga:
            raise ValueError(
                f"Estas predictoras revelan el resultado de la carrera o se calculan después de ella: {fuga}."
            )
        if objetivo in predictoras:
            raise ValueError("El objetivo no puede estar entre las predictoras.")

    # ------------------------------------------------------------------ #
    # Imputación (solo con entrenamiento)
    # ------------------------------------------------------------------ #
    def imputar(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Rellena los nulos con la mediana calculada solo con el entrenamiento."""
        self.nulos_antes_imputar = {
            "entrenamiento": int(X_train.isna().sum().sum()),
            "prueba": int(X_test.isna().sum().sum()),
        }
        self.imputador = SimpleImputer(strategy="median").fit(X_train)
        X_train_imp = pd.DataFrame(
            self.imputador.transform(X_train), index=X_train.index, columns=X_train.columns
        )
        X_test_imp = pd.DataFrame(
            self.imputador.transform(X_test), index=X_test.index, columns=X_test.columns
        )
        self._registrar(
            "imputar_mediana",
            "El modelo no admite nulos; la mediana se aprende solo del entrenamiento para no filtrar la prueba",
            len(X_train),
            len(X_train),
            f"nulos rellenados: entrenamiento={self.nulos_antes_imputar['entrenamiento']}, "
            f"prueba={self.nulos_antes_imputar['prueba']}",
        )
        print("[imputar] SimpleImputer(mediana) ajustado con entrenamiento y aplicado a prueba.")
        return X_train_imp, X_test_imp

    # ------------------------------------------------------------------ #
    # Evidencia antes / después
    # ------------------------------------------------------------------ #
    def comparar_antes_despues(
        self, df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, grupos: pd.Series
    ) -> pd.DataFrame:
        """Tabla comparativa del DataFrame original y del preparado (antes de imputar)."""
        or_imputados = int(self.detectar_imputados(df, columnas=["OR"]).loc["OR", "valores_imputados"])
        base = list(self.PREDICTORAS_BASE)
        antes = {
            "filas": len(df),
            "columnas": df.shape[1],
            "carreras": int(df[self.columna_grupo].nunique()),
            "filas duplicadas": int(df.duplicated().sum()),
            "OR imputados con la media": or_imputados,
            "saddle = 0 (sin dato)": int((df["saddle"] == 0).sum()),
            "nulos en las predictoras base": int(df[base].isna().sum().sum()),
            "% en posición de pago": round(float(pd.to_numeric(df[self.objetivo]).mean() * 100), 2),
        }
        despues = {
            "filas": len(X),
            "columnas": X.shape[1],
            "carreras": int(grupos.nunique()),
            "filas duplicadas": 0,
            "OR imputados con la media": 0,
            "saddle = 0 (sin dato)": int((X["saddle"] == 0).sum()),
            "nulos en las predictoras base": int(X[base].isna().sum().sum()),
            "% en posición de pago": round(float(y.mean() * 100), 2),
        }
        return pd.DataFrame({"antes (original)": antes, "después (preparado)": despues})

    # ------------------------------------------------------------------ #
    # Persistencia
    # ------------------------------------------------------------------ #
    def guardar(self, ruta: str | Path) -> Path:
        """Guarda el imputador y las columnas para reutilizarlos al predecir."""
        if self.imputador is None:
            raise RuntimeError("Llama a imputar() antes de guardar.")
        destino = Path(ruta)
        destino.parent.mkdir(parents=True, exist_ok=True)
        with destino.open("wb") as archivo:
            pickle.dump(
                {
                    "imputador": self.imputador,
                    "columnas": self.columnas_predictoras,
                    "objetivo": self.objetivo,
                },
                archivo,
            )
        return destino
