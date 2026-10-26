"""Preprocesamiento del dataset de carreras de caballos (Horse-Race-Prediction-EDA)."""
from __future__ import annotations
 
import pickle
from pathlib import Path
from typing import Any
 
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
 
 
class PreprocesamientoCarreras:
    """Prepara el dataset de carreras para los modelos de regresión y clasificación.
 
    Cada operación queda registrada en ``bitacora`` (qué problema resuelve y
    cómo cambió el número de filas) para poder presentarla como evidencia.
 
    Decisiones principales:
 
    - Los ratings ``OR`` y ``RPR`` publicados contienen valores imputados con la
      media por el autor del dataset. Los ratings reales son enteros y el valor
      imputado es un único decimal, por lo que se detectan sin usar estadísticas
      globales y se devuelven a ``NaN``.
    - Las columnas que describen el resultado de la carrera (``position``,
      ``positionL``, ``dist``, ``margin``, ``res_win``, ``res_place``) no pueden
      ser predictoras porque revelan la respuesta.
    - La división entrenamiento/prueba se hace por carrera (``rid``), para que
      los caballos de una misma carrera no queden repartidos entre ambos
      conjuntos.
    - El escalado se ajusta únicamente con el conjunto de entrenamiento.
    """
 
    COLUMNAS_RATING = ("OR", "RPR")
    COLUMNAS_RESULTADO = ("position", "positionL", "dist", "margin", "res_win", "res_place")
 
    def __init__(
        self,
        columna_grupo: str = "rid",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        if not isinstance(columna_grupo, str) or not columna_grupo.strip():
            raise ValueError("columna_grupo debe ser un nombre de columna válido.")
        if not 0 < test_size < 1:
            raise ValueError("test_size debe estar entre 0 y 1.")
        if not isinstance(random_state, int) or isinstance(random_state, bool) or random_state < 0:
            raise ValueError("random_state debe ser un entero mayor o igual que cero.")
 
        self.columna_grupo = columna_grupo
        self.test_size = test_size
        self.random_state = random_state
        self.escalador: StandardScaler | None = None
        self.columnas_predictoras: list[str] = []
        self.objetivo: str | None = None
        self.bitacora: list[dict[str, Any]] = []
        self.info_division: dict[str, Any] | None = None
 
    # ------------------------------------------------------------------ #
    # Diagnóstico
    # ------------------------------------------------------------------ #
    @staticmethod
    def diagnosticar(df: pd.DataFrame) -> pd.DataFrame:
        """Resume tipo, nulos y valores distintos de cada columna."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df debe ser un pandas.DataFrame.")
        return pd.DataFrame(
            {
                "tipo": df.dtypes.astype(str),
                "nulos": df.isna().sum(),
                "pct_nulos": (df.isna().mean() * 100).round(2),
                "distintos": df.nunique(),
            }
        )
 
    def detectar_imputados(
        self,
        df: pd.DataFrame,
        columnas: tuple[str, ...] | list[str] | None = None,
    ) -> pd.DataFrame:
        """Cuenta los valores no enteros de cada rating (huella de la imputación)."""
        columnas = list(columnas or self.COLUMNAS_RATING)
        self._validar_columnas(df, columnas)
        filas = []
        for columna in columnas:
            serie = df[columna]
            imputados = self._mascara_imputados(serie)
            filas.append(
                {
                    "columna": columna,
                    "valores_imputados": int(imputados.sum()),
                    "pct_imputados": round(float(imputados.mean() * 100), 2),
                    "valores_decimales_distintos": int(serie[imputados].nunique()),
                    "valor_imputado": (
                        float(serie[imputados].iloc[0]) if imputados.any() else None
                    ),
                }
            )
        return pd.DataFrame(filas).set_index("columna")
 
    # ------------------------------------------------------------------ #
    # Limpieza
    # ------------------------------------------------------------------ #
    def restaurar_nulos_imputados(
        self,
        df: pd.DataFrame,
        columnas: tuple[str, ...] | list[str] | None = None,
    ) -> pd.DataFrame:
        """Devuelve a ``NaN`` los ratings imputados con la media.
 
        Problema que resuelve: los valores imputados no son mediciones reales.
        Un 40 % de ``OR`` y un 11 % de ``RPR`` valen lo mismo, lo que crea un
        pico artificial y sesga cualquier modelo que los use.
        """
        columnas = list(columnas or self.COLUMNAS_RATING)
        self._validar_columnas(df, columnas)
        resultado = df.copy()
        detalle = []
        for columna in columnas:
            imputados = self._mascara_imputados(resultado[columna])
            resultado.loc[imputados, columna] = np.nan
            detalle.append(f"{columna}={int(imputados.sum())}")
        self._registrar(
            "restaurar_nulos_imputados",
            "Ratings imputados con la media por el autor del dataset",
            len(df),
            len(resultado),
            "valores devueltos a NaN: " + ", ".join(detalle),
        )
        return resultado
 
    def eliminar_duplicados(self, df: pd.DataFrame) -> pd.DataFrame:
        """Elimina filas completamente duplicadas."""
        resultado = df.drop_duplicates().reset_index(drop=True)
        self._registrar(
            "eliminar_duplicados",
            "Filas repetidas que contarían dos veces en el entrenamiento",
            len(df),
            len(resultado),
            f"duplicados eliminados: {len(df) - len(resultado)}",
        )
        return resultado
 
    def preparar_regresion(
        self,
        df: pd.DataFrame,
        objetivo: str = "RPR",
        predictoras: list[str] | tuple[str, ...] = ("OR", "age", "weight", "decimalPrice", "runners"),
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Construye ``X``, ``y`` y los grupos (carreras) para la regresión lineal.
 
        Pasos: eliminar duplicados, restaurar los ratings imputados como nulos,
        seleccionar columnas, convertir a numérico y descartar las filas con
        nulos en el objetivo o en las predictoras.
 
        Raises:
            KeyError: Si falta alguna columna.
            ValueError: Si una predictora es una columna de resultado (fuga de
                información) o coincide con el objetivo.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df debe ser un pandas.DataFrame.")
        predictoras = list(predictoras)
        if objetivo in predictoras:
            raise ValueError("El objetivo no puede estar entre las predictoras.")
        fuga = sorted(set(predictoras) & set(self.COLUMNAS_RESULTADO))
        if fuga:
            raise ValueError(f"Estas predictoras revelan el resultado de la carrera: {fuga}.")
        self._validar_columnas(df, predictoras + [objetivo, self.columna_grupo])
 
        self.bitacora = []
        datos = self.eliminar_duplicados(df)
        datos = self.restaurar_nulos_imputados(datos)
 
        columnas = predictoras + [objetivo, self.columna_grupo]
        seleccion = datos[columnas].copy()
        self._registrar(
            "seleccionar_columnas",
            "Quitar identificadores, texto de alta cardinalidad, columnas redundantes y de resultado",
            len(seleccion),
            len(seleccion),
            f"{len(columnas)} de {len(datos.columns)} columnas conservadas",
        )
 
        numericas = predictoras + [objetivo]
        seleccion[numericas] = seleccion[numericas].apply(pd.to_numeric, errors="coerce")
        seleccion[numericas] = seleccion[numericas].where(np.isfinite(seleccion[numericas]), np.nan)
        antes = len(seleccion)
        nulos_por_columna = seleccion[numericas].isna().sum()
        seleccion = seleccion.dropna(subset=numericas).reset_index(drop=True)
        self._registrar(
            "descartar_filas_con_nulos",
            "Un modelo lineal no admite nulos y el objetivo no debe imputarse",
            antes,
            len(seleccion),
            "nulos por columna: "
            + ", ".join(f"{c}={int(n)}" for c, n in nulos_por_columna.items() if n > 0),
        )
        if len(seleccion) < 10 or seleccion[objetivo].nunique() < 2:
            raise ValueError("No quedan datos suficientes para entrenar.")
 
        self.objetivo = objetivo
        self.columnas_predictoras = predictoras
        X = seleccion[predictoras]
        y = seleccion[objetivo]
        grupos = seleccion[self.columna_grupo]
        print(
            f"[preparar_regresion] X={predictoras}; y='{objetivo}'; "
            f"{len(seleccion)} filas de {len(df)}; {grupos.nunique()} carreras."
        )
        return X, y, grupos
 
    # ------------------------------------------------------------------ #
    # División y escalado
    # ------------------------------------------------------------------ #
    def dividir_por_carrera(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        grupos: pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Divide en entrenamiento y prueba sin separar las filas de una carrera.
 
        Devuelve ``(X_train, X_test, y_train, y_test)``. Una carrera completa
        queda en un solo conjunto, así el modelo se evalúa con carreras que no
        vio durante el entrenamiento.
        """
        if not (len(X) == len(y) == len(grupos)):
            raise ValueError("X, y y grupos deben tener la misma longitud.")
        divisor = GroupShuffleSplit(
            n_splits=1, test_size=self.test_size, random_state=self.random_state
        )
        idx_train, idx_test = next(divisor.split(X, y, groups=grupos))
        X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
        y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]
        compartidas = set(grupos.iloc[idx_train]) & set(grupos.iloc[idx_test])
        self.info_division = {
            "filas_entrenamiento": len(X_train),
            "filas_prueba": len(X_test),
            "carreras_entrenamiento": int(grupos.iloc[idx_train].nunique()),
            "carreras_prueba": int(grupos.iloc[idx_test].nunique()),
            "carreras_compartidas": len(compartidas),
            "test_size": self.test_size,
            "random_state": self.random_state,
        }
        self._registrar(
            "dividir_por_carrera",
            "Evitar que caballos de la misma carrera estén en entrenamiento y prueba",
            len(X),
            len(X),
            f"entrenamiento={len(X_train)}, prueba={len(X_test)}, "
            f"carreras compartidas={len(compartidas)}",
        )
        print(
            f"[dividir_por_carrera] entrenamiento={len(X_train)} filas; "
            f"prueba={len(X_test)} filas; carreras compartidas={len(compartidas)}."
        )
        return X_train, X_test, y_train, y_test
 
    def escalar(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Estandariza las predictoras ajustando solo con el entrenamiento.
 
        La media y la desviación se calculan con ``X_train`` y se aplican a
        ``X_test``. Para una regresión lineal el escalado no cambia las
        predicciones, pero permite comparar los coeficientes entre variables
        con unidades distintas.
        """
        self.escalador = StandardScaler().fit(X_train)
        X_train_esc = pd.DataFrame(
            self.escalador.transform(X_train), index=X_train.index, columns=X_train.columns
        )
        X_test_esc = pd.DataFrame(
            self.escalador.transform(X_test), index=X_test.index, columns=X_test.columns
        )
        self._registrar(
            "escalar",
            "Variables con unidades distintas (puntos de rating, kg, años) no son comparables",
            len(X_train),
            len(X_train),
            "StandardScaler ajustado solo con entrenamiento",
        )
        print("[escalar] StandardScaler ajustado con entrenamiento y aplicado a prueba.")
        return X_train_esc, X_test_esc
 
    def resumen_operaciones(self) -> pd.DataFrame:
        """Devuelve la bitácora de operaciones como DataFrame."""
        return pd.DataFrame(self.bitacora)
 
    def guardar(self, ruta: str | Path) -> Path:
        """Guarda el escalador y las columnas para reutilizarlos al predecir."""
        if self.escalador is None:
            raise RuntimeError("Llama a escalar() antes de guardar.")
        destino = Path(ruta)
        destino.parent.mkdir(parents=True, exist_ok=True)
        with destino.open("wb") as archivo:
            pickle.dump(
                {
                    "escalador": self.escalador,
                    "columnas": self.columnas_predictoras,
                    "objetivo": self.objetivo,
                },
                archivo,
            )
        return destino
 
    # ------------------------------------------------------------------ #
    # Auxiliares
    # ------------------------------------------------------------------ #
    @staticmethod
    def _mascara_imputados(serie: pd.Series) -> pd.Series:
        """Marca los valores no enteros: los ratings reales siempre son enteros."""
        valores = pd.to_numeric(serie, errors="coerce")
        return valores.notna() & (valores % 1 != 0)
 
    @staticmethod
    def _validar_columnas(df: pd.DataFrame, columnas: list[str]) -> None:
        faltantes = sorted(set(columnas).difference(df.columns))
        if faltantes:
            raise KeyError(f"Faltan columnas requeridas: {faltantes}.")
 
    def _registrar(
        self,
        operacion: str,
        problema: str,
        filas_antes: int,
        filas_despues: int,
        detalle: str,
    ) -> None:
        self.bitacora.append(
            {
                "operacion": operacion,
                "problema_que_resuelve": problema,
                "filas_antes": filas_antes,
                "filas_despues": filas_despues,
                "detalle": detalle,
            }
        )
