"""Diagnósticos y textos de interpretación para los modelos entrenados."""
from __future__ import annotations
 
from typing import Any
 
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
 
class InterpretacionModelos:
    """Calcula diagnósticos y redacta la interpretación a partir de los resultados.
 
    Los textos se construyen con los valores medidos en la ejecución, de modo
    que el notebook y el reporte ``.txt`` siempre describen lo que realmente
    ocurrió y no un texto escrito a mano.
    """
 
    # ------------------------------------------------------------------ #
    # Regresión: cálculos
    # ------------------------------------------------------------------ #
    @staticmethod
    def metricas_regresion(y_real: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
        """Devuelve R², MAE y RMSE."""
        reales = np.asarray(y_real, dtype=float)
        estimadas = np.asarray(y_pred, dtype=float)
        if reales.shape != estimadas.shape or reales.size == 0:
            raise ValueError("y_real y y_pred deben tener la misma longitud y no estar vacíos.")
        return {
            "r2": float(r2_score(reales, estimadas)),
            "mae": float(mean_absolute_error(reales, estimadas)),
            "rmse": float(np.sqrt(mean_squared_error(reales, estimadas))),
        }
 
    @classmethod
    def baseline_media(cls, y_train: pd.Series, y_test: pd.Series) -> dict[str, float]:
        """Métricas de un modelo que siempre predice la media del entrenamiento."""
        prediccion = np.full(len(y_test), float(np.mean(y_train)))
        return cls.metricas_regresion(y_test, prediccion)
 
    @staticmethod
    def r2_variable_sola(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        columna: str,
    ) -> float:
        """R² en prueba de una regresión que usa una sola predictora."""
        modelo = LinearRegression().fit(X_train[[columna]], y_train)
        return float(modelo.score(X_test[[columna]], y_test))
 
    @staticmethod
    def coeficientes_regresion(
        modelo: LinearRegression,
        columnas: list[str],
        escalador: Any,
    ) -> tuple[pd.DataFrame, float]:
        """Tabla de coeficientes y el intercepto en unidades originales.
 
        El modelo se entrena con variables estandarizadas. ``coef_estandarizado``
        indica cuánto cambia el objetivo por cada desviación estándar de la
        variable; ``coef_unidad_original`` indica cuánto cambia por cada unidad
        de la variable sin escalar. Ambos describen el mismo modelo.
        """
        coef_std = np.asarray(modelo.coef_, dtype=float)
        if len(coef_std) != len(columnas):
            raise ValueError("La cantidad de columnas no coincide con los coeficientes.")
        coef_orig = coef_std / escalador.scale_
        intercepto = float(
            modelo.intercept_ - np.sum(coef_std * escalador.mean_ / escalador.scale_)
        )
        absolutos = np.abs(coef_std)
        tabla = pd.DataFrame(
            {
                "coef_estandarizado": coef_std,
                "desv_variable": escalador.scale_,
                "coef_unidad_original": coef_orig,
                "peso_relativo_pct": absolutos / absolutos.sum() * 100,
            },
            index=columnas,
        ).sort_values("peso_relativo_pct", ascending=False)
        return tabla, intercepto
 
    @staticmethod
    def factores_inflacion_varianza(X: pd.DataFrame) -> pd.Series:
        """VIF de cada predictora: mide cuánto se explica con las demás."""
        vif = {}
        for columna in X.columns:
            otras = [c for c in X.columns if c != columna]
            if not otras:
                vif[columna] = 1.0
                continue
            r2 = LinearRegression().fit(X[otras], X[columna]).score(X[otras], X[columna])
            vif[columna] = float(1.0 / (1.0 - r2)) if r2 < 1 else float("inf")
        return pd.Series(vif, name="VIF")
 
    @staticmethod
    def diagnostico_residuos(
        y_real: pd.Series,
        y_pred: np.ndarray,
        y_train: pd.Series,
    ) -> dict[str, Any]:
        """Resume la forma de los residuos (real - predicho)."""
        reales = pd.Series(np.asarray(y_real, dtype=float))
        estimadas = pd.Series(np.asarray(y_pred, dtype=float))
        residuos = reales - estimadas
        desviacion = float(residuos.std())
        cuartiles = pd.qcut(estimadas, 4, labels=["Q1", "Q2", "Q3", "Q4"])
        desv_cuartil = residuos.groupby(cuartiles, observed=True).std()
        minimo, maximo = float(np.min(y_train)), float(np.max(y_train))
        q_bajo = float(np.quantile(y_train, 0.10))
        q_alto = float(np.quantile(y_train, 0.90))
        return {
            "objetivo_p10_entrenamiento": q_bajo,
            "objetivo_p90_entrenamiento": q_alto,
            "residuo_medio_decil_bajo": float(residuos[reales <= q_bajo].mean()),
            "residuo_medio_decil_alto": float(residuos[reales >= q_alto].mean()),
            "media": float(residuos.mean()),
            "desviacion": desviacion,
            "asimetria": float(residuos.skew()),
            "curtosis_exceso": float(residuos.kurt()),
            "pct_fuera_2_desviaciones": float((residuos.abs() > 2 * desviacion).mean() * 100),
            "desv_por_cuartil": {str(k): float(v) for k, v in desv_cuartil.items()},
            "prediccion_min": float(estimadas.min()),
            "prediccion_max": float(estimadas.max()),
            "objetivo_min_entrenamiento": minimo,
            "objetivo_max_entrenamiento": maximo,
            "predicciones_fuera_de_rango": int(((estimadas < minimo) | (estimadas > maximo)).sum()),
        }
 
    # ------------------------------------------------------------------ #
    # Regresión: textos
    # ------------------------------------------------------------------ #
    @staticmethod
    def interpretar_regresion(
        objetivo: str,
        metricas_train: dict[str, float],
        metricas_test: dict[str, float],
        baseline: dict[str, float],
        coeficientes: pd.DataFrame,
        vif: pd.Series,
        residuos: dict[str, Any],
        correlaciones: pd.Series | None = None,
    ) -> list[str]:
        """Redacta la interpretación de la regresión con los valores medidos.
 
        ``correlaciones`` (opcional) es la correlación de cada predictora con el
        objetivo; permite avisar cuando el signo de un coeficiente es contrario
        al de su correlación.
        """
        r2, mae, rmse = metricas_test["r2"], metricas_test["mae"], metricas_test["rmse"]
        if r2 >= 0.7:
            calidad = "alto"
        elif r2 >= 0.4:
            calidad = "moderado"
        elif r2 >= 0.1:
            calidad = "bajo"
        else:
            calidad = "prácticamente nulo"
        mejora = (1 - mae / baseline["mae"]) * 100 if baseline["mae"] else 0.0
 
        lineas = [
            f"Ajuste: el modelo explica el {r2 * 100:.1f} % de la variación de {objetivo} en "
            f"carreras que no vio al entrenar (R² = {r2:.3f}, nivel {calidad}). "
            f"Un modelo que siempre predice la media obtiene R² = {abs(baseline['r2']):.3f}.",
            f"Error: el MAE es {mae:.2f} puntos de {objetivo}, un {mejora:.0f} % menor que el "
            f"del modelo de la media ({baseline['mae']:.2f}). El RMSE ({rmse:.2f}) es mayor que "
            f"el MAE porque penaliza más los errores grandes.",
        ]
 
        diferencia = metricas_train["r2"] - r2
        if diferencia > 0.05:
            lineas.append(
                f"Generalización: el R² cae de {metricas_train['r2']:.3f} (entrenamiento) a "
                f"{r2:.3f} (prueba); hay señales de sobreajuste."
            )
        else:
            lineas.append(
                f"Generalización: R² de {metricas_train['r2']:.3f} en entrenamiento y {r2:.3f} en "
                f"prueba (diferencia {diferencia:+.3f}); no se observa sobreajuste."
            )
 
        principal = coeficientes.index[0]
        fila = coeficientes.iloc[0]
        lineas.append(
            f"Coeficientes: {principal} concentra el {fila['peso_relativo_pct']:.0f} % del peso "
            f"relativo del modelo. Por cada punto adicional de {principal}, {objetivo} cambia "
            f"{fila['coef_unidad_original']:+.3f} puntos, con las demás variables constantes."
        )
        otras = []
        for nombre, fila in coeficientes.iloc[1:].iterrows():
            otras.append(
                f"{nombre} ({fila['coef_estandarizado']:+.2f} puntos por desviación estándar)"
            )
        if otras:
            lineas.append("Efecto de las demás variables: " + "; ".join(otras) + ".")
        pequenas = coeficientes.index[coeficientes["coef_estandarizado"].abs() < 1].tolist()
        if pequenas:
            lineas.append(
                "Variables con efecto pequeño (menos de 1 punto por desviación estándar): "
                + ", ".join(pequenas)
                + ". Los signos describen asociación, no causalidad."
            )
 
        if correlaciones is not None:
            for nombre, fila in coeficientes.iterrows():
                if nombre not in correlaciones.index:
                    continue
                correlacion = float(correlaciones[nombre])
                if abs(correlacion) >= 0.1 and np.sign(correlacion) != np.sign(fila["coef_estandarizado"]):
                    lineas.append(
                        f"Signo contrario: {nombre} tiene correlación {correlacion:+.2f} con "
                        f"{objetivo} pero coeficiente {fila['coef_estandarizado']:+.2f}. Una vez "
                        f"descontado el efecto de las demás variables su aporte cambia de "
                        f"dirección; no debe leerse de forma aislada."
                    )
 
        vif_max = float(vif.max())
        if vif_max < 5:
            lineas.append(
                f"Multicolinealidad: el VIF máximo es {vif_max:.2f} ({vif.idxmax()}); es bajo, "
                f"así que los coeficientes son estables."
            )
        else:
            lineas.append(
                f"Multicolinealidad: el VIF de {vif.idxmax()} es {vif_max:.2f}; los coeficientes "
                f"de las variables correlacionadas deben leerse con cautela."
            )
 
        desviacion = residuos["desviacion"]
        if abs(residuos["media"]) < 0.05 * desviacion:
            sesgo = "sin sesgo sistemático"
        else:
            sesgo = f"con un sesgo medio de {residuos['media']:+.2f}"
        lineas.append(
            f"Residuos: media {residuos['media']:+.2f} ({sesgo}) y desviación {desviacion:.2f}."
        )
        if residuos["asimetria"] < -0.5:
            lineas.append(
                f"Los residuos tienen asimetría {residuos['asimetria']:.2f}: hay casos con "
                f"{objetivo} real mucho menor que el predicho, que el modelo tiende a sobreestimar."
            )
        elif residuos["asimetria"] > 0.5:
            lineas.append(
                f"Los residuos tienen asimetría {residuos['asimetria']:+.2f}: hay casos con "
                f"{objetivo} real mucho mayor que el predicho, que el modelo tiende a subestimar."
            )
        if residuos["curtosis_exceso"] > 1:
            lineas.append(
                f"Las colas son pesadas (curtosis en exceso {residuos['curtosis_exceso']:.1f}): "
                f"el {residuos['pct_fuera_2_desviaciones']:.1f} % de los residuos supera dos "
                f"desviaciones, más de lo que daría una distribución normal."
            )
        bajo, alto = residuos["residuo_medio_decil_bajo"], residuos["residuo_medio_decil_alto"]
        if abs(bajo) > 0.25 * desviacion or abs(alto) > 0.25 * desviacion:
            lineas.append(
                f"Tendencia hacia la media: en el 10 % de casos con {objetivo} real más bajo "
                f"(hasta {residuos['objetivo_p10_entrenamiento']:.0f}) el residuo medio es "
                f"{bajo:+.1f}, y en el 10 % más alto (desde "
                f"{residuos['objetivo_p90_entrenamiento']:.0f}) es {alto:+.1f}. Un residuo "
                f"negativo significa que el modelo sobreestima; uno positivo, que subestima."
            )
        cuartiles = residuos["desv_por_cuartil"]
        menor, mayor = min(cuartiles.values()), max(cuartiles.values())
        if menor > 0 and mayor / menor > 1.2:
            lineas.append(
                f"La dispersión del error crece con el valor predicho: de {cuartiles['Q1']:.1f} "
                f"(cuartil bajo) a {cuartiles['Q4']:.1f} (cuartil alto). El modelo es menos "
                f"preciso para los valores altos."
            )
        return lineas
 
    @staticmethod
    def limitaciones_regresion(
        objetivo: str,
        filas_originales: int,
        filas_usadas: int,
        variable_principal: str,
        r2_variable_principal: float,
        r2_modelo: float,
        residuos: dict[str, Any],
    ) -> list[str]:
        """Redacta las limitaciones de la regresión con los valores medidos."""
        descartadas = filas_originales - filas_usadas
        limitaciones = [
            f"{objetivo} es un rating posterior a la carrera: el modelo estima el desempeño a "
            f"partir de datos previos, pero no pronostica quién ganará una carrera.",
            f"Se descartaron {descartadas} de {filas_originales} filas "
            f"({descartadas / filas_originales * 100:.1f} %) por tener ratings imputados con la "
            f"media. Los resultados describen a los caballos con ratings registrados y pueden no "
            f"generalizarse al resto.",
            f"Casi todo el poder predictivo viene de {variable_principal}: por sí sola alcanza "
            f"R² = {r2_variable_principal:.3f}, y las demás variables aportan "
            f"{r2_modelo - r2_variable_principal:+.3f}.",
            "El modelo supone una relación lineal y no captura interacciones. Los caballos de una "
            "misma carrera no son independientes; se evitó mezclarlos entre entrenamiento y "
            "prueba, pero no se calcularon intervalos de confianza ni valores p.",
        ]
        if residuos["predicciones_fuera_de_rango"] > 0:
            limitaciones.append(
                f"La regresión lineal no está acotada: {residuos['predicciones_fuera_de_rango']} "
                f"predicciones quedaron fuera del rango observado en entrenamiento "
                f"[{residuos['objetivo_min_entrenamiento']:.0f}, "
                f"{residuos['objetivo_max_entrenamiento']:.0f}]."
            )
        return limitaciones
 