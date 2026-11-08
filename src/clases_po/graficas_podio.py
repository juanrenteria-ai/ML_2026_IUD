"""EA3 - Gráficas generadas por código (se guardan como PNG)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


class GraficasEA3:
    """Genera y guarda las gráficas del proyecto en ``directorio``.

    Cada método dibuja la figura, la guarda como PNG y devuelve la ruta, de
    modo que el notebook, el resumen y el README usan exactamente las mismas
    imágenes producidas por la ejecución.
    """

    def __init__(self, directorio: str | Path = "resultados_ea3/graficas", mostrar: bool = True) -> None:
        self.directorio = Path(directorio)
        self.directorio.mkdir(parents=True, exist_ok=True)
        self.mostrar = mostrar

    def _cerrar(self, fig: plt.Figure, nombre: str) -> Path:
        fig.tight_layout()
        ruta = self.directorio / nombre
        fig.savefig(ruta, dpi=130, bbox_inches="tight")
        if self.mostrar:
            plt.show()
        plt.close(fig)
        return ruta

    def distribucion_clases(self, distribucion: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        ax.bar(distribucion["clase"], distribucion["filas"], color=["tab:gray", "tab:green"])
        for i, (filas, pct) in enumerate(zip(distribucion["filas"], distribucion["porcentaje"])):
            ax.text(i, filas, f"{pct:.1f} %", ha="center", va="bottom")
        ax.set_ylim(0, distribucion["filas"].max() * 1.12)
        ax.set_ylabel("Filas")
        ax.set_title("Distribución del objetivo (res_place)")
        return self._cerrar(fig, "01_distribucion_clases.png")

    def antes_despues_or(self, or_original: pd.Series, or_restaurado: pd.Series) -> Path:
        fig, ejes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
        rango = (float(or_original.min()), float(or_original.max()))
        ejes[0].hist(or_original.dropna(), bins=60, range=rango, color="tab:orange")
        ejes[0].set_title("OR original (pico = valor imputado con la media)")
        ejes[1].hist(or_restaurado.dropna(), bins=60, range=rango, color="tab:blue")
        ejes[1].set_title("OR tras devolver los imputados a NaN")
        for ax in ejes:
            ax.set_xlabel("OR")
        ejes[0].set_ylabel("Filas")
        return self._cerrar(fig, "02_antes_despues_OR.png")

    def hiperparametros(self, tabla_profundidad: pd.DataFrame, tabla_arboles: pd.DataFrame) -> Path:
        fig, ejes = plt.subplots(1, 2, figsize=(11.5, 4))
        for hoja, grupo in tabla_profundidad.groupby("min_samples_leaf"):
            grupo = grupo.sort_values("max_depth")
            ejes[0].errorbar(
                grupo["max_depth"], grupo["pr_auc_cv"], yerr=grupo["pr_auc_cv_desv"],
                marker="o", capsize=3, label=f"min_samples_leaf={hoja} (validación)",
            )
            ejes[0].plot(grupo["max_depth"], grupo["pr_auc_entrenamiento"], "--", alpha=0.5,
                         label=f"min_samples_leaf={hoja} (entrenamiento)")
        ejes[0].set_xlabel("max_depth")
        ejes[0].set_ylabel("PR-AUC (media de 5 pliegues)")
        ejes[0].set_title("Profundidad y tamaño de hoja")
        ejes[0].legend(fontsize=7)
        ejes[1].errorbar(tabla_arboles["n_estimators"], tabla_arboles["pr_auc_cv"],
                         yerr=tabla_arboles["pr_auc_cv_desv"], marker="o", capsize=3, color="tab:green")
        ejes[1].set_xlabel("n_estimators")
        ejes[1].set_ylabel("PR-AUC (validación)")
        ejes[1].set_title("Número de árboles")
        return self._cerrar(fig, "03_validacion_cruzada.png")

    def matriz_confusion(self, matriz: list[list[int]], umbral: float) -> Path:
        m = np.array(matriz)
        fig, ax = plt.subplots(figsize=(5, 4.2))
        ax.imshow(m, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{m[i, j]:,}", ha="center", va="center", fontsize=13,
                        color="white" if m[i, j] > m.max() / 2 else "black")
        etiquetas = ["Fuera del podio", "Podio"]
        ax.set_xticks([0, 1], etiquetas)
        ax.set_yticks([0, 1], etiquetas)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.set_title(f"Matriz de confusión (prueba, umbral {umbral})")
        return self._cerrar(fig, "04_matriz_confusion.png")

    def curvas(self, y: pd.Series, prob_modelo: np.ndarray, prob_mercado: np.ndarray) -> Path:
        y = np.asarray(y, dtype=int)
        fig, ejes = plt.subplots(1, 2, figsize=(11, 4.2))
        for etiqueta, p, estilo in (("Random Forest", prob_modelo, "-"), ("Solo mercado", prob_mercado, "--")):
            fpr, tpr, _ = roc_curve(y, p)
            ejes[0].plot(fpr, tpr, estilo, label=etiqueta)
            prec, rec, _ = precision_recall_curve(y, p)
            ejes[1].plot(rec, prec, estilo, label=etiqueta)
        ejes[0].plot([0, 1], [0, 1], ":", color="gray", label="Azar")
        ejes[0].set_xlabel("Tasa de falsos positivos")
        ejes[0].set_ylabel("Recall")
        ejes[0].set_title("Curva ROC (prueba)")
        ejes[0].legend()
        ejes[1].axhline(y.mean(), ls=":", color="gray", label="Azar")
        ejes[1].set_xlabel("Recall")
        ejes[1].set_ylabel("Precisión")
        ejes[1].set_title("Curva precisión-recall (prueba)")
        ejes[1].legend()
        return self._cerrar(fig, "05_curvas_roc_pr.png")

    def importancias(self, tabla: pd.DataFrame) -> Path:
        t = tabla.sort_values("importancia_permutacion")
        fig, ejes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
        ejes[0].barh(t.index, t["importancia_permutacion"], xerr=t["permutacion_desv"], color="tab:blue")
        ejes[0].set_title("Por permutación (prueba): caída de PR-AUC")
        ejes[1].barh(t.index, t["importancia_impureza"], color="tab:green")
        ejes[1].set_title("Por impureza (entrenamiento)")
        return self._cerrar(fig, "06_importancia_variables.png")

    def calibracion(self, tabla: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        ax.plot(tabla["prob_predicha"], tabla["tasa_real"], "o-", label="Random Forest")
        tope = max(tabla["prob_predicha"].max(), tabla["tasa_real"].max()) * 1.05
        ax.plot([0, tope], [0, tope], ":", color="gray", label="Calibración perfecta")
        ax.set_xlabel("Probabilidad predicha (media por decil)")
        ax.set_ylabel("Tasa real de podio")
        ax.set_title("Calibración (prueba)")
        ax.legend()
        return self._cerrar(fig, "07_calibracion.png")

    def top_k(self, metricas: dict) -> Path:
        k = metricas["k_top"]
        valores = [metricas["precision_top_k_azar"], metricas["precision_top_k_mercado"], metricas["precision_top_k_modelo"]]
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.bar(["Azar", f"{k} favoritos\ndel mercado", f"{k} mejores\ndel modelo"], [v * 100 for v in valores],
               color=["tab:gray", "tab:orange", "tab:green"])
        for i, v in enumerate(valores):
            ax.text(i, v * 100, f"{v * 100:.1f} %", ha="center", va="bottom")
        ax.set_ylim(0, max(valores) * 100 * 1.2)
        ax.set_ylabel(f"% de los {k} elegidos por carrera que hacen podio")
        ax.set_title("Utilidad para la decisión (prueba)")
        return self._cerrar(fig, "08_top_k_por_carrera.png")
