"""EA3 - Interpretación automática, resumen .txt y documento de hallazgos.

Todo el texto se construye a partir de los resultados de la ejecución: no hay
cifras escritas a mano. Si cambian los datos o los parámetros, cambian el
resumen, la interpretación y los hallazgos.
"""
from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

MARCA_INICIO = "<!-- HALLAZGOS_EA3:INICIO -->"
MARCA_FIN = "<!-- HALLAZGOS_EA3:FIN -->"


def _pct(valor: float, decimales: int = 1) -> str:
    return f"{valor * 100:.{decimales}f} %".replace(".", ",")


def _num(valor: float, decimales: int = 3) -> str:
    return f"{valor:.{decimales}f}".replace(".", ",")


class InterpretacionEA3:
    """Convierte las métricas en frases para una persona del negocio."""

    @staticmethod
    def interpretar(res: dict[str, Any]) -> list[str]:
        mt, mtr = res["metricas_prueba"], res["metricas_entrenamiento"]
        base = res["baseline_mercado"]
        k = mt["k_top"]
        lineas = [
            f"En datos que el modelo no vio, ordena a los caballos con ROC-AUC {_num(mt['roc_auc'])} "
            f"(0,5 sería azar) y PR-AUC {_num(mt['pr_auc'])} frente a {_num(mt['pr_auc_azar'])} del azar.",
            f"Con el umbral {_num(mt['umbral'], 2)}, de cada 100 caballos que marca como podio acierta "
            f"{_pct(mt['precision'], 0)} (precisión) y detecta {_pct(mt['recall'], 0)} de los que realmente hacen "
            f"podio (recall); F1 = {_num(mt['f1'])}.",
            f"Matriz de confusión: {mt['tp']:,} aciertos de podio, {mt['fp']:,} falsas alarmas, "
            f"{mt['fn']:,} podios no detectados y {mt['tn']:,} descartes correctos.".replace(",", "."),
            f"Dentro de cada carrera, de los {k} caballos mejor puntuados por el modelo el "
            f"{_pct(mt['precision_top_k_modelo'])} termina en podio; con los {k} favoritos del mercado es "
            f"{_pct(mt['precision_top_k_mercado'])} y al azar {_pct(mt['precision_top_k_azar'])}.",
        ]
        mejora = mt["roc_auc"] - base["roc_auc"]
        if mejora > 0.01:
            lineas.append(
                f"El modelo mejora al mercado solo (ROC-AUC {_num(base['roc_auc'])}) en {_num(mejora)} puntos: "
                "el resto de variables aporta información adicional."
            )
        else:
            lineas.append(
                f"El modelo casi no mejora al mercado solo (ROC-AUC {_num(base['roc_auc'])} frente a "
                f"{_num(mt['roc_auc'])}): el precio ya resume casi toda la información disponible."
            )
        lineas.append(
            f"Entre entrenamiento y prueba la PR-AUC pasa de {_num(mtr['pr_auc'])} a {_num(mt['pr_auc'])}"
            + (" (diferencia pequeña: no hay sobreajuste importante)."
               if mtr["pr_auc"] - mt["pr_auc"] < 0.05 else " (diferencia notable: hay sobreajuste).")
        )
        imp = pd.DataFrame(res["importancias"]).set_index("variable")
        top = imp.head(3)
        lineas.append(
            "Variables que más pesan (por permutación): "
            + ", ".join(f"{v} ({_num(r['peso_relativo_pct'], 1)} %)" for v, r in top.iterrows())
            + "."
        )
        return lineas

    @staticmethod
    def limitaciones(res: dict[str, Any]) -> list[str]:
        mt = res["metricas_prueba"]
        cal = pd.DataFrame(res["calibracion"])
        desviacion = float(cal["diferencia"].abs().max())
        campos = pd.DataFrame(res["desempeno_campo"])
        lineas = [
            "Solo usa información previa a la carrera y disponible en el dataset (2019): no incluye estado del "
            "terreno, distancia de la carrera, forma reciente del caballo ni historial del jinete o del entrenador.",
            f"Un {_pct(res['comparacion']['antes (original)']['OR imputados con la media'] / res['comparacion']['antes (original)']['filas'], 0)} "
            "de los ratings OR venía imputado con la media por el autor del dataset; se devolvieron a nulo y se "
            "rellenaron con la mediana del entrenamiento, por lo que el rating es menos informativo de lo que sería con datos completos.",
            f"La precisión con el umbral {_num(mt['umbral'], 2)} es {_pct(mt['precision'], 0)}: "
            f"el {_pct(1 - mt['precision'], 0)} de los caballos señalados no hace podio. "
            "Una apuesta basada en el modelo sigue teniendo alto riesgo de pérdida.",
            f"Las probabilidades no están perfectamente calibradas: la mayor diferencia entre lo predicho y lo "
            f"real por decil es de {_num(desviacion * 100, 1)} puntos porcentuales (el modelo usa class_weight='balanced', que infla las "
            "probabilidades de la clase positiva). Úsense para ordenar, no como probabilidad literal.",
            "El resultado depende de la partición de carreras elegida (semilla "
            f"{res['semilla']}); con otra partición las métricas variarían ligeramente.",
            "El modelo no considera cuotas finales de pago ni costos de apuesta: no mide rentabilidad, "
            "solo qué tan bien identifica a quienes hacen podio.",
        ]
        if len(campos) > 1:
            peor = campos.loc[campos["roc_auc"].idxmin(), "corredores"]
            lineas.append(f"El desempeño cambia según el tamaño del campo; es más débil en carreras de {peor} corredores.")
        return lineas

    @staticmethod
    def recomendaciones(res: dict[str, Any]) -> list[str]:
        mt, base = res["metricas_prueba"], res["baseline_mercado"]
        k = mt["k_top"]
        mejora_top = mt["precision_top_k_modelo"] - mt["precision_top_k_mercado"]
        lineas = [
            f"Usar el modelo para **ordenar** a los caballos dentro de cada carrera y revisar primero a los {k} "
            "mejor puntuados, en lugar de decidir con un umbral fijo de probabilidad.",
        ]
        if mejora_top > 0.005:
            lineas.append(
                f"Priorizar al modelo sobre la regla 'apostar a los {k} favoritos': mejora el acierto por carrera en "
                f"{_num(mejora_top * 100, 1)} puntos porcentuales ({_pct(mt['precision_top_k_modelo'])} vs {_pct(mt['precision_top_k_mercado'])})."
            )
        else:
            lineas.append(
                f"No abandonar la regla simple de los {k} favoritos del mercado solo por este modelo: su ventaja por "
                f"carrera es mínima ({_pct(mt['precision_top_k_modelo'])} vs {_pct(mt['precision_top_k_mercado'])}). "
                "Úsese como segunda opinión."
            )
        imp = pd.DataFrame(res["importancias"]).set_index("variable")
        no_mercado = imp[~imp.index.isin(["decimalPrice", "prob_mercado_norm", "rank_mercado"])]
        if not no_mercado.empty:
            lineas.append(
                f"Fuera del mercado, vigilar primero **{no_mercado.index[0]}**, la variable no-mercado con más peso en el modelo."
            )
        lineas.append(
            "Antes de arriesgar dinero real, probar el modelo con carreras de otra temporada y medir la rentabilidad "
            "con cuotas reales; este análisis solo mide aciertos de podio."
        )
        lineas.append(
            "Mejorar los datos: completar el rating OR sin imputar y añadir terreno, distancia de la carrera y forma reciente."
        )
        return lineas

    @staticmethod
    def respuesta_negocio(res: dict[str, Any]) -> str:
        mt, base = res["metricas_prueba"], res["baseline_mercado"]
        imp = pd.DataFrame(res["importancias"]).set_index("variable")
        top = list(imp.index[:3])
        k = mt["k_top"]
        return (
            f"Sí es posible anticipar parcialmente qué caballos hacen podio. Los factores que más lo determinan son "
            f"{', '.join(top)}. El modelo acierta {_pct(mt['precision_top_k_modelo'])} de los {k} caballos que "
            f"propone por carrera (azar: {_pct(mt['precision_top_k_azar'])}; favoritos del mercado: "
            f"{_pct(mt['precision_top_k_mercado'])}), por lo que sirve como apoyo para priorizar apuestas de colocado, "
            "pero no como garantía de acierto."
        )


class ReporteEA3:
    """Genera el resumen de ejecución (.txt) y el documento de hallazgos (.md)."""

    ANCHO = 100

    def __init__(self, directorio: str | Path = "resultados_ea3") -> None:
        self.directorio = Path(directorio)

    # ------------------------------------------------------------------ #
    # Resumen .txt
    # ------------------------------------------------------------------ #
    def generar_resumen(self, res: dict[str, Any]) -> Path:
        self.directorio.mkdir(parents=True, exist_ok=True)
        destino = self.directorio / "resumen_ea3.txt"
        destino.write_text(self.construir_resumen(res), encoding="utf-8")
        return destino

    def construir_resumen(self, res: dict[str, Any]) -> str:
        ds, cfg = res["dataset"], res["configuracion"]
        mt, mtr, base = res["metricas_prueba"], res["metricas_entrenamiento"], res["baseline_mercado"]
        partes = [
            self._titulo("RESUMEN DE EJECUCIÓN - EA3: APRENDIZAJE SUPERVISADO (RANDOM FOREST)"),
            f"Fecha de ejecución: {res['fecha_ejecucion']}",
            "",
            self._titulo("1. PREGUNTA DE NEGOCIO"),
            *self._campo("Pregunta", res["pregunta"]["pregunta"]),
            *self._campo("Quién se beneficia", res["pregunta"]["beneficiario"]),
            *self._campo("Decisión que apoya", res["pregunta"]["decision"]),
            "",
            self._titulo("2. DATASET"),
            f"  Identificador : {ds['dataset']}",
            f"  Enlace        : {ds['enlace']}",
            f"  Configuración : {ds['config']} | split: {ds['split']} | archivo: {ds['archivo_origen']}",
            f"  Licencia      : {ds['licencia']}",
            f"  DataFrame     : {ds['filas']:,} filas x {ds['columnas']} columnas".replace(",", "."),
            "",
            self._titulo("3. PREPROCESAMIENTO APLICADO"),
            self._tabla(
                ["operación", "filas antes", "filas después", "detalle"],
                [[o["operacion"], f"{o['filas_antes']:,}".replace(",", "."), f"{o['filas_despues']:,}".replace(",", "."), o["detalle"]]
                 for o in res["preprocesamiento"]],
            ),
            "  Comparación antes / después:",
            self._tabla(
                ["indicador", "antes (original)", "después (preparado)"],
                [[k, str(v), str(res["comparacion"]["después (preparado)"][k])]
                 for k, v in res["comparacion"]["antes (original)"].items()],
            ),
            *self._campo("Escalado", "No se escala: el Random Forest divide por umbrales y es invariante a la escala de las variables."),
            "",
            self._titulo("4. MODELO Y CONFIGURACIÓN DEL EXPERIMENTO"),
            f"  Tipo de aprendizaje : Supervisado - clasificación binaria",
            f"  Algoritmo           : {res['algoritmo']}",
            f"  Variable objetivo   : {res['objetivo']} (clases: " + ", ".join(f"{k} = {v}" for k, v in res["clases"].items()) + ")",
            f"  Variables           : {', '.join(res['predictoras'])}",
            f"  Semilla aleatoria   : {res['semilla']}",
            f"  División            : {cfg['division']} - entrenamiento {cfg['filas_entrenamiento']:,} filas "
            f"({cfg['carreras_entrenamiento']:,} carreras), prueba {cfg['filas_prueba']:,} filas "
            f"({cfg['carreras_prueba']:,} carreras), carreras compartidas = {cfg['carreras_compartidas']}".replace(",", "."),
            f"  Umbral de decisión  : {mt['umbral']}",
            f"  Hiperparámetros elegidos (validación cruzada {res['n_splits']} pliegues por carrera, criterio PR-AUC):",
            "    " + ", ".join(f"{k}={v}" for k, v in res["hiperparametros_elegidos"].items()),
            "  Otros parámetros (valores del modelo entrenado):",
            "    " + ", ".join(f"{k}={v}" for k, v in res["hiperparametros_completos"].items()
                              if k in ("class_weight", "criterion", "max_features", "bootstrap", "n_jobs")),
            "",
            "  Evidencia de la validación cruzada (profundidad y tamaño de hoja):",
            self._tabla(
                ["max_depth", "min_samples_leaf", "PR-AUC val.", "PR-AUC entren.", "ROC-AUC val."],
                [[r["max_depth"], r["min_samples_leaf"], _num(r["pr_auc_cv"], 4), _num(r["pr_auc_entrenamiento"], 4), _num(r["roc_auc_cv"], 4)]
                 for r in res["cv_profundidad"]],
            ),
            "  Evidencia de la validación cruzada (número de árboles):",
            self._tabla(
                ["n_estimators", "PR-AUC val.", "ROC-AUC val."],
                [[r["n_estimators"], _num(r["pr_auc_cv"], 4), _num(r["roc_auc_cv"], 4)] for r in res["cv_arboles"]],
            ),
            "",
            self._titulo("5. MÉTRICAS"),
            self._tabla(
                ["métrica", "entrenamiento", "prueba"],
                [[n, _num(mtr[c]), _num(mt[c])] for n, c in (
                    ("Exactitud", "exactitud"), ("Exactitud balanceada", "exactitud_balanceada"),
                    ("Precisión", "precision"), ("Recall", "recall"), ("F1", "f1"),
                    ("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"))],
            ),
            f"  Referencias en prueba: exactitud de responder siempre 'fuera del podio' = {_num(mt['exactitud_baseline'])}; "
            f"PR-AUC del azar = {_num(mt['pr_auc_azar'])}; ROC-AUC solo con el mercado = {_num(base['roc_auc'])}; "
            f"PR-AUC solo con el mercado = {_num(base['pr_auc'])}.",
            f"  Matriz de confusión (prueba) [filas: real; columnas: predicho 0/1]: {mt['matriz_confusion']}",
            f"  Top-{mt['k_top']} por carrera (prueba, {mt['carreras_top_k']:,} carreras): modelo {_pct(mt['precision_top_k_modelo'])}"
            f" | favoritos del mercado {_pct(mt['precision_top_k_mercado'])} | azar {_pct(mt['precision_top_k_azar'])}".replace(",", "."),
            "",
            "  Importancia de variables (permutación sobre la prueba):",
            self._tabla(
                ["variable", "caída PR-AUC", "peso relativo %", "impureza"],
                [[r["variable"], _num(r["importancia_permutacion"], 4), _num(r["peso_relativo_pct"], 1), _num(r["importancia_impureza"], 4)]
                 for r in res["importancias"]],
            ),
            "",
            self._titulo("6. INTERPRETACIÓN"),
            *self._vinetas(res["interpretacion"]),
            "",
            self._titulo("7. LIMITACIONES"),
            *self._vinetas(res["limitaciones"]),
            "",
            self._titulo("8. RESPUESTA A LA PREGUNTA DE NEGOCIO Y RECOMENDACIONES"),
            *self._campo("Respuesta", res["respuesta_negocio"]),
            *self._vinetas([r.replace("**", "") for r in res["recomendaciones"]]),
            "",
            "Gráficas generadas (carpeta graficas/): " + ", ".join(sorted(Path(v).name for v in res["graficas"].values())),
            "",
        ]
        return "\n".join(partes)

    # ------------------------------------------------------------------ #
    # Documento de hallazgos .md
    # ------------------------------------------------------------------ #
    def construir_hallazgos(self, res: dict[str, Any], prefijo_imagenes: str = "graficas/") -> str:
        mt, base = res["metricas_prueba"], res["baseline_mercado"]
        g = {k: f"{prefijo_imagenes}{Path(v).name}" for k, v in res["graficas"].items()}
        imp = pd.DataFrame(res["importancias"])
        tabla_imp = "| Variable | Peso relativo (%) | Caída de PR-AUC al desordenarla |\n|---|---:|---:|\n" + "\n".join(
            f"| `{r['variable']}` | {_num(r['peso_relativo_pct'], 1)} | {_num(r['importancia_permutacion'], 4)} |" for _, r in imp.iterrows()
        )
        tabla_metricas = (
            "| Métrica | Entrenamiento | Prueba |\n|---|---:|---:|\n"
            + "\n".join(
                f"| {n} | {_num(res['metricas_entrenamiento'][c])} | {_num(mt[c])} |"
                for n, c in (("Precisión", "precision"), ("Recall", "recall"), ("F1", "f1"), ("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"))
            )
        )
        md = [
            f"_Generado automáticamente por la ejecución del {res['fecha_ejecucion']}. No editar a mano: se reescribe en cada ejecución._",
            "",
            "### Respuesta a la pregunta de negocio",
            "",
            f"> **{res['pregunta']['pregunta']}**",
            "",
            res["respuesta_negocio"],
            "",
            "### Hallazgos principales",
            "",
            *[f"- {l}" for l in res["interpretacion"]],
            "",
            tabla_metricas,
            "",
            f"![Top-k por carrera]({g['top_k']})",
            "",
            f"![Curvas ROC y PR]({g['curvas']})",
            "",
            f"![Matriz de confusión]({g['matriz_confusion']})",
            "",
            "**Qué variables pesan:**",
            "",
            tabla_imp,
            "",
            f"![Importancia de variables]({g['importancias']})",
            "",
            "### Qué tan confiables son los resultados y sus limitaciones",
            "",
            f"![Calibración]({g['calibracion']})",
            "",
            *[f"- {l}" for l in res["limitaciones"]],
            "",
            "### Recomendaciones",
            "",
            *[f"{i}. {l}" for i, l in enumerate(res["recomendaciones"], start=1)],
            "",
            "### Evidencia del preprocesamiento y del modelo",
            "",
            f"![Distribución del objetivo]({g['distribucion_clases']})",
            "",
            f"![OR antes y después]({g['antes_despues_or']})",
            "",
            f"![Validación cruzada]({g['hiperparametros']})",
            "",
        ]
        return "\n".join(md)

    def generar_hallazgos(self, res: dict[str, Any]) -> Path:
        self.directorio.mkdir(parents=True, exist_ok=True)
        destino = self.directorio / "hallazgos_ea3.md"
        destino.write_text(self.construir_hallazgos(res, "graficas/"), encoding="utf-8")
        return destino

    def actualizar_readme(self, ruta_readme: str | Path, res: dict[str, Any], prefijo_imagenes: str) -> Path:
        """Inserta el documento de hallazgos entre las marcas del README."""
        ruta = Path(ruta_readme)
        contenido = ruta.read_text(encoding="utf-8")
        bloque = f"{MARCA_INICIO}\n{self.construir_hallazgos(res, prefijo_imagenes)}\n{MARCA_FIN}"
        if MARCA_INICIO in contenido and MARCA_FIN in contenido:
            antes = contenido.split(MARCA_INICIO)[0]
            despues = contenido.split(MARCA_FIN, 1)[1]
            nuevo = f"{antes}{bloque}{despues}"
        else:
            nuevo = contenido.rstrip() + "\n\n## Documento de hallazgos EA3\n\n" + bloque + "\n"
        ruta.write_text(nuevo, encoding="utf-8")
        return ruta

    # ------------------------------------------------------------------ #
    # Formato
    # ------------------------------------------------------------------ #
    @staticmethod
    def _titulo(texto: str) -> str:
        return "=" * ReporteEA3.ANCHO + f"\n{texto}\n" + "=" * ReporteEA3.ANCHO

    @staticmethod
    def _campo(etiqueta: str, texto: str) -> list[str]:
        return textwrap.wrap(f"{etiqueta}: {texto}", ReporteEA3.ANCHO, initial_indent="  ", subsequent_indent="    ")

    @staticmethod
    def _vinetas(items: list[str]) -> list[str]:
        salida: list[str] = []
        for item in items:
            salida.extend(textwrap.wrap(item, ReporteEA3.ANCHO, initial_indent="  - ", subsequent_indent="    "))
        return salida

    @staticmethod
    def _tabla(encabezado: list[str], filas: list[list[Any]]) -> str:
        filas = [[str(c) for c in f] for f in filas]
        anchos = [max(len(encabezado[i]), *(len(f[i]) for f in filas)) if filas else len(encabezado[i]) for i in range(len(encabezado))]
        # Limita la última columna (texto largo) para no desbordar el ancho.
        linea = lambda f: "  " + " | ".join(c.ljust(anchos[i]) if i < len(f) - 1 else c for i, c in enumerate(f))
        sep = "  " + "-+-".join("-" * (anchos[i] if i < len(anchos) - 1 else min(anchos[i], 40)) for i in range(len(anchos)))
        return "\n".join([linea(encabezado), sep, *(linea(f) for f in filas)])


class ConsolidadorEA3:
    """Reúne en un solo diccionario todo lo que produjo la ejecución."""

    @staticmethod
    def consolidar(
        *,
        datos,
        df_original: pd.DataFrame,
        prep,
        comparacion: pd.DataFrame,
        distribucion: pd.DataFrame,
        modelo,
        busqueda: dict[str, pd.DataFrame],
        n_splits: int,
        metricas_train: dict[str, Any],
        metricas_test: dict[str, Any],
        baseline: dict[str, float],
        calibracion: pd.DataFrame,
        desempeno_campo: pd.DataFrame,
        sensibilidad: pd.DataFrame,
        importancias: pd.DataFrame,
        graficas: dict[str, Path],
        fecha_ejecucion: str | None = None,
    ) -> dict[str, Any]:
        """Devuelve el diccionario de resultados (sin interpretación todavía)."""
        info = datos.informacion()
        info["columnas"] = int(df_original.shape[1])
        info["filas"] = int(df_original.shape[0])
        return {
            "fecha_ejecucion": fecha_ejecucion or datetime.now().isoformat(timespec="seconds"),
            "pregunta": {
                "pregunta": datos.PREGUNTA_NEGOCIO,
                "beneficiario": datos.QUIEN_SE_BENEFICIA,
                "decision": datos.DECISION_QUE_APOYA,
            },
            "dataset": info,
            "preprocesamiento": prep.resumen_operaciones().to_dict("records"),
            "comparacion": comparacion.to_dict(),
            "distribucion_clases": distribucion.reset_index().to_dict("records"),
            "tipo_aprendizaje": "Supervisado - clasificación binaria",
            "algoritmo": "Random Forest (RandomForestClassifier de scikit-learn)",
            "objetivo": datos.OBJETIVO,
            "clases": {str(k): v for k, v in datos.CLASES.items()},
            "predictoras": list(prep.columnas_predictoras),
            "semilla": prep.random_state,
            "n_splits": n_splits,
            "configuracion": {**prep.info_division, "division": "por carrera (GroupShuffleSplit)"},
            "hiperparametros_elegidos": dict(modelo.parametros),
            "hiperparametros_completos": modelo.hiperparametros_completos(),
            "cv_profundidad": busqueda["profundidad_hoja"].to_dict("records"),
            "cv_arboles": busqueda["n_arboles"].to_dict("records"),
            "metricas_entrenamiento": metricas_train,
            "metricas_prueba": metricas_test,
            "baseline_mercado": baseline,
            "calibracion": calibracion.reset_index().to_dict("records"),
            "desempeno_campo": desempeno_campo.reset_index().to_dict("records"),
            "sensibilidad_umbral": sensibilidad.reset_index().to_dict("records"),
            "importancias": importancias.reset_index(names="variable").to_dict("records"),
            "graficas": {k: str(v) for k, v in graficas.items()},
        }

    @staticmethod
    def agregar_interpretacion(res: dict[str, Any]) -> dict[str, Any]:
        """Añade interpretación, limitaciones, recomendaciones y respuesta de negocio."""
        res["interpretacion"] = InterpretacionEA3.interpretar(res)
        res["limitaciones"] = InterpretacionEA3.limitaciones(res)
        res["recomendaciones"] = InterpretacionEA3.recomendaciones(res)
        res["respuesta_negocio"] = InterpretacionEA3.respuesta_negocio(res)
        return res
