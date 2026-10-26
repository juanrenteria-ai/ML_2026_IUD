"""Generación automática del resumen en texto de los dos modelos."""
from __future__ import annotations
 
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any
 
ANCHO = 92
 
 
class ReporteModelos:
    """Construye el resumen ``.txt`` a partir de los resultados de ambos modelos.
 
    Todo el contenido sale de los diccionarios guardados por los notebooks
    (``resultados/regresion.json`` y ``resultados/clasificacion.json``); no hay
    cifras escritas a mano, así que el reporte refleja la última ejecución.
    """
 
    def __init__(self, directorio: str | Path = "resultados") -> None:
        self.directorio = Path(directorio)
 
    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #
    def generar(self, conjunto: dict[str, Any]) -> Path:
        """Escribe el reporte y devuelve su ruta.
 
        Raises:
            KeyError: Si ``conjunto`` no trae las claves esperadas.
        """
        for clave in ("regresion", "clasificacion", "conclusiones"):
            if clave not in conjunto:
                raise KeyError(f"Falta '{clave}' en los resultados conjuntos.")
        texto = self.construir(conjunto)
        self.directorio.mkdir(parents=True, exist_ok=True)
        marca = datetime.now().strftime("%Y%m%d_%H%M%S")
        destino = self.directorio / f"resumen_modelos_{marca}.txt"
        destino.write_text(texto, encoding="utf-8")
        return destino
 
    def construir(self, conjunto: dict[str, Any]) -> str:
        """Devuelve el texto completo del reporte."""
        reg, clf = conjunto["regresion"], conjunto["clasificacion"]
        partes = [
            self._titulo("RESUMEN DE MODELOS - EA2"),
            self._cabecera(reg, clf),
            self._seccion_dataset(reg, clf),
            self._seccion_preprocesamiento(reg, clf),
            self._seccion_regresion(reg),
            self._seccion_clasificacion(clf),
            self._seccion_conclusiones(conjunto["conclusiones"], reg, clf),
        ]
        return "\n".join(partes).rstrip() + "\n"
 
    # ------------------------------------------------------------------ #
    # Secciones
    # ------------------------------------------------------------------ #
    def _cabecera(self, reg: dict, clf: dict) -> str:
        return "\n".join(
            [
                f"Reporte generado el:            {datetime.now().isoformat(timespec='seconds')}",
                f"Regresión ejecutada el:         {reg.get('fecha_ejecucion', 'desconocida')}",
                f"Clasificación ejecutada el:     {clf.get('fecha_ejecucion', 'desconocida')}",
                "",
            ]
        )
 
    def _seccion_dataset(self, reg: dict, clf: dict) -> str:
        info = clf.get("dataset") or reg.get("dataset") or {}
        lineas = [
            f"Dataset:            {info.get('dataset', 'mayacheruty/Horse-Race-Prediction-EDA')}",
            "Origen:             Hugging Face (descarga automática al ejecutar)",
            f"Split:              {info.get('split', 'train')}",
            f"Filas originales:   {clf['filas_originales']}",
            f"Columnas:           {info.get('columnas', 'n/d')}",
            "Contenido:          una fila por caballo en una carrera (edad, peso, cuota de",
            "                    mercado, ratings, resultado).",
        ]
        return self._bloque("1. DATASET", lineas)
 
    def _seccion_preprocesamiento(self, reg: dict, clf: dict) -> str:
        lineas: list[str] = []
        for nombre, res in (("Regresión (objetivo RPR)", reg), ("Clasificación (objetivo res_win)", clf)):
            lineas.append(nombre)
            pasos = res.get("preprocesamiento", [])
            if not pasos:
                lineas.append("  (sin bitácora de preprocesamiento en los resultados)")
            for i, paso in enumerate(pasos, 1):
                cambio = f"{paso['filas_antes']} -> {paso['filas_despues']} filas"
                lineas.extend(self._envolver(f"{i}. {paso['operacion']}: {paso['detalle']} ({cambio})", "  ", "     "))
            lineas.append(
                f"  Filas usadas: {res['filas_usadas']} de {res['filas_originales']} "
                f"({res['filas_usadas'] / res['filas_originales'] * 100:.1f} %)."
            )
            lineas.append("")
        return self._bloque("2. PREPROCESAMIENTO", lineas)
 
    def _seccion_regresion(self, reg: dict) -> str:
        mt, mr, mb = reg["metricas_entrenamiento"], reg["metricas_prueba"], reg["metricas_baseline"]
        cfg = reg["configuracion"]
        lineas = [
            f"Objetivo:      {reg['objetivo']} (numérico): rating de desempeño del caballo en la carrera.",
            f"Variables:     {', '.join(reg['predictoras'])}",
            f"Algoritmo:     {reg['algoritmo']}",
            *self._campo("Hiperparámetros:", self._params(reg["hiperparametros"])),
            *self._campo("Configuración:", f"{cfg['division']}; {cfg['filas_entrenamiento']} filas de entrenamiento "
            f"({cfg['carreras_entrenamiento']} carreras) y {cfg['filas_prueba']} de prueba "
            f"({cfg['carreras_prueba']} carreras); semilla {cfg['random_state']}; {cfg['escalado']}."),
            "",
            "Coeficientes (ordenados por peso relativo):",
            self._tabla(
                ["variable", "coef. estandarizado", "coef. unidad original", "peso %"],
                [
                    [c["variable"], f"{c['coef_estandarizado']:+.4f}", f"{c['coef_unidad_original']:+.4f}",
                     f"{c['peso_relativo_pct']:.1f}"]
                    for c in reg["coeficientes"]
                ],
            ),
            f"Intercepto (unidades originales): {reg['intercepto_original']:.4f}",
            "",
            "Métricas:",
            self._tabla(
                ["métrica", "entrenamiento", "prueba", "baseline (la media)"],
                [
                    [k.upper(), f"{mt[k]:.3f}", f"{mr[k]:.3f}", f"{abs(mb[k]) if k == 'r2' else mb[k]:.3f}"]
                    for k in ("r2", "mae", "rmse")
                ],
            ),
            "",
            "Interpretación:",
            *self._vinetas(reg["interpretacion"]),
            "",
            "Limitaciones:",
            *self._vinetas(reg["limitaciones"]),
        ]
        return self._bloque("3. MODELO DE REGRESIÓN LINEAL", lineas)
 
    def _seccion_clasificacion(self, clf: dict) -> str:
        mt, mc = clf["metricas_entrenamiento"], clf["metricas_prueba"]
        cfg = clf["configuracion"]
        matriz = clf["metricas_prueba"]["matriz_confusion"]
        distribucion = ", ".join(
            f"{d['clase']} = {d['filas']} ({d['porcentaje']:.2f} %)" for d in clf["distribucion_clases"]
        )
        claves = [
            ("exactitud", "exactitud"), ("exactitud_baseline", "exactitud (siempre 'No ganó')"),
            ("exactitud_balanceada", "exactitud balanceada"), ("precision", "precisión"),
            ("recall", "recall"), ("f1", "F1"), ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"),
        ]
        lineas = [
            f"Objetivo:      {clf['objetivo']} (binario): 1 = el caballo ganó la carrera, 0 = no ganó.",
            f"Clases:        {distribucion}",
            f"Variables:     {', '.join(clf['predictoras'])}",
            f"Algoritmo:     {clf['algoritmo']}",
            *self._campo("Hiperparámetros:", self._params(clf["hiperparametros"])),
            *self._campo("Configuración:", f"{cfg['division']}; {cfg['filas_entrenamiento']} filas de entrenamiento "
            f"({cfg['carreras_entrenamiento']} carreras) y {cfg['filas_prueba']} de prueba "
            f"({cfg['carreras_prueba']} carreras); semilla {cfg['random_state']}; umbral {cfg['umbral']}; "
            f"{cfg['escalado']}."),
            "",
            "Comparación de algoritmos (validación cruzada por carrera, solo entrenamiento):",
            self._tabla(
                ["algoritmo", "precisión", "recall", "F1", "ROC-AUC", "PR-AUC"],
                [
                    [a["algoritmo"], *(f"{a[k]:.3f}" for k in ("precision", "recall", "f1", "roc_auc", "pr_auc"))]
                    for a in clf["comparacion_algoritmos"]
                ],
            ),
            "",
            "Coeficientes (ordenados por peso relativo):",
            self._tabla(
                ["variable", "coef. estandarizado", "odds ratio / desv.", "peso %"],
                [
                    [c["variable"], f"{c['coef_estandarizado']:+.4f}", f"{c['odds_ratio_por_desv']:.3f}",
                     f"{c['peso_relativo_pct']:.1f}"]
                    for c in clf["coeficientes"]
                ],
            ),
            "",
            "Métricas:",
            self._tabla(
                ["métrica", "entrenamiento", "prueba"],
                [[nombre, f"{mt[k]:.3f}", f"{mc[k]:.3f}"] for k, nombre in claves],
            ),
            "",
            "Matriz de confusión (prueba; filas = real, columnas = predicho):",
            self._tabla(
                ["", "pred. No ganó", "pred. Ganó"],
                [["real No ganó", f"{matriz[0][0]}", f"{matriz[0][1]}"],
                 ["real Ganó", f"{matriz[1][0]}", f"{matriz[1][1]}"]],
            ),
            "",
            "Interpretación:",
            *self._vinetas(clf["interpretacion"]),
            "",
            "Limitaciones:",
            *self._vinetas(clf["limitaciones"]),
        ]
        return self._bloque("4. MODELO DE CLASIFICACIÓN", lineas)
 
    def _seccion_conclusiones(self, conclusiones: list[str], reg: dict, clf: dict) -> str:
        lineas = [
            *self._vinetas(conclusiones),
            "",
            "Resumen de métricas principales (prueba):",
            self._tabla(
                ["modelo", "métrica principal", "referencia"],
                [
                    ["Regresión lineal", f"R2 = {reg['metricas_prueba']['r2']:.3f}",
                     f"MAE {reg['metricas_prueba']['mae']:.2f} vs {reg['metricas_baseline']['mae']:.2f} (media)"],
                    ["Clasificación", f"F1 = {clf['metricas_prueba']['f1']:.3f}",
                     f"PR-AUC {clf['metricas_prueba']['pr_auc']:.3f} vs {clf['metricas_prueba']['pr_auc_azar']:.3f} (azar)"],
                ],
            ),
        ]
        return self._bloque("5. CONCLUSIONES Y LIMITACIONES GENERALES", lineas)
 
    # ------------------------------------------------------------------ #
    # Utilidades de formato
    # ------------------------------------------------------------------ #
    @staticmethod
    def _titulo(texto: str) -> str:
        return "=" * ANCHO + f"\n{texto}\n" + "=" * ANCHO + "\n"
 
    @staticmethod
    def _bloque(titulo: str, lineas: list[str]) -> str:
        return "-" * ANCHO + f"\n{titulo}\n" + "-" * ANCHO + "\n" + "\n".join(lineas) + "\n"
 
    @staticmethod
    def _envolver(texto: str, sangria: str, continuacion: str) -> list[str]:
        return textwrap.wrap(texto, ANCHO, initial_indent=sangria, subsequent_indent=continuacion) or [sangria]
 
    @staticmethod
    def _campo(etiqueta: str, texto: str) -> list[str]:
        """Una línea 'etiqueta valor' con el valor ajustado al ancho del reporte."""
        sangria = " " * 17
        return textwrap.wrap(
            texto, ANCHO, initial_indent=etiqueta.ljust(17), subsequent_indent=sangria,
            break_on_hyphens=False,
        )
 
    def _vinetas(self, items: list[str]) -> list[str]:
        salida: list[str] = []
        for item in items:
            salida.extend(self._envolver(item, "  - ", "    "))
        return salida
 
    @staticmethod
    def _params(parametros: dict[str, Any]) -> str:
        return ", ".join(f"{k}={v}" for k, v in parametros.items())
 
    @staticmethod
    def _tabla(encabezado: list[str], filas: list[list[str]]) -> str:
        columnas = list(zip(encabezado, *filas))
        anchos = [max(len(str(celda)) for celda in col) for col in columnas]
 
        def linea(valores: list[str]) -> str:
            return "  " + "  ".join(str(v).ljust(a) for v, a in zip(valores, anchos)).rstrip()
 
        separador = "  " + "  ".join("-" * a for a in anchos)
        return "\n".join([linea(encabezado), separador, *(linea(f) for f in filas)])
 