"""Persistencia de los resultados de cada modelo para combinarlos entre notebooks."""
from __future__ import annotations
 
import json
from datetime import datetime
from pathlib import Path
from typing import Any
 
import numpy as np
import pandas as pd
 
 
class AlmacenResultados:
    """Guarda y recupera los resultados de un modelo como archivo JSON.
 
    Cada notebook guarda su resultado con un nombre fijo (``regresion``,
    ``clasificacion``). El notebook que va al final los carga y los combina.
    Cada guardado agrega la fecha y hora de ejecución, para que el reporte
    refleje cuándo se produjeron los resultados.
    """
 
    def __init__(self, directorio: str | Path = "resultados") -> None:
        self.directorio = Path(directorio)
 
    def ruta(self, nombre: str) -> Path:
        """Devuelve la ruta del archivo JSON de un resultado."""
        if not isinstance(nombre, str) or not nombre.strip():
            raise ValueError("nombre debe ser un texto no vacío.")
        if any(separador in nombre for separador in ("/", "\\")):
            raise ValueError("nombre no puede contener separadores de carpeta.")
        return self.directorio / f"{nombre}.json"
 
    def guardar(self, nombre: str, resultados: dict[str, Any]) -> Path:
        """Guarda un diccionario de resultados y devuelve la ruta del archivo."""
        if not isinstance(resultados, dict):
            raise TypeError("resultados debe ser un diccionario.")
        destino = self.ruta(nombre)
        destino.parent.mkdir(parents=True, exist_ok=True)
        contenido = {
            "fecha_ejecucion": datetime.now().isoformat(timespec="seconds"),
            **resultados,
        }
        with destino.open("w", encoding="utf-8") as archivo:
            json.dump(contenido, archivo, ensure_ascii=False, indent=2, default=self._convertir)
        return destino
 
    def cargar(self, nombre: str) -> dict[str, Any]:
        """Carga los resultados guardados por otro notebook.
 
        Raises:
            FileNotFoundError: Si el notebook que los produce aún no se ejecutó.
        """
        origen = self.ruta(nombre)
        if not origen.is_file():
            raise FileNotFoundError(
                f"No existe {origen}. Ejecute primero el notebook que genera "
                f"los resultados de '{nombre}'."
            )
        with origen.open("r", encoding="utf-8") as archivo:
            return json.load(archivo)
 
    @staticmethod
    def _convertir(objeto: Any) -> Any:
        """Convierte tipos de NumPy, pandas y rutas a tipos que admite JSON."""
        if isinstance(objeto, np.integer):
            return int(objeto)
        if isinstance(objeto, np.floating):
            return float(objeto)
        if isinstance(objeto, np.bool_):
            return bool(objeto)
        if isinstance(objeto, np.ndarray):
            return objeto.tolist()
        if isinstance(objeto, pd.Series):
            return objeto.to_dict()
        if isinstance(objeto, pd.DataFrame):
            return objeto.to_dict("records")
        if isinstance(objeto, Path):
            return str(objeto)
        raise TypeError(f"No se puede guardar un objeto de tipo {type(objeto).__name__}.")
 