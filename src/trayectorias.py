"""Trayectorias de borrascas: descarga los modelos, sigue el centro y publica en docs/datos_borrascas/.

Lo ejecuta .github/workflows/trayectorias.yml tras cada pasada de 00 y 12 UTC.
Ayuda:  python src/trayectorias.py -h
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--demo":
        from borrascas import demo
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        demo.main(sys.argv[2])
    else:
        from borrascas.generar import main
        main()
