import logging
import os
import sys
from pathlib import Path

from streamlit.web import cli

import metrics
from captioner import DEFAULT_MODEL, get_captioner

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.getLogger("caption_app").setLevel(logging.INFO)
    metrics.start_metrics_server()
    preload = os.environ.get("PRELOAD_MODEL", DEFAULT_MODEL)
    if preload:
        get_captioner(preload)
    sys.argv = ["streamlit", "run", str(Path(__file__).with_name("app.py")), *sys.argv[1:]]
    sys.exit(cli.main())
