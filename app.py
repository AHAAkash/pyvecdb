"""Vercel entrypoint for the public, read-only demo."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pyvecdb.api import create_app
from pyvecdb.demo import load_demo

app = create_app(read_only=True)
load_demo(app.state.pipeline)
