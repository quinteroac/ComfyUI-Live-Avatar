"""Serves this folder on this machine only. Browsers will not load the avatar's modules from file:// paths."""
import functools
import http.server
import os
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".mjs": "text/javascript", ".wasm": "application/wasm"}


handler = functools.partial(Handler, directory=os.path.dirname(os.path.abspath(__file__)))
print(f"Avatar:      http://127.0.0.1:{PORT}/")  # noqa: T201
print(f"OBS overlay: http://127.0.0.1:{PORT}/?bg=transparent&ui=0")  # noqa: T201
http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler).serve_forever()
