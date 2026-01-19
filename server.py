import runpy

ns = runpy.run_path("server_big.py")
app = ns["app"]
