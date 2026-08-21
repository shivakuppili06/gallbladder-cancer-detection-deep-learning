from flask import Flask, send_from_directory
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=None)


@app.route("/")
def index():
    if (BASE_DIR / "frontend" / "index.html").exists():
        return send_from_directory(BASE_DIR / "frontend", "index.html")
    return send_from_directory(BASE_DIR, "index.html")


if __name__ == "__main__":
    # Serves index.html on http://localhost:8501
    # The actual inference API is server.py, running separately on port 5000.
    app.run(host="0.0.0.0", port=8501, debug=False)