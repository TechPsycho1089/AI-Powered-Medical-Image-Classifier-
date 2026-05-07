"""
run_app.py  —  Run this to start the Flask prediction server.
Usage:
    python run_app.py
Then open:  http://localhost:8080
"""
from src.app import app

if __name__ == "__main__":
    print("\n[SERVER] Starting Malaria Cell Classifier API...")
    print("[SERVER] POST image to: http://localhost:8080/predict")
    print("[SERVER] Health check:  http://localhost:8080/health\n")
    app.run(debug=True, host="0.0.0.0", port=8080)
