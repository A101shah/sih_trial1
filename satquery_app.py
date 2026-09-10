"""
SatQuery AI: Interactive Multimodal Remote-Sensing AI Assistant
Problem Statement: SIH26167
Main Application Entrypoint
"""

import sys
import os

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def main():
    port = 8080
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass

    print("==========================================================")
    print("           SATQUERY AI - REMOTE SENSING ASSISTANT          ")
    print("                     SIH Problem: SIH26167                ")
    print("==========================================================")
    print(f"Starting server on http://localhost:{port} ...")
    print(f"Interactive API Docs available at http://localhost:{port}/docs")
    print("Open this URL in your browser to interact with SatQuery AI.")
    print("==========================================================")

    try:
        import uvicorn
        from satquery.server.fastapi_app import app
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except Exception as e:
        print(f"Uvicorn fallback due to: {e}. Starting standard multi-threaded HTTP server...")
        from satquery.server.app import run_server
        run_server(host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
