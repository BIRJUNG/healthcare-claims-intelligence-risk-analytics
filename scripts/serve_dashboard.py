from __future__ import annotations

import argparse
import http.server
import socketserver
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the Healthcare Claims Intelligence dashboard locally.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8061)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    dashboard = root / "reports" / "dashboard" / "healthcare_claims_intelligence_dashboard.html"
    if not dashboard.exists():
        raise SystemExit("Dashboard not found. Run python scripts/run_healthcare_claims_pipeline.py first.")

    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer((args.host, args.port), handler) as server:
        print(f"Serving {root}")
        print(f"Open http://{args.host}:{args.port}/reports/dashboard/healthcare_claims_intelligence_dashboard.html")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

