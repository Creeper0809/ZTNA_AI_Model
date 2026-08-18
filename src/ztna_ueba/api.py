"""Minimal HTTP API for the explainable timeline PoC."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .inference import TrustPredictor
from .service import ExplainableTrustService
from .timeline import TimelineStore


def create_server(
    service: ExplainableTrustService,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def _json_response(self, status: int, payload: object) -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str) -> None:
            self._json_response(status, {"error": message})

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler contract
            path = urlparse(self.path).path
            if path != "/v1/assess":
                self._error(404, "endpoint not found")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                event = (
                    payload["event"]
                    if isinstance(payload, dict) and isinstance(payload.get("event"), dict)
                    else payload
                )
                result = service.assess(event)
            except (ValueError, TypeError, KeyError) as exc:
                self._error(400, str(exc))
                return
            self._json_response(201, result)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self._json_response(200, {"status": "ok"})
                return
            if parsed.path.startswith("/v1/actors/") and parsed.path.endswith("/timeline"):
                actor_id = unquote(parsed.path[len("/v1/actors/") : -len("/timeline")]).strip("/")
                if not actor_id:
                    self._error(400, "actor_id is required")
                    return
                query = parse_qs(parsed.query)
                suspicious_only = query.get("suspicious_only", ["true"])[0].lower() not in {
                    "0",
                    "false",
                    "no",
                }
                try:
                    limit = int(query.get("limit", ["100"])[0])
                except ValueError:
                    self._error(400, "limit must be an integer")
                    return
                result = service.actor_timeline(
                    actor_id,
                    suspicious_only=suspicious_only,
                    limit=limit,
                    start=query.get("start", [None])[0],
                    end=query.get("end", [None])[0],
                )
                self._json_response(200, result)
                return
            if parsed.path.startswith("/v1/events/"):
                event_id = unquote(parsed.path[len("/v1/events/") :]).strip("/")
                event = service.raw_event(event_id)
                if event is None:
                    self._error(404, "event not found")
                else:
                    self._json_response(200, {"event_id": event_id, "event": event})
                return
            self._error(404, "endpoint not found")

        def log_message(self, format: str, *args: object) -> None:
            return

    return ThreadingHTTPServer((host, port), Handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the explainable ZTNA-UEBA PoC API")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--database", default="artifacts/explainable_timeline.sqlite")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    database = Path(args.database)
    database.parent.mkdir(parents=True, exist_ok=True)
    predictor = TrustPredictor(args.checkpoint, device=args.device)
    store = TimelineStore(database)
    service = ExplainableTrustService(predictor, store)
    server = create_server(service, args.host, args.port)
    print(f"ZTNA-UEBA PoC API listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        store.close()


if __name__ == "__main__":
    main()
