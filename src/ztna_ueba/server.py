"""Operational FastAPI server and background explanation queue."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import os
from pathlib import Path
import re
import secrets
from threading import BoundedSemaphore, Lock
from typing import Any, Literal, Mapping
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .inference import TrustPredictor
from .service import ExplainableTrustService
from .timeline import SUSPICIOUS_STAGES, TimelineStore


class AssessmentEnvelope(BaseModel):
    """Stable external contract for arbitrary event schemas."""

    model_config = ConfigDict(extra="forbid")

    event: dict[str, Any]
    explanation_mode: Literal["auto", "none", "full"] = "auto"


class BatchAssessmentEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: list[dict[str, Any]] = Field(min_length=1, max_length=100)
    explanation_mode: Literal["auto", "none"] = "auto"


class ReviewUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["new", "investigating", "resolved", "false_positive"]
    note: str | None = Field(default=None, max_length=2000)


class ExplanationQueue:
    """Bounded in-process queue for expensive exact subset explanations."""

    def __init__(
        self,
        service: ExplainableTrustService,
        workers: int = 1,
        capacity: int = 256,
    ) -> None:
        self.service = service
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, min(int(workers), 4)),
            thread_name_prefix="ztna-explanation",
        )
        self._running: set[str] = set()
        self._lock = Lock()
        self._capacity = BoundedSemaphore(max(1, min(int(capacity), 10_000)))

    def submit(
        self,
        event: Mapping[str, object],
        *,
        event_id: str,
        prediction: Mapping[str, object] | None = None,
    ) -> bool:
        with self._lock:
            if event_id in self._running:
                return False
            if not self._capacity.acquire(blocking=False):
                self.service.store.set_explanation_status(event_id, "deferred")
                return False
            self._running.add(event_id)
        self.service.store.set_explanation_status(event_id, "pending")
        future = self._executor.submit(
            self._run,
            dict(event),
            event_id,
            None if prediction is None else dict(prediction),
        )
        future.add_done_callback(lambda _: self._finish(event_id))
        return True

    def _run(
        self,
        event: Mapping[str, object],
        event_id: str,
        prediction: Mapping[str, object] | None,
    ) -> None:
        self.service.store.set_explanation_status(event_id, "processing")
        try:
            self.service.complete_explanation(event, prediction=prediction)
        except Exception as exc:  # pragma: no cover - defensive worker boundary
            self.service.store.set_explanation_status(
                event_id, "failed", error=f"{type(exc).__name__}: {exc}"
            )

    def _finish(self, event_id: str) -> None:
        with self._lock:
            self._running.discard(event_id)
            self._capacity.release()

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)


def _validate_event_shape(event: Mapping[str, object]) -> None:
    if not event:
        raise HTTPException(status_code=422, detail="event must not be empty")
    if len(event) > 512:
        raise HTTPException(status_code=413, detail="event has more than 512 fields")
    external_event_id = event.get("event_id")
    if external_event_id is not None:
        event_id = str(external_event_id)
        if len(event_id) > 128 or not re.fullmatch(r"[A-Za-z0-9._:-]+", event_id):
            raise HTTPException(
                status_code=422,
                detail="event_id must use 1-128 letters, numbers, '.', '_', ':' or '-'",
            )

    def depth(value: object, level: int = 0) -> int:
        if level > 8:
            return level
        if isinstance(value, Mapping):
            return max((depth(item, level + 1) for item in value.values()), default=level)
        if isinstance(value, list):
            return max((depth(item, level + 1) for item in value), default=level)
        return level

    if depth(event) > 8:
        raise HTTPException(status_code=413, detail="event nesting exceeds 8 levels")


def create_app(
    service: ExplainableTrustService,
    *,
    api_key: str = "",
    explanation_workers: int = 1,
    scoring_concurrency: int = 2,
    explanation_queue_capacity: int = 256,
    max_request_bytes: int = 1_048_576,
    dashboard_directory: str | Path | None = None,
) -> FastAPI:
    """Create the operational API without loading a model at import time."""

    explanation_queue = ExplanationQueue(
        service, explanation_workers, capacity=explanation_queue_capacity
    )
    scoring_slots = BoundedSemaphore(max(1, min(int(scoring_concurrency), 32)))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        for interrupted in service.store.incomplete_explanations(
            limit=explanation_queue_capacity
        ):
            explanation_queue.submit(
                interrupted["event"], event_id=str(interrupted["event_id"])
            )
        yield
        explanation_queue.close()

    app = FastAPI(
        title="ZTNA-UEBA Trust Scoring API",
        version="1.0.0",
        description=(
            "가변 로그의 UEBA 기준선 편차와 필드 조합으로 Trust Score와 "
            "ZTNA 정책을 반환하고, 의심 이벤트의 상세 설명을 비동기로 생성한다."
        ),
        lifespan=lifespan,
    )
    app.state.service = service
    app.state.explanation_queue = explanation_queue

    @app.middleware("http")
    async def operational_headers(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        content_length = request.headers.get("Content-Length")
        if content_length:
            try:
                if int(content_length) > max_request_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "request body is too large"},
                        headers={"X-Request-ID": request_id},
                    )
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "invalid Content-Length header"},
                    headers={"X-Request-ID": request_id},
                )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    def require_api_key(
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        authorization: str | None = Header(default=None),
    ) -> None:
        if not api_key:
            return
        bearer = ""
        if authorization and authorization.lower().startswith("bearer "):
            bearer = authorization[7:].strip()
        header_match = bool(x_api_key) and secrets.compare_digest(x_api_key, api_key)
        bearer_match = bool(bearer) and secrets.compare_digest(bearer, api_key)
        if not header_match and not bearer_match:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="valid API key required",
            )

    auth = [Depends(require_api_key)]

    @app.get("/health/live", tags=["operations"])
    def health_live() -> dict:
        return {"status": "ok"}

    @app.get("/health/ready", tags=["operations"])
    def health_ready() -> dict:
        try:
            service.store.dashboard_summary(hours=1)
        except Exception as exc:  # pragma: no cover - defensive health boundary
            raise HTTPException(status_code=503, detail=f"storage unavailable: {exc}") from exc
        return {"status": "ready", "model_loaded": True, "storage_ready": True}

    def assess_one(
        event: Mapping[str, object],
        explanation_mode: Literal["auto", "none", "full"],
    ) -> dict:
        _validate_event_shape(event)
        with scoring_slots:
            if explanation_mode == "full":
                return service.assess(event)
            result, prediction = service.prepare_fast(
                event,
                explanation_status="pending",
            )

        stage = str((result.get("decision", {}).get("policy") or {}).get("stage") or "shadow")
        confidence = float(result.get("decision", {}).get("confidence") or 0.0)
        requires_detail = (
            explanation_mode == "auto"
            and (stage in SUSPICIOUS_STAGES or confidence < 0.75)
        )
        if requires_detail:
            queued = explanation_queue.submit(
                event,
                event_id=str(result["event_id"]),
                prediction=prediction,
            )
            result["explanation_status"] = "pending" if queued else "deferred"
        else:
            service.store.set_explanation_status(str(result["event_id"]), "skipped")
            result["explanation_status"] = "skipped"
        return result

    @app.post(
        "/api/v1/assess",
        status_code=status.HTTP_201_CREATED,
        dependencies=auth,
        tags=["scoring"],
    )
    def assess(payload: AssessmentEnvelope) -> dict:
        return assess_one(payload.event, payload.explanation_mode)

    @app.post(
        "/api/v1/assess/batch",
        status_code=status.HTTP_201_CREATED,
        dependencies=auth,
        tags=["scoring"],
    )
    def assess_batch(payload: BatchAssessmentEnvelope) -> dict:
        results = [
            assess_one(event, payload.explanation_mode) for event in payload.events
        ]
        return {
            "accepted": len(results),
            "results": results,
        }

    @app.get("/api/v1/overview", dependencies=auth, tags=["dashboard"])
    def overview(hours: int = Query(default=24, ge=1, le=8760)) -> dict:
        return service.store.dashboard_summary(hours=hours)

    @app.get("/api/v1/events", dependencies=auth, tags=["dashboard"])
    def events(
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        suspicious_only: bool = False,
        policy_stage: str | None = None,
        actor_id: str | None = None,
        source_type: str | None = None,
    ) -> dict:
        return service.store.list_events(
            limit=limit,
            offset=offset,
            suspicious_only=suspicious_only,
            policy_stage=policy_stage,
            actor_id=actor_id,
            source_type=source_type,
        )

    @app.get("/api/v1/events/{event_id}", dependencies=auth, tags=["dashboard"])
    def event_detail(event_id: str) -> dict:
        result = service.store.event_detail(event_id)
        if result is None:
            raise HTTPException(status_code=404, detail="event not found")
        return result

    @app.post(
        "/api/v1/events/{event_id}/explanation",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=auth,
        tags=["dashboard"],
    )
    def retry_explanation(event_id: str) -> dict:
        event = service.raw_event(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="event not found")
        queued = explanation_queue.submit(event, event_id=event_id)
        return {
            "event_id": event_id,
            "explanation_status": "pending",
            "queued": queued,
        }

    @app.patch(
        "/api/v1/events/{event_id}/review",
        dependencies=auth,
        tags=["dashboard"],
    )
    def update_review(event_id: str, payload: ReviewUpdate) -> dict:
        if not service.store.set_review(event_id, payload.status, note=payload.note):
            raise HTTPException(status_code=404, detail="event not found")
        return {"event_id": event_id, "review_status": payload.status}

    @app.get(
        "/api/v1/actors/{actor_id}/timeline",
        dependencies=auth,
        tags=["dashboard"],
    )
    def actor_timeline(
        actor_id: str,
        suspicious_only: bool = True,
        limit: int = Query(default=100, ge=1, le=1000),
        start: str | None = None,
        end: str | None = None,
    ) -> dict:
        return service.actor_timeline(
            actor_id,
            suspicious_only=suspicious_only,
            limit=limit,
            start=start,
            end=end,
        )

    static_path = Path(dashboard_directory) if dashboard_directory else Path(__file__).with_name("dashboard")
    if static_path.is_dir():
        app.mount("/", StaticFiles(directory=static_path, html=True), name="dashboard")

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ZTNA-UEBA operational API")
    parser.add_argument("--checkpoint", default=os.getenv("ZTNA_CHECKPOINT"))
    parser.add_argument(
        "--database", default=os.getenv("ZTNA_DATABASE", "artifacts/operations.sqlite")
    )
    parser.add_argument("--host", default=os.getenv("ZTNA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("ZTNA_PORT", "8080")))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=os.getenv("ZTNA_DEVICE", "auto"))
    parser.add_argument("--api-key", default=os.getenv("ZTNA_API_KEY", ""))
    parser.add_argument(
        "--explanation-workers",
        type=int,
        default=int(os.getenv("ZTNA_EXPLANATION_WORKERS", "1")),
    )
    parser.add_argument(
        "--scoring-concurrency",
        type=int,
        default=int(os.getenv("ZTNA_SCORING_CONCURRENCY", "2")),
    )
    parser.add_argument(
        "--explanation-queue-capacity",
        type=int,
        default=int(os.getenv("ZTNA_EXPLANATION_QUEUE_CAPACITY", "256")),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint:
        raise SystemExit("--checkpoint or ZTNA_CHECKPOINT is required")
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not args.api_key:
        raise SystemExit("ZTNA_API_KEY is required when binding to a non-loopback host")

    database = Path(args.database)
    database.parent.mkdir(parents=True, exist_ok=True)
    predictor = TrustPredictor(args.checkpoint, device=args.device)
    store = TimelineStore(database)
    service = ExplainableTrustService(predictor, store)
    app = create_app(
        service,
        api_key=args.api_key,
        explanation_workers=args.explanation_workers,
        scoring_concurrency=args.scoring_concurrency,
        explanation_queue_capacity=args.explanation_queue_capacity,
    )

    import uvicorn

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        store.close()


if __name__ == "__main__":
    main()
