"""SQLite-backed user event timeline for explainable ZTNA-UEBA decisions."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from threading import Lock
from typing import Mapping

from .operator_explain import common_link_indicators, sanitize_event


SUSPICIOUS_STAGES = frozenset({"monitor", "step_up", "restrict", "deny"})


class TimelineStore:
    """Persist assessments and reconstruct an actor-centric suspicious timeline."""

    def __init__(self, database: str | Path = ":memory:") -> None:
        self.database = str(database)
        self._connection = sqlite3.connect(self.database, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = Lock()
        self._create_schema()

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS assessed_events (
                    event_id TEXT PRIMARY KEY,
                    actor_id TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    source_type TEXT,
                    event_type TEXT,
                    risk_score REAL,
                    raw_risk_score REAL NOT NULL,
                    trust_score REAL,
                    confidence REAL NOT NULL,
                    policy_stage TEXT NOT NULL,
                    policy_action TEXT,
                    suspicious INTEGER NOT NULL,
                    explanation_status TEXT NOT NULL DEFAULT 'completed',
                    explanation_error TEXT,
                    explanation_updated_at TEXT,
                    review_status TEXT NOT NULL DEFAULT 'new',
                    review_note TEXT,
                    reviewed_at TEXT,
                    readable_log_json TEXT NOT NULL,
                    explanation_json TEXT NOT NULL,
                    event_snapshot_json TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_actor_time "
                "ON assessed_events(actor_id, occurred_at)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_policy_time "
                "ON assessed_events(policy_stage, occurred_at)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_suspicious_time "
                "ON assessed_events(suspicious, occurred_at)"
            )
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Add operational columns to databases created by the PoC version."""

        with self._lock, self._connection:
            existing = {
                str(row["name"])
                for row in self._connection.execute("PRAGMA table_info(assessed_events)")
            }
            additions = {
                "explanation_status": "TEXT NOT NULL DEFAULT 'completed'",
                "explanation_error": "TEXT",
                "explanation_updated_at": "TEXT",
                "review_status": "TEXT NOT NULL DEFAULT 'new'",
                "review_note": "TEXT",
                "reviewed_at": "TEXT",
            }
            for name, definition in additions.items():
                if name not in existing:
                    self._connection.execute(
                        f"ALTER TABLE assessed_events ADD COLUMN {name} {definition}"
                    )

    def close(self) -> None:
        self._connection.close()

    def record(
        self,
        record: Mapping[str, object],
        prediction: Mapping[str, object],
        explanation: Mapping[str, object],
        *,
        explanation_status: str = "completed",
    ) -> str:
        event = dict(explanation["event"])
        event_id = str(event["event_id"])
        policy = dict(prediction.get("policy") or {})
        stage = str(policy.get("stage") or "shadow")
        risk_score = prediction.get("risk_score")
        raw_risk = float(prediction.get("raw_model_risk_probability") or 0.0)
        effective_risk = float(risk_score) if risk_score is not None else raw_risk
        suspicious = stage in SUSPICIOUS_STAGES or effective_risk >= 0.20
        values = (
            event_id,
            str(event["actor_id"]),
            str(event["occurred_at"]),
            event.get("source_type"),
            event.get("event_type"),
            None if risk_score is None else float(risk_score),
            raw_risk,
            None if prediction.get("trust_score") is None else float(prediction["trust_score"]),
            float(prediction.get("confidence") or 0.0),
            stage,
            policy.get("action"),
            1 if suspicious else 0,
            explanation_status,
            None,
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            json.dumps(explanation["readable_log"], ensure_ascii=False, default=str),
            json.dumps(explanation, ensure_ascii=False, default=str),
            json.dumps(sanitize_event(record), ensure_ascii=False, default=str),
        )
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO assessed_events (
                    event_id, actor_id, occurred_at, source_type, event_type,
                    risk_score, raw_risk_score, trust_score, confidence,
                    policy_stage, policy_action, suspicious, explanation_status,
                    explanation_error, explanation_updated_at, readable_log_json,
                    explanation_json, event_snapshot_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    actor_id=excluded.actor_id,
                    occurred_at=excluded.occurred_at,
                    source_type=excluded.source_type,
                    event_type=excluded.event_type,
                    risk_score=excluded.risk_score,
                    raw_risk_score=excluded.raw_risk_score,
                    trust_score=excluded.trust_score,
                    confidence=excluded.confidence,
                    policy_stage=excluded.policy_stage,
                    policy_action=excluded.policy_action,
                    suspicious=excluded.suspicious,
                    explanation_status=excluded.explanation_status,
                    explanation_error=excluded.explanation_error,
                    explanation_updated_at=excluded.explanation_updated_at,
                    readable_log_json=excluded.readable_log_json,
                    explanation_json=excluded.explanation_json,
                    event_snapshot_json=excluded.event_snapshot_json
                """,
                values,
            )
        return event_id

    def set_explanation_status(
        self,
        event_id: str,
        status: str,
        *,
        error: str | None = None,
    ) -> bool:
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE assessed_events
                SET explanation_status = ?, explanation_error = ?,
                    explanation_updated_at = ?
                WHERE event_id = ?
                """,
                (status, error, now, event_id),
            )
        return cursor.rowcount > 0

    def incomplete_explanations(self, *, limit: int = 256) -> list[dict]:
        """Return sanitized events whose background work was interrupted."""

        safe_limit = max(1, min(int(limit), 10_000))
        with self._lock:
            rows = list(
                self._connection.execute(
                    """
                    SELECT event_id, event_snapshot_json
                    FROM assessed_events
                    WHERE explanation_status IN ('pending', 'processing')
                    ORDER BY explanation_updated_at ASC, occurred_at ASC
                    LIMIT ?
                    """,
                    (safe_limit,),
                )
            )
        return [
            {
                "event_id": row["event_id"],
                "event": json.loads(row["event_snapshot_json"]),
            }
            for row in rows
        ]

    def update_explanation(
        self,
        event_id: str,
        explanation: Mapping[str, object],
        *,
        status: str = "completed",
        error: str | None = None,
    ) -> bool:
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE assessed_events
                SET readable_log_json = ?, explanation_json = ?,
                    explanation_status = ?, explanation_error = ?,
                    explanation_updated_at = ?
                WHERE event_id = ?
                """,
                (
                    json.dumps(explanation["readable_log"], ensure_ascii=False, default=str),
                    json.dumps(explanation, ensure_ascii=False, default=str),
                    status,
                    error,
                    now,
                    event_id,
                ),
            )
        return cursor.rowcount > 0

    def list_events(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        suspicious_only: bool = False,
        policy_stage: str | None = None,
        actor_id: str | None = None,
        source_type: str | None = None,
    ) -> dict:
        clauses = ["1 = 1"]
        parameters: list[object] = []
        if suspicious_only:
            clauses.append("suspicious = 1")
        if policy_stage:
            clauses.append("policy_stage = ?")
            parameters.append(policy_stage)
        if actor_id:
            clauses.append("actor_id = ?")
            parameters.append(actor_id)
        if source_type:
            clauses.append("source_type = ?")
            parameters.append(source_type)
        where = " AND ".join(clauses)
        safe_limit = max(1, min(int(limit), 500))
        safe_offset = max(0, int(offset))
        with self._lock:
            total = int(
                self._connection.execute(
                    f"SELECT COUNT(*) FROM assessed_events WHERE {where}", parameters
                ).fetchone()[0]
            )
            rows = list(
                self._connection.execute(
                    f"""
                    SELECT event_id, actor_id, occurred_at, source_type, event_type,
                           risk_score, raw_risk_score, trust_score, confidence,
                           policy_stage, policy_action, suspicious,
                           explanation_status, explanation_error,
                           review_status, readable_log_json
                    FROM assessed_events
                    WHERE {where}
                    ORDER BY occurred_at DESC, event_id DESC
                    LIMIT ? OFFSET ?
                    """,
                    [*parameters, safe_limit, safe_offset],
                )
            )
        return {
            "total": total,
            "limit": safe_limit,
            "offset": safe_offset,
            "items": [self._event_summary(row) for row in rows],
        }

    def event_detail(self, event_id: str) -> dict | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM assessed_events WHERE event_id = ?", (event_id,)
            ).fetchone()
        if row is None:
            return None
        explanation = json.loads(row["explanation_json"])
        return {
            **self._event_summary(row),
            "event": explanation.get("event", {}),
            "readable_log": json.loads(row["readable_log_json"]),
            "evidence": {
                "risk_increasing": explanation.get("risk_increasing_evidence", []),
                "risk_decreasing": explanation.get("risk_decreasing_evidence", []),
                "influence_check": explanation.get("counterfactual_evidence", {}),
            },
            "audit": explanation.get("audit", {}),
            "review_note": row["review_note"],
            "reviewed_at": row["reviewed_at"],
            "raw_event": json.loads(row["event_snapshot_json"]),
        }

    def dashboard_summary(self, *, hours: int = 24) -> dict:
        since = (
            datetime.now(timezone.utc) - timedelta(hours=max(1, min(int(hours), 24 * 365)))
        ).isoformat().replace("+00:00", "Z")
        with self._lock:
            totals = self._connection.execute(
                """
                SELECT COUNT(*) AS total,
                       COALESCE(SUM(suspicious), 0) AS suspicious,
                       AVG(COALESCE(risk_score, raw_risk_score)) AS average_risk,
                       SUM(CASE WHEN explanation_status IN ('pending', 'processing')
                                THEN 1 ELSE 0 END) AS pending_explanations,
                       MAX(occurred_at) AS latest_event_at
                FROM assessed_events WHERE occurred_at >= ?
                """,
                (since,),
            ).fetchone()
            policy_rows = list(
                self._connection.execute(
                    """SELECT policy_stage, COUNT(*) AS count
                       FROM assessed_events WHERE occurred_at >= ?
                       GROUP BY policy_stage ORDER BY count DESC""",
                    (since,),
                )
            )
            source_rows = list(
                self._connection.execute(
                    """SELECT COALESCE(source_type, 'unknown') AS source, COUNT(*) AS count
                       FROM assessed_events WHERE occurred_at >= ?
                       GROUP BY COALESCE(source_type, 'unknown') ORDER BY count DESC""",
                    (since,),
                )
            )
        total = int(totals["total"] or 0)
        suspicious = int(totals["suspicious"] or 0)
        return {
            "window_hours": max(1, min(int(hours), 24 * 365)),
            "total_events": total,
            "suspicious_events": suspicious,
            "suspicious_rate": suspicious / total if total else 0.0,
            "average_risk": float(totals["average_risk"] or 0.0),
            "pending_explanations": int(totals["pending_explanations"] or 0),
            "latest_event_at": totals["latest_event_at"],
            "policy_counts": {
                str(row["policy_stage"]): int(row["count"]) for row in policy_rows
            },
            "source_counts": {
                str(row["source"]): int(row["count"]) for row in source_rows
            },
        }

    def set_review(
        self, event_id: str, status: str, *, note: str | None = None
    ) -> bool:
        allowed = {"new", "investigating", "resolved", "false_positive"}
        if status not in allowed:
            raise ValueError(f"review status must be one of {sorted(allowed)}")
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE assessed_events
                SET review_status = ?, review_note = ?, reviewed_at = ?
                WHERE event_id = ?
                """,
                (status, note, now, event_id),
            )
        return cursor.rowcount > 0

    @staticmethod
    def _event_summary(row: sqlite3.Row) -> dict:
        readable = json.loads(row["readable_log_json"])
        return {
            "event_id": row["event_id"],
            "actor_id": row["actor_id"],
            "occurred_at": row["occurred_at"],
            "source_type": row["source_type"],
            "event_type": row["event_type"],
            "risk_score": row["risk_score"],
            "raw_risk_score": row["raw_risk_score"],
            "trust_score": row["trust_score"],
            "confidence": row["confidence"],
            "policy": {
                "stage": row["policy_stage"],
                "action": row["policy_action"],
            },
            "suspicious": bool(row["suspicious"]),
            "explanation_status": row["explanation_status"],
            "explanation_error": row["explanation_error"],
            "review_status": row["review_status"],
            "headline": readable.get("headline"),
            "event_summary": readable.get("event_summary") or readable.get("event"),
        }

    def actor_timeline(
        self,
        actor_id: str,
        *,
        suspicious_only: bool = True,
        limit: int = 100,
        start: str | None = None,
        end: str | None = None,
    ) -> dict:
        clauses = ["actor_id = ?"]
        parameters: list[object] = [actor_id]
        if suspicious_only:
            clauses.append("suspicious = 1")
        if start:
            clauses.append("occurred_at >= ?")
            parameters.append(start)
        if end:
            clauses.append("occurred_at <= ?")
            parameters.append(end)
        parameters.append(max(1, min(int(limit), 1000)))
        query = (
            "SELECT * FROM (SELECT * FROM assessed_events WHERE "
            + " AND ".join(clauses)
            + " ORDER BY occurred_at DESC LIMIT ?) ORDER BY occurred_at ASC"
        )
        with self._lock:
            rows = list(self._connection.execute(query, parameters))

        timeline = []
        previous_snapshot: dict | None = None
        previous_event_id: str | None = None
        actor_display = "해당 사용자"
        for row in rows:
            snapshot = json.loads(row["event_snapshot_json"])
            explanation = json.loads(row["explanation_json"])
            actor_display = explanation.get("event", {}).get("actor_display") or actor_display
            links = []
            if previous_snapshot is not None:
                links = common_link_indicators(previous_snapshot, snapshot)
                links.insert(
                    0,
                    {
                        "type": "same_actor",
                        "previous_event_id": previous_event_id,
                        "message": "같은 사용자 계정의 이전 이벤트와 시간순으로 연결했다.",
                    },
                )
            timeline.append(
                {
                    "event_id": row["event_id"],
                    "occurred_at": row["occurred_at"],
                    "source_type": row["source_type"],
                    "event_type": row["event_type"],
                    "risk_score": row["risk_score"],
                    "raw_risk_score": row["raw_risk_score"],
                    "trust_score": row["trust_score"],
                    "confidence": row["confidence"],
                    "policy": {
                        "stage": row["policy_stage"],
                        "action": row["policy_action"],
                    },
                    "readable_log": json.loads(row["readable_log_json"]),
                    "reasons": (
                        explanation.get("risk_increasing_evidence", [])
                        + explanation.get("risk_decreasing_evidence", [])
                    ),
                    "linked_from_previous": links,
                    "raw_event_reference": f"timeline://events/{row['event_id']}",
                }
            )
            previous_snapshot = snapshot
            previous_event_id = row["event_id"]

        policy_counts = Counter(item["policy"]["stage"] for item in timeline)
        sources = sorted({str(item["source_type"]) for item in timeline if item["source_type"]})
        risks = [
            float(item["risk_score"] if item["risk_score"] is not None else item["raw_risk_score"])
            for item in timeline
        ]
        summary = {
            "actor_id": actor_id,
            "returned_events": len(timeline),
            "suspicious_only": suspicious_only,
            "first_seen": timeline[0]["occurred_at"] if timeline else None,
            "last_seen": timeline[-1]["occurred_at"] if timeline else None,
            "maximum_risk": max(risks) if risks else None,
            "source_count": len(sources),
            "sources": sources,
            "policy_counts": dict(sorted(policy_counts.items())),
        }
        if timeline:
            summary["narrative"] = (
                f"{actor_display}에서 {len(timeline)}건의 "
                f"{'의심 ' if suspicious_only else ''}이벤트가 시간순으로 확인됐다. "
                f"관련 로그 원천은 {len(sources)}개이고 최대 위험도는 {max(risks):.3f}이다."
            )
        else:
            summary["narrative"] = "해당 사용자에게 조건에 맞는 이벤트가 없다."
        return {"summary": summary, "timeline": timeline}

    def raw_event(self, event_id: str) -> dict | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT event_snapshot_json FROM assessed_events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return None if row is None else json.loads(row["event_snapshot_json"])
