const state = {
  apiKey: sessionStorage.getItem("ztnaApiKey") || "",
  selectedEventId: null,
  refreshTimer: null,
};

const policyLabels = {
  shadow: "모의 운영", allow: "허용", monitor: "관찰", step_up: "추가 인증",
  restrict: "접근 제한", deny: "차단",
};
const explanationLabels = {
  pending: "분석 대기", processing: "분석 중", completed: "분석 완료",
  failed: "분석 실패", skipped: "기본 근거",
  deferred: "분석 지연",
};
const reviewLabels = {
  new: "신규", investigating: "조사 중", resolved: "종결", false_positive: "오탐",
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}

function percent(value, digits = 1) {
  return `${(Number(value || 0) * 100).toFixed(digits)}%`;
}

function formatTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : new Intl.DateTimeFormat("ko-KR", {
    month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit",
    hour12: false,
  }).format(date);
}

async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (state.apiKey) headers["X-API-Key"] = state.apiKey;
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const error = new Error(payload.detail || `요청 실패 (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

function setConnection(ok, message) {
  const chip = document.querySelector("#apiState");
  chip.className = `status-chip ${ok ? "" : "error"}`;
  chip.querySelector("span").textContent = ok ? "API 정상" : "연결 오류";
  const banner = document.querySelector("#connectionBanner");
  if (ok) {
    banner.classList.add("hidden");
  } else {
    banner.textContent = message || "API에 연결할 수 없습니다.";
    banner.classList.remove("hidden");
  }
}

function renderBars(targetId, values, labelMap = {}) {
  const target = document.querySelector(targetId);
  target.classList.remove("loading-block");
  const rows = Object.entries(values || {}).sort((a, b) => b[1] - a[1]);
  if (!rows.length) {
    target.innerHTML = '<p class="muted">표시할 데이터가 없습니다.</p>';
    return;
  }
  const maximum = Math.max(...rows.map(([, count]) => Number(count)), 1);
  target.innerHTML = rows.slice(0, 6).map(([label, count]) => `
    <div class="bar-row">
      <span class="bar-label">${escapeHtml(labelMap[label] || label)}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.max(3, Number(count) / maximum * 100)}%"></div></div>
      <span class="bar-value">${Number(count).toLocaleString()}</span>
    </div>`).join("");
}

function renderOverview(data) {
  document.querySelector("#totalEvents").textContent = Number(data.total_events).toLocaleString();
  document.querySelector("#suspiciousEvents").textContent = Number(data.suspicious_events).toLocaleString();
  document.querySelector("#suspiciousRate").textContent = `전체의 ${percent(data.suspicious_rate)}`;
  document.querySelector("#averageRisk").textContent = Number(data.average_risk || 0).toFixed(3);
  document.querySelector("#pendingExplanations").textContent = Number(data.pending_explanations).toLocaleString();
  document.querySelector("#lastUpdated").textContent = `갱신 ${new Date().toLocaleTimeString("ko-KR", { hour12: false })}`;
  renderBars("#policyChart", data.policy_counts, policyLabels);
  renderBars("#sourceChart", data.source_counts);
}

function scoreClass(risk) {
  const value = Number(risk || 0);
  return value >= .7 ? "high" : value >= .35 ? "medium" : "";
}

function badge(label, css) {
  return `<span class="badge ${escapeHtml(css || "")}">${escapeHtml(label)}</span>`;
}

function renderEvents(data) {
  const body = document.querySelector("#eventRows");
  const empty = document.querySelector("#emptyEvents");
  document.querySelector("#eventCount").textContent = `${Number(data.total).toLocaleString()}건`;
  if (!data.items.length) {
    body.innerHTML = "";
    empty.classList.remove("hidden");
    return;
  }
  empty.classList.add("hidden");
  body.innerHTML = data.items.map(item => {
    const risk = item.risk_score ?? item.raw_risk_score;
    const stage = item.policy?.stage || "shadow";
    const status = item.explanation_status || "skipped";
    return `<tr data-event-id="${escapeHtml(item.event_id)}" tabindex="0">
      <td>${escapeHtml(formatTime(item.occurred_at))}</td>
      <td class="event-name"><strong>${escapeHtml(item.headline || item.event_type || "이벤트")}</strong><small>${escapeHtml(item.event_summary || "")}</small></td>
      <td>${escapeHtml(item.actor_id)}</td>
      <td>${escapeHtml((item.source_type || "unknown").toUpperCase())}</td>
      <td><span class="score ${scoreClass(risk)}">${Number(risk || 0).toFixed(3)}</span></td>
      <td><span class="score">${item.trust_score == null ? "—" : Number(item.trust_score).toFixed(1)}</span></td>
      <td>${badge(policyLabels[stage] || stage, stage)}</td>
      <td>${badge(explanationLabels[status] || status, status)}</td>
    </tr>`;
  }).join("");
  body.querySelectorAll("tr").forEach(row => {
    const open = () => openEvent(row.dataset.eventId);
    row.addEventListener("click", open);
    row.addEventListener("keydown", event => { if (event.key === "Enter") open(); });
  });
}

function evidenceHtml(rows, candidateRows = []) {
  if (!rows?.length) return '<p class="muted">표시할 필드 근거가 없습니다.</p>';
  const candidates = new Map((candidateRows || []).map(row => [row.field, row]));
  return `<div class="evidence-list">${rows.slice(0, 6).map(row => {
    const deviation = Number(row.baseline_anomaly_score || 0);
    const contribution = Number(row.model_risk_contribution || 0);
    const candidate = candidates.get(row.field);
    return `<article class="evidence-card">
      <div class="evidence-top"><strong>${escapeHtml(row.field_label || row.field)}</strong><code>${escapeHtml(row.field)}</code></div>
      <div class="evidence-metrics">
        <span class="metric-pill">관측값 ${escapeHtml(row.observed_value)}</span>
        <span class="metric-pill">가중치 ${percent(row.field_weight, 2)}</span>
        <span class="metric-pill">기준선 이탈도 ${(deviation * 100).toFixed(2)}%</span>
        <span class="metric-pill">위험 기여 ${contribution >= 0 ? "+" : ""}${contribution.toFixed(4)}</span>
      </div>
      <p>${escapeHtml(row.message || row.selection_basis?.reason || "")}</p>
      ${candidate?.selection_reason ? `<p class="selection-note"><strong>후보 선정:</strong> ${escapeHtml(candidate.selection_reason)}</p>` : ""}
    </article>`;
  }).join("")}</div>`;
}

function timelineHtml(data) {
  const items = data?.timeline || [];
  if (!items.length) return '<p class="muted">연결된 의심 이벤트가 없습니다.</p>';
  return `<div class="timeline">${items.map(item => `
    <div class="timeline-item">
      <time>${escapeHtml(formatTime(item.occurred_at))} · ${escapeHtml((item.source_type || "unknown").toUpperCase())}</time>
      <strong>${escapeHtml(item.readable_log?.headline || item.event_type || "이벤트")}</strong>
      <p>${escapeHtml(item.readable_log?.event_summary || item.readable_log?.event || "")}</p>
    </div>`).join("")}</div>`;
}

function influenceSummaryHtml(check) {
  if (!check || check.status === "unavailable" || !check.coalition_analysis) {
    return '<p class="muted">상세 조합 분석 결과가 아직 없습니다.</p>';
  }
  const discovery = check.candidate_discovery || {};
  const coalition = check.coalition_analysis || {};
  const confidence = check.explanation_confidence || {};
  const selected = discovery.selected_field_labels || discovery.selected_fields || check.candidate_fields || [];
  const limitations = confidence.limitations || [];
  return `
    <div class="analysis-grid">
      <div><span>후보 탐색</span><strong>${Number(discovery.pool_size || selected.length)}개 → ${Number(coalition.candidate_count || selected.length)}개</strong><small>기여도·가중치·이탈도·쌍 효과</small></div>
      <div><span>조합 재검사</span><strong>${Number(coalition.evaluated_coalitions || 0)}개</strong><small>모델 실행 ${Number(coalition.model_forward_calls || coalition.model_reinferences || 0)}회</small></div>
      <div><span>설명 품질</span><strong>${escapeHtml(confidence.grade_label || "—")}</strong><small>판정 정확도와 다른 지표</small></div>
    </div>
    <div class="selected-fields"><span>최종 조합 필드</span>${selected.map(label => badge(label, "")).join("")}</div>
    ${limitations.length ? `<ul class="limitations">${limitations.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : ""}`;
}

function renderDetail(detail, timeline) {
  const risk = detail.risk_score ?? detail.raw_risk_score;
  const stage = detail.policy?.stage || "shadow";
  const status = detail.explanation_status || "skipped";
  document.querySelector("#drawerTitle").textContent = detail.headline || "이벤트 상세";
  document.querySelector("#drawerContent").innerHTML = `
    <div class="detail-score-grid">
      <div class="detail-score"><span>위험도</span><strong class="score ${scoreClass(risk)}">${Number(risk || 0).toFixed(3)}</strong></div>
      <div class="detail-score"><span>Trust Score</span><strong>${detail.trust_score == null ? "—" : Number(detail.trust_score).toFixed(2)}</strong></div>
      <div class="detail-score"><span>정책</span><strong>${escapeHtml(policyLabels[stage] || stage)}</strong></div>
      <div class="detail-score"><span>분석</span><strong>${escapeHtml(explanationLabels[status] || status)}</strong></div>
    </div>
    <section class="detail-section">
      <h3>사람이 읽는 판정 근거</h3>
      <div class="narrative">
        <div class="narrative-row"><span>발생 행위</span><p>${escapeHtml(detail.readable_log?.event_summary || detail.readable_log?.event || "")}</p></div>
        <div class="narrative-row"><span>평소 행동</span><p>${escapeHtml(detail.readable_log?.usual_behavior || detail.readable_log?.baseline_reason || "")}</p></div>
        <div class="narrative-row"><span>현재 행동</span><p>${escapeHtml(detail.readable_log?.current_behavior || "")}</p></div>
        <div class="narrative-row"><span>UEBA 판단</span><p>${escapeHtml(detail.readable_log?.ueba_judgment || detail.readable_log?.judgment_reason || detail.readable_log?.reason || "")}</p></div>
        <div class="narrative-row"><span>정책 결과</span><p>${escapeHtml(detail.readable_log?.policy_reason || detail.readable_log?.decision || "")}</p></div>
      </div>
    </section>
    <section class="detail-section">
      <details class="technical-validation">
        <summary>기술 검증 보기</summary>
        <div class="narrative compact">
          <div class="narrative-row"><span>영향 검증</span><p>${escapeHtml(detail.readable_log?.technical_validation_reason || detail.readable_log?.model_influence_reason || "상세 조합 분석을 기다리고 있습니다.")}</p></div>
          <div class="narrative-row"><span>검사 범위</span><p>${escapeHtml(detail.readable_log?.analysis_scope_reason || "상세 분석 완료 후 제공합니다.")}</p></div>
          <div class="narrative-row"><span>설명 품질</span><p>${escapeHtml(detail.readable_log?.explanation_confidence_reason || "상세 분석 완료 후 계산합니다.")}</p></div>
        </div>
        ${influenceSummaryHtml(detail.evidence?.influence_check)}
      </details>
    </section>
    <section class="detail-section">
      <div class="panel-heading" style="padding:0 0 13px">
        <h3>위험을 높인 필드</h3>
        ${badge(explanationLabels[status] || status, status)}
      </div>
      ${evidenceHtml(detail.evidence?.risk_increasing, detail.evidence?.influence_check?.candidate_discovery?.candidate_rows)}
      ${(status === "failed" || status === "skipped" || status === "deferred") ? `<div class="detail-actions" style="margin-top:12px"><button id="retryExplanation" class="button secondary" type="button">상세 조합 분석 실행</button></div>` : ""}
    </section>
    <section class="detail-section">
      <h3>같은 사용자의 의심 이벤트</h3>
      ${timelineHtml(timeline)}
    </section>
    <section class="detail-section">
      <h3>관제 처리</h3>
      <div class="review-controls">
        <select id="reviewStatus">
          ${Object.entries(reviewLabels).map(([value, label]) => `<option value="${value}" ${detail.review_status === value ? "selected" : ""}>${label}</option>`).join("")}
        </select>
        <input id="reviewNote" maxlength="2000" placeholder="판정 메모" value="${escapeHtml(detail.review_note || "")}">
        <button id="saveReview" class="button primary" type="button">저장</button>
      </div>
    </section>
    <section class="detail-section">
      <details><summary>정제된 원본 이벤트 보기</summary><pre>${escapeHtml(JSON.stringify(detail.raw_event, null, 2))}</pre></details>
    </section>`;

  document.querySelector("#retryExplanation")?.addEventListener("click", () => retryExplanation(detail.event_id));
  document.querySelector("#saveReview")?.addEventListener("click", () => saveReview(detail.event_id));
}

async function openEvent(eventId) {
  state.selectedEventId = eventId;
  document.querySelector("#drawerBackdrop").classList.remove("hidden");
  const drawer = document.querySelector("#eventDrawer");
  drawer.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
  document.querySelector("#drawerContent").innerHTML = '<div class="empty-state"><div class="empty-icon">◎</div><p>판정 근거를 불러오는 중입니다.</p></div>';
  try {
    const detail = await api(`/api/v1/events/${encodeURIComponent(eventId)}`);
    const timeline = await api(`/api/v1/actors/${encodeURIComponent(detail.actor_id)}/timeline?limit=30`);
    renderDetail(detail, timeline);
  } catch (error) {
    document.querySelector("#drawerContent").innerHTML = `<div class="banner">${escapeHtml(error.message)}</div>`;
  }
}

function closeDrawer() {
  state.selectedEventId = null;
  document.querySelector("#drawerBackdrop").classList.add("hidden");
  const drawer = document.querySelector("#eventDrawer");
  drawer.classList.remove("open");
  drawer.setAttribute("aria-hidden", "true");
}

async function retryExplanation(eventId) {
  try {
    await api(`/api/v1/events/${encodeURIComponent(eventId)}/explanation`, { method: "POST" });
    await openEvent(eventId);
    await loadDashboard(false);
  } catch (error) { alert(error.message); }
}

async function saveReview(eventId) {
  const reviewStatus = document.querySelector("#reviewStatus").value;
  const note = document.querySelector("#reviewNote").value;
  try {
    await api(`/api/v1/events/${encodeURIComponent(eventId)}/review`, {
      method: "PATCH", body: JSON.stringify({ status: reviewStatus, note }),
    });
    const button = document.querySelector("#saveReview");
    button.textContent = "저장됨";
    setTimeout(() => { button.textContent = "저장"; }, 1200);
  } catch (error) { alert(error.message); }
}

function eventQuery() {
  const params = new URLSearchParams({ limit: "100" });
  params.set("suspicious_only", document.querySelector("#suspiciousFilter").checked ? "true" : "false");
  const policy = document.querySelector("#policyFilter").value;
  const actor = document.querySelector("#actorFilter").value.trim();
  if (policy) params.set("policy_stage", policy);
  if (actor) params.set("actor_id", actor);
  return params.toString();
}

async function loadDashboard(showLoading = true) {
  if (showLoading) document.querySelector("#apiState").className = "status-chip loading";
  try {
    const [health, overview, events] = await Promise.all([
      api("/health/ready"), api("/api/v1/overview?hours=24"), api(`/api/v1/events?${eventQuery()}`),
    ]);
    if (health.status !== "ready") throw new Error("API가 아직 준비되지 않았습니다.");
    renderOverview(overview);
    renderEvents(events);
    setConnection(true);
    if (state.selectedEventId) await openEvent(state.selectedEventId);
  } catch (error) {
    setConnection(false, error.status === 401 ? "API 키가 필요하거나 올바르지 않습니다. 상단의 API 키 버튼에서 설정하세요." : error.message);
  }
}

document.querySelector("#refreshButton").addEventListener("click", () => loadDashboard());
document.querySelector("#suspiciousFilter").addEventListener("change", () => loadDashboard(false));
document.querySelector("#policyFilter").addEventListener("change", () => loadDashboard(false));
document.querySelector("#actorFilter").addEventListener("change", () => loadDashboard(false));
document.querySelector("#closeDrawer").addEventListener("click", closeDrawer);
document.querySelector("#drawerBackdrop").addEventListener("click", closeDrawer);
document.addEventListener("keydown", event => { if (event.key === "Escape") closeDrawer(); });

const keyDialog = document.querySelector("#apiKeyDialog");
document.querySelector("#apiKeyButton").addEventListener("click", () => {
  document.querySelector("#apiKeyInput").value = state.apiKey;
  keyDialog.showModal();
});
keyDialog.addEventListener("close", () => {
  if (keyDialog.returnValue === "save") {
    state.apiKey = document.querySelector("#apiKeyInput").value.trim();
    sessionStorage.setItem("ztnaApiKey", state.apiKey);
    loadDashboard();
  } else if (keyDialog.returnValue === "clear") {
    state.apiKey = "";
    sessionStorage.removeItem("ztnaApiKey");
    loadDashboard();
  }
});

loadDashboard();
state.refreshTimer = setInterval(() => loadDashboard(false), 15000);
