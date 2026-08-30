"use strict";

const API_ROOT = "/api/v1/competition/demo";
const DECISION_LABELS = Object.freeze({
  COMPLIANT: "局部符合",
  NON_COMPLIANT: "局部不符合",
  INSUFFICIENT_INFORMATION: "局部信息不足",
});

const REASON_PRESENTATIONS = Object.freeze({
  NUMERIC_LIMIT_SATISFIED: "当前方案控制值达到该项规范限值要求。",
  NUMERIC_LIMIT_VIOLATED: "当前方案控制值与该项规范限值发生局部冲突。",
  REQUIREMENT_DECOMPOSITION_UNRESOLVED: "现有证据不足以形成该局部规范要求的确定性违规结论。",
});

const state = { metadata: null, runningCaseId: null };

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function shortId(value) {
  return value.length > 16 ? `${value.slice(0, 8)}…${value.slice(-6)}` : value;
}

function setEnvironmentLoading() {
  const summary = document.querySelector("#environment-summary");
  summary.replaceChildren(
    element("span", "status-indicator is-loading"),
    element("strong", "", "正在验证运行时权威…"),
  );
  document.querySelector("#refresh-status").disabled = true;
}

function renderReadiness(metadata) {
  const ready = metadata.status === "READY";
  const summary = document.querySelector("#environment-summary");
  summary.replaceChildren(
    element("span", `status-indicator ${ready ? "is-ready" : "is-blocked"}`),
    element("strong", "", ready ? "READY · 资格化运行时可用" : "NOT_READY · 运行时权威不可用"),
  );

  const checks = document.querySelector("#readiness-checks");
  checks.replaceChildren();
  metadata.readiness_checks.forEach((check) => {
    const card = element("article", `check-card ${check.ready ? "is-ready" : "is-blocked"}`);
    const header = element("div", "check-card-header");
    header.append(
      element("span", "check-symbol", check.ready ? "✓" : "!"),
      element("strong", "", check.label),
    );
    card.append(header, element("p", "", check.detail));
    checks.append(card);
  });
  document.querySelector("#remediation").hidden = ready;
  document.querySelector("#refresh-status").disabled = false;
}

function renderPlans(metadata) {
  const grid = document.querySelector("#plan-grid");
  grid.replaceChildren();
  metadata.plans.forEach((plan, index) => {
    const card = element("article", "plan-card");
    const top = element("div", "plan-card-top");
    top.append(
      element("span", "plan-index", `0${index + 1}`),
      element("span", `verified-badge ${plan.verified ? "is-ready" : "is-blocked"}`, plan.verified ? "字节已验证" : "验证失败"),
    );
    card.append(
      top,
      element("h3", "", plan.display_name),
      element("p", "document-id", `Document · ${shortId(plan.document_id)}`),
      element("p", "plan-meta", `${plan.case_count} 个固定局部审查目标`),
    );
    grid.append(card);
  });
}

function renderCases(metadata) {
  const ready = metadata.status === "READY";
  const grid = document.querySelector("#case-grid");
  grid.replaceChildren();
  metadata.cases.forEach((reviewCase) => {
    const card = element("article", "case-card");
    card.dataset.caseId = reviewCase.case_id;
    const eyebrow = element("div", "case-eyebrow");
    eyebrow.append(
      element("span", "case-id", reviewCase.case_id),
      element("span", "case-location", `第 ${reviewCase.page_number} 页`),
    );
    const citation = element("p", "case-citation", `${reviewCase.standard_code} · 第 ${reviewCase.article_number} 条`);
    const button = element("button", "run-button", "运行局部审查");
    button.type = "button";
    button.disabled = !ready;
    button.addEventListener("click", () => runCase(reviewCase.case_id, button));
    card.append(
      eyebrow,
      element("h3", "", reviewCase.label),
      element("p", "case-description", reviewCase.description),
      citation,
      button,
    );
    grid.append(card);
  });
}

function renderMetadata(metadata) {
  state.metadata = metadata;
  document.querySelector("#sample-mode-notice").textContent = metadata.sample_mode_notice;
  document.querySelector("#scope-notice").textContent = metadata.scope_notice;
  renderReadiness(metadata);
  renderPlans(metadata);
  renderCases(metadata);
}

function factText(fact) {
  const value = fact.normalized_value;
  if (value === null || value === undefined) return fact.source_text;
  return `${String(value)}${fact.unit ? ` ${fact.unit}` : ""}`;
}

function humanExplanation(result) {
  return REASON_PRESENTATIONS[result.reason_code] || result.local_explanation;
}

function missingInformationPresentation(result, item) {
  if (
    result.decision === "INSUFFICIENT_INFORMATION"
    && result.reason_code === "REQUIREMENT_DECOMPOSITION_UNRESOLVED"
  ) {
    return "当前证据尚不足以确定一个可进行确定性比较的单一、无条件规范要求。";
  }
  return item.detail || item.message || item.code || "需要补充局部证据";
}

function evidencePanel(title, kicker, text, metadataLines, className) {
  const panel = element("article", `evidence-panel ${className}`);
  panel.append(element("p", "panel-kicker", kicker), element("h3", "", title));
  const quote = element("blockquote", "evidence-quote", text);
  panel.append(quote);
  const metadata = element("dl", "evidence-meta");
  metadataLines.forEach(([term, description]) => {
    metadata.append(element("dt", "", term), element("dd", "", description));
  });
  panel.append(metadata);
  return panel;
}

function provenanceDetails(result) {
  const details = element("details", "provenance");
  details.append(element("summary", "", "技术溯源信息"));
  const list = element("dl", "provenance-grid");
  const values = [
    ["Finding ID", result.technical_provenance.finding_id],
    ["Comparison ID", result.technical_provenance.comparison_id],
    ["Reason Code", result.reason_code],
    ["Decision Scope", result.technical_provenance.decision_scope],
    ["Document ID", result.technical_provenance.document_id],
    ["PDF SHA-256", result.technical_provenance.document_sha256],
    ["Standard ID", result.technical_provenance.standard_id],
    ["Source SHA-256", result.technical_provenance.standard_source_sha256],
    ["Evidence ID", result.technical_provenance.evidence_id],
    ["Requirement ID", result.technical_provenance.requirement_id],
    ["ReviewUnit ID", result.technical_provenance.review_unit_id],
    ["PlanFact IDs", result.technical_provenance.plan_fact_ids.join(" · ") || "—"],
  ];
  if (result.missing_information.length) {
    values.push(["Raw Missing Information", result.missing_information.map((item) => item.detail || item.message || item.code).filter(Boolean).join(" · ")]);
  }
  values.push(["Raw C.3 Summary", result.summary]);
  values.forEach(([term, value]) => list.append(element("dt", "", term), element("dd", "", value)));
  details.append(list);
  return details;
}

function renderResult(result) {
  const expectedLabel = DECISION_LABELS[result.decision];
  if (!expectedLabel || expectedLabel !== result.decision_label) {
    renderSafeFailure("返回的局部判断标签未通过一致性校验。");
    return;
  }

  const section = document.querySelector("#result-section");
  section.hidden = false;
  document.querySelector("#result-message").textContent = result.message;
  const content = document.querySelector("#result-content");
  content.replaceChildren();
  const presentation = humanExplanation(result);

  const resultHeader = element("article", `finding-header decision-${result.decision.toLowerCase()}`);
  const decision = element("div", "decision-lockup");
  decision.append(
    element("span", "decision-overline", "LOCAL DECISION"),
    element("strong", "decision-label", expectedLabel),
    element("span", "decision-code", result.decision),
  );
  const explanation = element("div", "finding-explanation");
  explanation.append(
    element("p", "finding-context", "当前局部比较"),
    element("p", "summary", presentation),
    element("p", "local-explanation", "判断依据来自下方方案原文、规范原文与实时确定性比较。"),
  );
  resultHeader.append(decision, explanation);

  const factValues = result.plan_evidence.plan_facts.map(factText).join(" · ") || "未形成结构化 PlanFact";
  const planPanel = evidencePanel(
    "方案依据",
    `第 ${result.plan_evidence.page_number} 页 · ${result.plan_evidence.document_display_name}`,
    result.plan_evidence.exact_text,
    [["方案事实", factValues], ["引用位置", `${result.plan_evidence.char_start}–${result.plan_evidence.char_end}`]],
    "plan-evidence",
  );
  const comparisonPanel = element("article", "comparison-panel");
  comparisonPanel.append(element("p", "panel-kicker", "DETERMINISTIC COMPARISON"), element("h3", "", "局部比较"));
  if (result.plan_evidence.plan_facts.length && result.reason_code.startsWith("NUMERIC_LIMIT_")) {
    const facts = element("div", "comparison-facts");
    facts.append(element("span", "comparison-fact-label", "方案结构化事实"));
    result.plan_evidence.plan_facts.forEach((fact) => {
      facts.append(element("strong", "comparison-fact-value", factText(fact)));
    });
    facts.append(
      element("span", "comparison-versus", "对照"),
      element("strong", "comparison-standard", `${result.standard_evidence.standard_code} 第 ${result.standard_evidence.article_number} 条`),
    );
    comparisonPanel.append(facts);
  } else {
    comparisonPanel.append(element("p", "comparison-unresolved", "当前证据未形成可确定比较的结构化事实。"));
  }
  comparisonPanel.append(
    element("div", "comparison-result", expectedLabel),
    element("p", "comparison-reason", presentation),
    element("p", "scope-code", result.decision_scope),
  );
  const standardPanel = evidencePanel(
    "规范依据",
    `${result.standard_evidence.standard_code} · 第 ${result.standard_evidence.article_number} 条`,
    result.standard_evidence.exact_requirement_text,
    [["规范页码", `${result.standard_evidence.page_start}–${result.standard_evidence.page_end}`], ["证据类型", "资格化规范要求"]],
    "standard-evidence",
  );
  const trace = element("div", "evidence-trace");
  trace.append(planPanel, comparisonPanel, standardPanel);

  content.append(resultHeader);
  if (result.missing_information.length) {
    const missing = element("section", "missing-information");
    missing.append(element("h3", "", "仍需补充的信息"));
    const list = element("ul", "");
    result.missing_information.forEach((item) => {
      list.append(element("li", "", missingInformationPresentation(result, item)));
    });
    missing.append(list);
    content.append(missing);
  }
  content.append(trace);
  content.append(element("p", "result-scope-notice", result.scope_notice), provenanceDetails(result));
  section.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderSafeFailure(detail) {
  const section = document.querySelector("#result-section");
  section.hidden = false;
  document.querySelector("#result-message").textContent = "SAFE_FAILURE";
  const content = document.querySelector("#result-content");
  const panel = element("article", "safe-failure");
  panel.append(
    element("h3", "", "系统已安全停止，本次未生成局部审查结果。"),
    element("p", "", detail),
  );
  content.replaceChildren(panel);
  section.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function runCase(caseId, button) {
  if (state.runningCaseId) return;
  state.runningCaseId = caseId;
  const originalText = button.textContent;
  button.disabled = true;
  button.textContent = "正在实时重建…";
  document.querySelectorAll(".run-button").forEach((item) => { item.disabled = true; });
  try {
    const response = await fetch(`${API_ROOT}/cases/${encodeURIComponent(caseId)}/run`, {
      method: "POST",
      headers: { Accept: "application/json" },
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "局部审查服务不可用。");
    renderResult(payload);
  } catch (error) {
    renderSafeFailure(error instanceof Error ? error.message : "局部审查服务不可用。");
  } finally {
    state.runningCaseId = null;
    button.textContent = originalText;
    document.querySelectorAll(".run-button").forEach((item) => {
      item.disabled = state.metadata?.status !== "READY";
    });
  }
}

async function loadMetadata() {
  setEnvironmentLoading();
  try {
    const response = await fetch(API_ROOT, { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error("环境检查接口不可用。");
    renderMetadata(await response.json());
  } catch (error) {
    renderSafeFailure(error instanceof Error ? error.message : "后端服务不可用。");
    document.querySelector("#environment-summary").replaceChildren(
      element("span", "status-indicator is-blocked"),
      element("strong", "", "NOT_READY · 后端服务不可用"),
    );
    document.querySelector("#remediation").hidden = false;
    document.querySelector("#refresh-status").disabled = false;
  }
}

document.querySelector("#refresh-status").addEventListener("click", loadMetadata);
loadMetadata();
