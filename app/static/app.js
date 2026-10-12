"use strict";

const roleSelect = document.getElementById("role");
const roleBrief = document.getElementById("role-brief");
const fileInput = document.getElementById("file");
const fileName = document.getElementById("file-name");
const form = document.getElementById("screen-form");
const submitButton = document.getElementById("submit");
const result = document.getElementById("result");
const MAX_BYTES = 5 * 1024 * 1024;

let roles = [];

// Build DOM with textContent only, so text taken from a resume can never be interpreted as HTML.
function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "class") el.className = value;
    else el.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child !== null && child !== undefined && child !== false) el.append(child);
  }
  return el;
}

function showMessage(text, className) {
  result.replaceChildren(h("p", { class: className, role: className === "error" ? "alert" : "status" }, text));
}

async function loadRoles() {
  try {
    const response = await fetch("/api/roles");
    if (!response.ok) throw new Error();
    roles = await response.json();
    roleSelect.replaceChildren(...roles.map((r) => new Option(r.title, r.id)));
    renderRoleBrief();
  } catch {
    showMessage("Could not load the list of roles. Reload the page to try again.", "error");
  }
}

function renderRoleBrief() {
  const role = roles.find((r) => r.id === roleSelect.value);
  roleBrief.textContent = role ? `Must have: ${role.must_have.join(", ")}.` : "";
}

function scoreClass(decision) {
  return decision === "Shortlisted" ? "pass" : decision === "Needs review" ? "review" : "fail";
}

function evidenceNote(hit) {
  const parts = [];
  if (hit.matched_term.toLowerCase() !== hit.label.toLowerCase()) parts.push(`found as ${hit.matched_term}`);
  if (hit.evidence === "listed") parts.push("listed in skills only");
  if (hit.evidence === "demonstrated") parts.push("used in experience or projects");
  return parts.join(", ");
}

function skillGroup(title, group, total) {
  const found = group.matched.length;
  const items = [
    ...group.matched.map((hit) =>
      h("li", {}, h("mark", { class: "hit" }, hit.label), h("span", { class: "note" }, evidenceNote(hit)))
    ),
    ...group.missing.map((label) =>
      h("li", {}, h("span", { class: "miss" }, label), h("span", { class: "note" }, "not found"))
    ),
  ];
  return h("section", {}, h("h3", {}, `${title}: ${found} of ${total} found`), h("ul", { class: "skills" }, items));
}

function ledger(breakdown, score) {
  const rows = breakdown.map((item) => {
    const fill = h("span", { class: "fill" });
    fill.style.width = `${item.max ? (item.earned / item.max) * 100 : 0}%`;
    return h(
      "tr",
      {},
      h("th", { scope: "row" }, item.label),
      h("td", { class: "num" }, `${item.earned} of ${item.max}`),
      h("td", {}, h("span", { class: "bar" }, fill)),
      h("td", { class: "detail" }, item.detail)
    );
  });
  return h(
    "table",
    {},
    h("caption", { class: "sr-only" }, `Score breakdown, ${score} out of 100`),
    h("thead", { class: "sr-only" }, h("tr", {}, ...["Component", "Points", "Share of maximum", "Detail"].map((t) => h("th", {}, t)))),
    h("tbody", {}, rows)
  );
}

function facts(candidate, meta) {
  const rows = [
    ["Name", candidate.name],
    ["Email", candidate.email],
    ["Phone", candidate.phone],
    ["Links", candidate.links.join(", ")],
    ["Experience", `${candidate.experience_years} years`],
    ["Education", candidate.education],
    ["Text source", meta.extraction === "ocr" ? "OCR of scanned pages" : "Text layer of the PDF"],
  ].filter(([, value]) => value);
  return h("dl", { class: "facts" }, rows.flatMap(([term, value]) => [h("dt", {}, term), h("dd", {}, value)]));
}

function render(data) {
  const cls = scoreClass(data.decision);
  const s = data.skills;
  const nodes = [
    h(
      "header",
      {},
      h("h2", { class: `verdict ${cls}` }, data.decision),
      h("p", { class: "verdict-line" }, `${data.role.title}: ${data.score} out of 100. Must-have skills found: ${data.summary.must_have_matched} of ${data.summary.must_have_total}.`)
    ),
    data.warnings.length ? h("div", { class: "warnings", role: "note" }, data.warnings.map((w) => h("p", {}, w))) : null,
    h("section", {}, h("h3", {}, "How the score adds up"), ledger(data.breakdown, data.score)),
    skillGroup("Must-have skills", s.must_have, s.must_have.matched.length + s.must_have.missing.length),
    skillGroup("Good-to-have skills", s.good_to_have, s.good_to_have.matched.length + s.good_to_have.missing.length),
    skillGroup("Role keywords", s.keywords, s.keywords.matched.length + s.keywords.missing.length),
    h("section", {}, h("h3", {}, "Why this decision"), h("ul", { class: "reasons" }, data.reasons.map((r) => h("li", {}, r)))),
    h("section", {}, h("h3", {}, "Candidate details"), facts(data.candidate, data.meta)),
    s.other_detected.length
      ? h("section", {}, h("h3", {}, "Other skills detected"), h("ul", { class: "chips" }, s.other_detected.map((x) => h("li", {}, x))))
      : null,
  ];
  result.replaceChildren(...nodes.filter(Boolean));
  result.focus();
}

roleSelect.addEventListener("change", renderRoleBrief);

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  fileName.textContent = file ? file.name : "Choose a PDF or drop it here";
  fileInput.closest(".drop").classList.toggle("has-file", Boolean(file));
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) return showMessage("Choose a PDF resume first.", "error");
  if (file.size > MAX_BYTES) return showMessage("This file is larger than 5 MB. Upload a smaller PDF.", "error");

  submitButton.disabled = true;
  submitButton.textContent = "Screening…";
  showMessage("Reading the resume…", "empty");
  try {
    const response = await fetch("/api/screen", { method: "POST", body: new FormData(form) });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = typeof payload.detail === "string" ? payload.detail : "Something went wrong on the server.";
      return showMessage(`Could not screen this resume. ${detail}`, "error");
    }
    render(payload);
  } catch {
    showMessage("Could not reach the server. Check your connection and try again.", "error");
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Screen resume";
  }
});

loadRoles();
