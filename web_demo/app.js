const DATA_URL = "./data/demo_val_2021-11-28.json";

const TIER_COLORS = {
  "Early / On Time": "#2D6A4F",
  "Minor (0-15)": "#74A57F",
  "Moderate (15-60)": "#D9A441",
  "Heavy (60-120)": "#E76F51",
  "Severe (120-240)": "#C44536",
  "Extreme (240-720)": "#8D2B3A",
  "Ultra (720+)": "#4A0D1A",
};

const CORRECTNESS_COLORS = {
  "Exact Tier Match": "#2D6A4F",
  "Tier Miss": "#C44536",
};

const ALERT_RESULT_COLORS = {
  TP: "#2D6A4F",
  FP: "#C44536",
  FN: "#B08D57",
  TN: "#8FA8A3",
};

const SEVERITY_FILTERS = {
  all: -1,
  heavy_plus: 3,
  severe_plus: 4,
  extreme_plus: 5,
  ultra_only: 6,
};

const state = {
  data: null,
  rows: [],
  snapshotIndex: 0,
  horizonFilter: "all",
  severityFilter: "all",
  mapMode: "severity",
  search: "",
  sortOrder: "severity_desc",
  pageSize: 25,
  page: 1,
  playing: false,
  timer: null,
  selectedKey: null,
};

const el = {};

document.addEventListener("DOMContentLoaded", async () => {
  cacheElements();
  bindControls();
  await loadData();
  renderAll();
});

function cacheElements() {
  [
    "dataset-meta", "play-toggle", "step-back", "step-forward", "timeline-slider",
    "snapshot-label", "snapshot-subtitle", "map-mode", "horizon-filter",
    "severity-filter", "sort-order", "search-box", "page-size",
    "metric-flights", "metric-alerts", "metric-match", "metric-mae", "metric-prob",
    "map-caption", "legend-row", "map-plot", "table-body", "page-prev", "page-next",
    "page-label", "selection-empty", "selection-card",
  ].forEach((id) => {
    el[id] = document.getElementById(id);
  });
  el["nav-tabs"] = [...document.querySelectorAll(".nav-tab")];
  el["tab-panels"] = [...document.querySelectorAll(".tab-panel")];
  el["jump-tabs"] = [...document.querySelectorAll("[data-jump-tab]")];
}

function bindControls() {
  el["nav-tabs"].forEach((button) => {
    button.addEventListener("click", () => activateTab(button.dataset.tab));
  });
  el["jump-tabs"].forEach((button) => {
    button.addEventListener("click", () => activateTab(button.dataset.jumpTab));
  });
  el["play-toggle"].addEventListener("click", togglePlay);
  el["step-back"].addEventListener("click", () => stepTimeline(-1));
  el["step-forward"].addEventListener("click", () => stepTimeline(1));
  el["timeline-slider"].addEventListener("input", (evt) => {
    state.snapshotIndex = Number(evt.target.value);
    state.page = 1;
    stopPlay();
    renderAll();
  });
  el["map-mode"].addEventListener("change", (evt) => {
    state.mapMode = evt.target.value;
    renderLegend();
    renderMap();
  });
  el["horizon-filter"].addEventListener("change", (evt) => {
    state.horizonFilter = evt.target.value;
    state.page = 1;
    renderAll();
  });
  el["severity-filter"].addEventListener("change", (evt) => {
    state.severityFilter = evt.target.value;
    state.page = 1;
    renderAll();
  });
  el["sort-order"].addEventListener("change", (evt) => {
    state.sortOrder = evt.target.value;
    state.page = 1;
    renderTable();
    renderMap();
  });
  el["search-box"].addEventListener("input", (evt) => {
    state.search = evt.target.value.trim().toLowerCase();
    state.page = 1;
    renderAll();
  });
  el["page-size"].addEventListener("change", (evt) => {
    state.pageSize = Number(evt.target.value);
    state.page = 1;
    renderTable();
  });
  el["page-prev"].addEventListener("click", () => {
    state.page = Math.max(1, state.page - 1);
    renderTable();
  });
  el["page-next"].addEventListener("click", () => {
    const pageCount = Math.max(1, Math.ceil(getFilteredRows().length / state.pageSize));
    state.page = Math.min(pageCount, state.page + 1);
    renderTable();
  });
}

async function loadData() {
  const response = await fetch(DATA_URL);
  if (!response.ok) {
    throw new Error(`Could not load demo data from ${DATA_URL}`);
  }
  state.data = await response.json();
  state.rows = state.data.rowsData.map(expandRow);
  el["dataset-meta"].textContent = `${state.data.meta.date} | ${state.data.meta.split.toUpperCase()} | ${numberWithCommas(state.data.meta.rows)} replay rows`;
  el["timeline-slider"].max = String(state.data.snapshotTimes.length - 1);
  renderLegend();
}

function togglePlay() {
  if (state.playing) {
    stopPlay();
    return;
  }
  state.playing = true;
  el["play-toggle"].textContent = "Pause";
  state.timer = window.setInterval(() => {
    state.snapshotIndex = (state.snapshotIndex + 1) % state.data.snapshotTimes.length;
    el["timeline-slider"].value = String(state.snapshotIndex);
    renderAll();
  }, 900);
}

function stopPlay() {
  state.playing = false;
  el["play-toggle"].textContent = "Play";
  if (state.timer) {
    window.clearInterval(state.timer);
    state.timer = null;
  }
}

function stepTimeline(delta) {
  stopPlay();
  const max = state.data.snapshotTimes.length - 1;
  state.snapshotIndex = Math.min(max, Math.max(0, state.snapshotIndex + delta));
  el["timeline-slider"].value = String(state.snapshotIndex);
  state.page = 1;
  renderAll();
}

function renderAll() {
  renderTimeline();
  renderMetrics();
  renderMap();
  renderTable();
  renderSelection();
}

function activateTab(tabName) {
  el["nav-tabs"].forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tabName);
  });
  el["tab-panels"].forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.panel === tabName);
  });
  if (tabName === "flight-demo" && state.data) {
    window.setTimeout(() => {
      renderAll();
      if (window.Plotly && el["map-plot"]) {
        Plotly.Plots.resize(el["map-plot"]);
      }
    }, 80);
  }
}

function renderTimeline() {
  const summary = state.data.snapshotSummary[state.snapshotIndex];
  el["timeline-slider"].value = String(state.snapshotIndex);
  el["snapshot-label"].textContent = summary.snapshotLabel;
  el["snapshot-subtitle"].textContent = `${numberWithCommas(summary.flights)} flights | ${(summary.tierMatchRate * 100).toFixed(1)}% exact tier match | ${summary.meanAbsErr.toFixed(1)} min MAE`;
}

function getFilteredRows() {
  const severityThreshold = SEVERITY_FILTERS[state.severityFilter];
  let rows = state.rows.filter((row) => row.snap === state.snapshotIndex);

  if (state.horizonFilter !== "all") {
    rows = rows.filter((row) => String(row.horizon) === state.horizonFilter);
  }
  if (severityThreshold >= 0) {
    rows = rows.filter((row) => row.predTierOrder >= severityThreshold);
  }
  if (state.search) {
    rows = rows.filter((row) => {
      const haystack = `${row.flightId} ${row.airline} ${row.tail} ${row.route}`.toLowerCase();
      return haystack.includes(state.search);
    });
  }

  rows = [...rows];
  rows.sort(sorterFor(state.sortOrder));
  return rows;
}

function expandRow(raw) {
  if (raw.flightId !== undefined) {
    return raw;
  }
  const coords = state.data.meta.airportCoords || {};
  const originCoords = coords[raw.o] || [null, null];
  const destCoords = coords[raw.d] || [null, null];
  const predTier = state.data.meta.tierLabels[raw.po];
  const actualTier = state.data.meta.tierLabels[raw.ao];
  return {
    snap: raw.snap,
    flightId: raw.f,
    origin: raw.o,
    dest: raw.d,
    route: `${raw.o} -> ${raw.d}`,
    dep: raw.dep,
    arr: raw.arr,
    tail: raw.tail,
    airline: raw.air,
    horizon: raw.h,
    horizonLabel: state.data.meta.horizonLabels[String(raw.h)] || state.data.meta.horizonLabels[raw.h] || String(raw.h),
    hoursToDeparture: raw.h2d,
    pred: raw.p,
    predTier,
    predTierOrder: raw.po,
    severeProb: raw.sp,
    severeAlert: raw.sa,
    actual: raw.a,
    actualTier,
    actualTierOrder: raw.ao,
    actualSevere: raw.a >= 120,
    absErr: raw.ae,
    tierMatch: raw.tm,
    correctness: raw.tm ? "Exact Tier Match" : "Tier Miss",
    correctnessOrder: raw.tm ? 0 : 1,
    alertResult: raw.ar,
    alertOrder: raw.alertOrder,
    originLat: originCoords[0],
    originLon: originCoords[1],
    destLat: destCoords[0],
    destLon: destCoords[1],
  };
}

function sorterFor(key) {
  if (key === "severe_prob_desc") {
    return (a, b) => compareDesc(a.severeProb, b.severeProb) || compareDesc(a.pred, b.pred);
  }
  if (key === "abs_err_desc") {
    return (a, b) => compareDesc(a.absErr, b.absErr) || compareDesc(a.pred, b.pred);
  }
  if (key === "actual_desc") {
    return (a, b) => compareDesc(a.actual, b.actual) || compareDesc(a.pred, b.pred);
  }
  if (key === "route_az") {
    return (a, b) => a.route.localeCompare(b.route) || compareDesc(a.pred, b.pred);
  }
  if (key === "airline_az") {
    return (a, b) => a.airline.localeCompare(b.airline) || a.route.localeCompare(b.route);
  }
  return (a, b) =>
    compareDesc(a.predTierOrder, b.predTierOrder) ||
    compareDesc(a.severeProb, b.severeProb) ||
    compareDesc(a.pred, b.pred);
}

function renderMetrics() {
  const rows = getFilteredRows();
  const severeAlerts = rows.filter((row) => row.severeAlert).length;
  const exactMatches = rows.filter((row) => row.tierMatch).length;
  const meanAbsErr = rows.length ? average(rows.map((row) => row.absErr)) : 0;
  const maxProb = rows.length ? Math.max(...rows.map((row) => row.severeProb)) : 0;

  el["metric-flights"].textContent = numberWithCommas(rows.length);
  el["metric-alerts"].textContent = numberWithCommas(severeAlerts);
  el["metric-match"].textContent = `${rows.length ? ((exactMatches / rows.length) * 100).toFixed(1) : "0.0"}%`;
  el["metric-mae"].textContent = `${meanAbsErr.toFixed(1)} min`;
  el["metric-prob"].textContent = maxProb.toFixed(3);
}

function renderLegend() {
  const items = [];
  const palette = state.mapMode === "severity"
    ? TIER_COLORS
    : state.mapMode === "correctness"
      ? CORRECTNESS_COLORS
      : ALERT_RESULT_COLORS;

  Object.entries(palette).forEach(([label, color]) => {
    items.push(`
      <span class="legend-chip">
        <span class="legend-swatch" style="background:${color}"></span>
        ${label}
      </span>
    `);
  });
  el["legend-row"].innerHTML = items.join("");
}

function renderMap() {
  const flightPanel = document.querySelector('[data-panel="flight-demo"]');
  if (!flightPanel || !flightPanel.classList.contains("active")) {
    return;
  }
  const rows = getFilteredRows();
  const routeRows = rows.slice(0, 45);
  const airportMap = new Map();

  rows.forEach((row) => {
    const key = row.origin;
    const entry = airportMap.get(key) || {
      airport: row.origin,
      lat: row.originLat,
      lon: row.originLon,
      count: 0,
      maxPredTierOrder: -1,
      worstAlertOrder: 99,
      mismatchCount: 0,
      topColor: "#8FA8A3",
      topLabel: "",
    };
    entry.count += 1;
    entry.maxPredTierOrder = Math.max(entry.maxPredTierOrder, row.predTierOrder);
    entry.worstAlertOrder = Math.min(entry.worstAlertOrder, row.alertOrder);
    entry.mismatchCount += row.tierMatch ? 0 : 1;
    entry.topColor = rowColor(row);
    entry.topLabel = row.predTier;
    airportMap.set(key, entry);
  });

  const traces = [];
  routeRows.forEach((row) => {
    if (!Number.isFinite(row.originLat) || !Number.isFinite(row.destLat)) {
      return;
    }
    traces.push({
      type: "scattergeo",
      mode: "lines",
      lon: [row.originLon, row.destLon],
      lat: [row.originLat, row.destLat],
      line: { width: rowKey(row) === state.selectedKey ? 4 : 1.8, color: rowColor(row) },
      opacity: rowKey(row) === state.selectedKey ? 0.95 : 0.24,
      hoverinfo: "skip",
      showlegend: false,
    });
  });

  const airportValues = [...airportMap.values()];
  traces.push({
    type: "scattergeo",
    mode: "markers+text",
    lon: airportValues.map((item) => item.lon),
    lat: airportValues.map((item) => item.lat),
    text: airportValues.map((item) => item.airport),
    textposition: "top center",
    marker: {
      size: airportValues.map((item) => 10 + Math.sqrt(item.count) * 2.2),
      color: airportValues.map((item) => item.topColor),
      opacity: 0.88,
      line: { color: "#F7F2E8", width: 1.2 },
    },
    hovertemplate: airportValues.map((item) =>
      `<b>${item.airport}</b><br>${item.count} flights in view<extra></extra>`
    ),
    showlegend: false,
  });

  const selected = rows.find((row) => rowKey(row) === state.selectedKey);
  if (selected && Number.isFinite(selected.originLat) && Number.isFinite(selected.destLat)) {
    traces.push({
      type: "scattergeo",
      mode: "lines+markers",
      lon: [selected.originLon, selected.destLon],
      lat: [selected.originLat, selected.destLat],
      marker: { size: 8, color: "#16202A" },
      line: { width: 4, color: "#16202A" },
      hovertemplate: `<b>${selected.route}</b><br>Pred ${selected.pred} min<extra></extra>`,
      showlegend: false,
    });
  }

  el["map-caption"].textContent = mapCaption();
  Plotly.react(
    el["map-plot"],
    traces,
    {
      margin: { l: 0, r: 0, t: 0, b: 0 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      geo: {
        scope: "usa",
        projection: { type: "albers usa" },
        bgcolor: "rgba(0,0,0,0)",
        lakecolor: "#f7f2e8",
        landcolor: "#efe7d8",
        subunitcolor: "#d7cec2",
        countrycolor: "#d7cec2",
        coastlinecolor: "#d7cec2",
        showland: true,
        showsubunits: true,
      },
    },
    { responsive: true, displayModeBar: false }
  );
}

function rowColor(row) {
  if (state.mapMode === "correctness") {
    return CORRECTNESS_COLORS[row.correctness];
  }
  if (state.mapMode === "alerts") {
    return ALERT_RESULT_COLORS[row.alertResult];
  }
  return TIER_COLORS[row.predTier] || "#8FA8A3";
}

function mapCaption() {
  if (state.mapMode === "correctness") {
    return "Top routes and airports colored by exact predicted-tier correctness.";
  }
  if (state.mapMode === "alerts") {
    return "Top routes and airports colored by severe-alert outcome using the 0.60 threshold.";
  }
  return "Top routes and airports colored by predicted delay tier at the selected snapshot.";
}

function renderTable() {
  const rows = getFilteredRows();
  const pageCount = Math.max(1, Math.ceil(rows.length / state.pageSize));
  state.page = Math.min(state.page, pageCount);
  const start = (state.page - 1) * state.pageSize;
  const pageRows = rows.slice(start, start + state.pageSize);

  el["page-label"].textContent = `Page ${state.page} / ${pageCount}`;
  el["table-body"].innerHTML = pageRows.map((row) => {
    const selectedClass = rowKey(row) === state.selectedKey ? "selected" : "";
    return `
      <tr class="${selectedClass}" data-row-key="${rowKey(row)}">
        <td class="mono">${row.flightId}</td>
        <td>${escapeHtml(row.airline)}</td>
        <td class="mono">${escapeHtml(row.tail || "-")}</td>
        <td>${escapeHtml(row.route)}</td>
        <td>${escapeHtml(row.horizonLabel)}</td>
        <td>${escapeHtml(row.dep)}</td>
        <td>${escapeHtml(row.arr)}</td>
        <td>${row.pred.toFixed(1)}</td>
        <td><span class="mini-pill" style="background:${colorAlpha(TIER_COLORS[row.predTier], 0.12)};color:${TIER_COLORS[row.predTier]}">${escapeHtml(row.predTier)}</span></td>
        <td>${row.severeProb.toFixed(3)}</td>
        <td>${row.severeAlert ? "true" : "false"}</td>
        <td>${row.actual.toFixed(1)}</td>
        <td><span class="mini-pill" style="background:${colorAlpha(TIER_COLORS[row.actualTier], 0.12)};color:${TIER_COLORS[row.actualTier]}">${escapeHtml(row.actualTier)}</span></td>
        <td>${row.absErr.toFixed(1)}</td>
        <td><span class="mini-pill" style="background:${colorAlpha(CORRECTNESS_COLORS[row.correctness], 0.12)};color:${CORRECTNESS_COLORS[row.correctness]}">${escapeHtml(row.correctness)}</span></td>
        <td><span class="mini-pill" style="background:${colorAlpha(ALERT_RESULT_COLORS[row.alertResult], 0.12)};color:${ALERT_RESULT_COLORS[row.alertResult]}">${escapeHtml(row.alertResult)}</span></td>
      </tr>
    `;
  }).join("");

  [...el["table-body"].querySelectorAll("tr")].forEach((tr) => {
    tr.addEventListener("click", () => {
      state.selectedKey = tr.dataset.rowKey;
      renderMap();
      renderSelection();
      renderTable();
    });
  });
}

function renderSelection() {
  const rows = getFilteredRows();
  const selected = rows.find((row) => rowKey(row) === state.selectedKey);
  if (!selected) {
    el["selection-empty"].classList.remove("hidden");
    el["selection-card"].classList.add("hidden");
    el["selection-card"].innerHTML = "";
    return;
  }

  el["selection-empty"].classList.add("hidden");
  el["selection-card"].classList.remove("hidden");
  el["selection-card"].innerHTML = `
    <div class="selection-topline">
      <div>
        <div class="selection-route">${escapeHtml(selected.route)}</div>
        <div class="subtle-line">Flight ID ${selected.flightId} | ${escapeHtml(selected.airline)} | Tail ${escapeHtml(selected.tail || "-")}</div>
      </div>
      <div class="selection-pill" style="background:${colorAlpha(TIER_COLORS[selected.predTier], 0.12)};color:${TIER_COLORS[selected.predTier]}">
        ${escapeHtml(selected.predTier)}
      </div>
    </div>
    <div class="selection-pills">
      <span class="mini-pill" style="background:${colorAlpha(ALERT_RESULT_COLORS[selected.alertResult], 0.12)};color:${ALERT_RESULT_COLORS[selected.alertResult]}">
        ${escapeHtml(selected.alertResult)}
      </span>
      <span class="mini-pill" style="background:${colorAlpha(CORRECTNESS_COLORS[selected.correctness], 0.12)};color:${CORRECTNESS_COLORS[selected.correctness]}">
        ${escapeHtml(selected.correctness)}
      </span>
      <span class="mini-pill" style="background:#edf3f5;color:#1f4e5f">
        ${escapeHtml(selected.horizonLabel)}
      </span>
    </div>
    <div class="selection-grid">
      <div class="selection-stat"><span>Predicted Delay</span><strong>${selected.pred.toFixed(1)} min</strong></div>
      <div class="selection-stat"><span>Actual Delay</span><strong>${selected.actual.toFixed(1)} min</strong></div>
      <div class="selection-stat"><span>Severe Probability</span><strong>${selected.severeProb.toFixed(3)}</strong></div>
      <div class="selection-stat"><span>Absolute Error</span><strong>${selected.absErr.toFixed(1)} min</strong></div>
      <div class="selection-stat"><span>Scheduled Departure</span><strong>${escapeHtml(selected.dep)}</strong></div>
      <div class="selection-stat"><span>Scheduled Arrival</span><strong>${escapeHtml(selected.arr)}</strong></div>
    </div>
  `;
}

function rowKey(row) {
  return `${row.snap}|${row.flightId}|${row.horizon}`;
}

function numberWithCommas(value) {
  return new Intl.NumberFormat("en-US").format(value);
}

function compareDesc(a, b) {
  if (a === b) return 0;
  return a > b ? -1 : 1;
}

function average(values) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function colorAlpha(hex, alpha) {
  const raw = hex.replace("#", "");
  const bigint = Number.parseInt(raw, 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
