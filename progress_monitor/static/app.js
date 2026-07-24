(function () {
  "use strict";

  const STAGE_GROUPS = [
    { label: "2D Pose Estimation", keys: ["body", "full_body", "hand", "merged"] },
    { label: "3D Pipeline", keys: ["depth", "pose_3d", "fitted"] },
  ];

  const CHECK_SVG =
    '<svg viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">' +
    '<path d="M3 8.5L6.2 12L13 4" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"/></svg>';

  let latestData = null;
  let stageLabels = {};
  let stageFilter = null; // stage key currently being used as a "missing this" filter, or null

  const el = {
    projectFolder: document.getElementById("project-folder"),
    scanMeta: document.getElementById("scan-meta"),
    refreshBtn: document.getElementById("refresh-btn"),
    tiles: document.getElementById("tiles"),
    sidebarLinks: document.getElementById("sidebar-links"),
    summaryThead: document.querySelector("#summary-table thead"),
    summaryTbody: document.querySelector("#summary-table tbody"),
    tree: document.getElementById("tree"),
    search: document.getElementById("search"),
    statusFilter: document.getElementById("status-filter"),
    stageChip: document.getElementById("stage-chip"),
    expandAll: document.getElementById("expand-all"),
    collapseAll: document.getElementById("collapse-all"),
    emptyState: document.getElementById("empty-state"),
  };

  function fmtPct(n) {
    return `${Math.round(n)}%`;
  }

  function timeAgo(unixSeconds) {
    const seconds = Math.max(0, Math.round(Date.now() / 1000 - unixSeconds));
    if (seconds < 5) return "just now";
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.round(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.round(minutes / 60);
    return `${hours}h ago`;
  }

  function computeAggregate(recordings, stageKeys) {
    const agg = { total: recordings.length, fullyComplete: 0, byStage: {} };
    stageKeys.forEach((key) => (agg.byStage[key] = 0));
    for (const rec of recordings) {
      if (rec.completed === stageKeys.length) agg.fullyComplete += 1;
      for (const key of stageKeys) {
        if (rec.stages[key]) agg.byStage[key] += 1;
      }
    }
    return agg;
  }

  function allRecordings(data) {
    const out = [];
    for (const school of data.schools) {
      for (const participant of school.participants) {
        for (const rec of participant.recordings) {
          out.push({ school: school.name, participant: participant.name, ...rec });
        }
      }
    }
    return out;
  }

  function render(data) {
    latestData = data;
    stageLabels = Object.fromEntries(data.stages);
    const stageKeys = data.stages.map(([key]) => key);
    const flatRecordings = allRecordings(data);
    const globalAgg = computeAggregate(flatRecordings, stageKeys);

    el.projectFolder.textContent = data.project_folder;
    el.scanMeta.textContent = `${data.total_recordings} recordings · scanned ${timeAgo(
      data.generated_at
    )} (${data.scan_duration_ms} ms)`;

    renderTiles(globalAgg, stageKeys);
    renderSidebar(data, stageKeys);
    renderSummaryTable(data, stageKeys);
    renderTree(data, stageKeys);
    applyFilters();
  }

  function renderTiles(globalAgg, stageKeys) {
    const total = globalAgg.total || 1;
    const headlinePct = (globalAgg.fullyComplete / total) * 100;

    const tiles = [];
    tiles.push(`
      <button class="tile headline" data-tile="complete">
        <div class="tile-label">✓ Fully complete</div>
        <div class="tile-value">${globalAgg.fullyComplete}<span class="tile-sub"> / ${globalAgg.total}</span></div>
        <div class="tile-sub">${fmtPct(headlinePct)} of all recordings</div>
        <div class="bar-track"><div class="bar-fill" style="width:${headlinePct}%"></div></div>
      </button>
    `);

    for (const key of stageKeys) {
      const count = globalAgg.byStage[key];
      const pct = (count / total) * 100;
      tiles.push(`
        <button class="tile" data-tile="stage" data-stage="${key}">
          <div class="tile-label">${stageLabels[key]}</div>
          <div class="tile-value">${count}<span class="tile-sub"> / ${globalAgg.total}</span></div>
          <div class="tile-sub">${fmtPct(pct)} done</div>
          <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
        </button>
      `);
    }
    el.tiles.innerHTML = tiles.join("");

    el.tiles.querySelectorAll(".tile").forEach((tile) => {
      tile.addEventListener("click", () => {
        if (tile.dataset.tile === "complete") {
          el.statusFilter.value = el.statusFilter.value === "complete" ? "all" : "complete";
        } else {
          const key = tile.dataset.stage;
          stageFilter = stageFilter === key ? null : key;
        }
        renderChip();
        applyFilters();
        updateTileActiveStates();
      });
    });
    updateTileActiveStates();
  }

  function updateTileActiveStates() {
    el.tiles.querySelectorAll(".tile").forEach((tile) => {
      if (tile.dataset.tile === "complete") {
        tile.classList.toggle("active", el.statusFilter.value === "complete");
      } else {
        tile.classList.toggle("active", stageFilter === tile.dataset.stage);
      }
    });
  }

  function renderChip() {
    if (!stageFilter) {
      el.stageChip.innerHTML = "";
      return;
    }
    el.stageChip.innerHTML = `<span class="chip">Missing: ${stageLabels[stageFilter]} <button id="clear-stage-chip">✕</button></span>`;
    document.getElementById("clear-stage-chip").addEventListener("click", () => {
      stageFilter = null;
      renderChip();
      applyFilters();
      updateTileActiveStates();
    });
  }

  function renderSidebar(data, stageKeys) {
    const links = data.schools.map((school) => {
      const recs = school.participants.flatMap((p) => p.recordings);
      const agg = computeAggregate(recs, stageKeys);
      const id = `school-${slug(school.name)}`;
      return `
        <div class="sidebar-link" data-jump="${id}">
          <span>${school.name}</span>
          <span class="count">${agg.fullyComplete}/${agg.total}</span>
        </div>`;
    });
    el.sidebarLinks.innerHTML = links.join("");
    el.sidebarLinks.querySelectorAll(".sidebar-link").forEach((node) => {
      node.addEventListener("click", () => {
        const target = document.getElementById(node.dataset.jump);
        if (target) {
          target.open = true;
          target.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      });
    });
  }

  function renderSummaryTable(data, stageKeys) {
    const headCells = [
      "School",
      "Participants",
      "Recordings",
      ...stageKeys.map((k) => stageLabels[k]),
      "Fully complete",
    ];
    el.summaryThead.innerHTML = `<tr>${headCells.map((h) => `<th>${h}</th>`).join("")}</tr>`;

    const rows = data.schools.map((school) => {
      const recs = school.participants.flatMap((p) => p.recordings);
      const agg = computeAggregate(recs, stageKeys);
      const total = agg.total || 1;
      const stageCells = stageKeys
        .map((key) => {
          const count = agg.byStage[key];
          const pct = (count / total) * 100;
          return `<td><div class="mini-bar-cell">
            <div class="mini-bar-track"><div class="mini-bar-fill" style="width:${pct}%"></div></div>
            <span class="mini-bar-pct">${fmtPct(pct)}</span>
          </div></td>`;
        })
        .join("");
      const headlinePct = (agg.fullyComplete / total) * 100;
      const id = `school-${slug(school.name)}`;
      return `
        <tr data-jump="${id}">
          <td>${school.name}</td>
          <td class="num">${school.participants.length}</td>
          <td class="num">${agg.total}</td>
          ${stageCells}
          <td><div class="mini-bar-cell headline">
            <div class="mini-bar-track"><div class="mini-bar-fill" style="width:${headlinePct}%"></div></div>
            <span class="mini-bar-pct">${agg.fullyComplete}/${agg.total}</span>
          </div></td>
        </tr>`;
    });
    el.summaryTbody.innerHTML = rows.join("");
    el.summaryTbody.querySelectorAll("tr").forEach((row) => {
      row.addEventListener("click", () => {
        const target = document.getElementById(row.dataset.jump);
        if (target) {
          target.open = true;
          target.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      });
    });
  }

  function slug(name) {
    return name.toLowerCase().replace(/[^a-z0-9]+/g, "-");
  }

  function renderTree(data, stageKeys) {
    const totalStages = stageKeys.length;
    const groupHeaderCells = STAGE_GROUPS.map(
      (group) => `<th colspan="${group.keys.length}">${group.label}</th>`
    ).join("");
    const stageHeaderCells = STAGE_GROUPS.flatMap((group) =>
      group.keys.map((key, idx) => `<th class="${idx === 0 ? "group-sep" : ""}">${stageLabels[key]}</th>`)
    ).join("");

    const schoolsHtml = data.schools.map((school) => {
      const schoolRecs = school.participants.flatMap((p) => p.recordings);
      const schoolAgg = computeAggregate(schoolRecs, stageKeys);
      const participantsHtml = school.participants
        .map((participant) => {
          const pAgg = computeAggregate(participant.recordings, stageKeys);
          const rowsHtml = participant.recordings
            .map((rec) => {
              const cells = STAGE_GROUPS.flatMap((group) =>
                group.keys.map((key, idx) => {
                  const done = !!rec.stages[key];
                  return `<td class="${idx === 0 ? "group-sep" : ""}">
                    <span class="stage-dot ${done ? "done" : "pending"}" title="${stageLabels[key]}: ${
                    done ? "complete" : "not yet"
                  }">${done ? CHECK_SVG : "–"}</span>
                  </td>`;
                })
              ).join("");
              return `
                <tr data-school="${escapeAttr(school.name)}" data-participant="${escapeAttr(
                participant.name
              )}" data-recording="${escapeAttr(rec.name)}" data-completed="${rec.completed}" data-stages='${JSON.stringify(
                rec.stages
              )}'>
                  <td class="rec-name" title="${escapeAttr(rec.name)}">${rec.name}</td>
                  ${cells}
                  <td class="rec-progress">${rec.completed}/${totalStages}</td>
                </tr>`;
            })
            .join("");

          return `
            <details class="participant" data-participant-agg data-school="${escapeAttr(
              school.name
            )}" data-participant="${escapeAttr(participant.name)}">
              <summary>
                <span class="chevron">▶</span>
                <span>${participant.name}</span>
                <span class="p-counts">${pAgg.fullyComplete}/${pAgg.total} complete</span>
              </summary>
              <div class="participant-body">
                <table class="recordings">
                  <thead>
                    <tr class="group-row"><th></th>${groupHeaderCells}<th></th></tr>
                    <tr class="stage-row"><th class="rec-name">Recording</th>${stageHeaderCells}<th>Stages</th></tr>
                  </thead>
                  <tbody>${rowsHtml}</tbody>
                </table>
              </div>
            </details>`;
        })
        .join("");

      const id = `school-${slug(school.name)}`;
      return `
        <details class="school" id="${id}" open data-school="${escapeAttr(school.name)}">
          <summary>
            <span class="chevron">▶</span>
            <span>${school.name}</span>
            <span class="school-counts">${school.participants.length} participants · ${
        schoolAgg.total
      } recordings · ${schoolAgg.fullyComplete}/${schoolAgg.total} fully complete</span>
          </summary>
          <div class="school-body">${participantsHtml || '<p class="empty-state">No recordings yet.</p>'}</div>
        </details>`;
    });

    el.tree.innerHTML = schoolsHtml.join("");
  }

  function escapeAttr(value) {
    return String(value).replace(/"/g, "&quot;");
  }

  function applyFilters() {
    if (!latestData) return;
    const searchText = el.search.value.trim().toLowerCase();
    const statusValue = el.statusFilter.value;
    const totalStages = latestData.stages.length;

    let anyVisible = false;

    document.querySelectorAll("table.recordings tbody tr").forEach((row) => {
      const haystack = `${row.dataset.school} ${row.dataset.participant} ${row.dataset.recording}`.toLowerCase();
      let visible = true;

      if (searchText && !haystack.includes(searchText)) visible = false;

      const completed = Number(row.dataset.completed);
      if (visible && statusValue === "complete" && completed !== totalStages) visible = false;
      if (visible && statusValue === "incomplete" && completed === totalStages) visible = false;

      if (visible && stageFilter) {
        const stages = JSON.parse(row.dataset.stages);
        if (stages[stageFilter]) visible = false; // showing only what's still missing this stage
      }

      row.classList.toggle("hidden", !visible);
      if (visible) anyVisible = true;
    });

    const filtersActive = Boolean(searchText || statusValue !== "all" || stageFilter);

    document.querySelectorAll("details.participant").forEach((details) => {
      const visibleRows = details.querySelectorAll("table.recordings tbody tr:not(.hidden)").length;
      const show = visibleRows > 0;
      details.classList.toggle("hidden", !show);
      if (filtersActive) details.open = show;
    });

    document.querySelectorAll("details.school").forEach((details) => {
      const visibleParticipants = details.querySelectorAll("details.participant:not(.hidden)").length;
      const show = visibleParticipants > 0;
      details.classList.toggle("hidden", !show);
      if (filtersActive) details.open = show;
    });

    el.emptyState.classList.toggle("hidden", anyVisible);
  }

  async function fetchData() {
    const response = await fetch("/api/data");
    return response.json();
  }

  async function refresh() {
    el.refreshBtn.disabled = true;
    el.refreshBtn.textContent = "Refreshing…";
    try {
      const response = await fetch("/api/refresh", { method: "POST" });
      render(await response.json());
    } finally {
      el.refreshBtn.disabled = false;
      el.refreshBtn.textContent = "Refresh";
    }
  }

  el.refreshBtn.addEventListener("click", refresh);
  el.search.addEventListener("input", applyFilters);
  el.statusFilter.addEventListener("change", () => {
    applyFilters();
    updateTileActiveStates();
  });
  el.expandAll.addEventListener("click", () => {
    document.querySelectorAll("details").forEach((d) => (d.open = true));
  });
  el.collapseAll.addEventListener("click", () => {
    document.querySelectorAll("details.participant").forEach((d) => (d.open = false));
  });

  fetchData().then(render);
  setInterval(() => fetchData().then(render), 15000);
})();
