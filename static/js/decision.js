/* Decision Optimization page.
   Assumption sliders post to /api/scenario, which recomputes the portfolio optimisation on
   the stored counterfactual scores. No model re-scoring happens, so the page stays responsive. */

let DECISION = null;

function currentAssumptions() {
  const freight = {};
  document.querySelectorAll('input.freight').forEach(el => {
    freight[el.dataset.mode] = parseFloat(el.value);
  });
  return {
    freight_cost: freight,
    late_penalty_fixed:   parseFloat(document.getElementById('late_penalty_fixed').value),
    late_penalty_rate:    parseFloat(document.getElementById('late_penalty_rate').value),
    holding_rate_per_day: parseFloat(document.getElementById('holding_rate_per_day').value),
    speed_value_per_day:  parseFloat(document.getElementById('speed_value_per_day').value)
  };
}

function currentConstraints() {
  const v = parseFloat(document.getElementById('max_late').value);
  return v >= 1 ? {} : { max_late_probability: v };
}

function syncOutputs() {
  document.querySelectorAll('input.freight').forEach(el => {
    el.nextElementSibling.textContent = '$' + Number(el.value).toFixed(0);
  });
  const fmt = {
    late_penalty_fixed:   v => '$' + Number(v).toFixed(0),
    late_penalty_rate:    v => (100 * v).toFixed(0) + '%',
    holding_rate_per_day: v => (100 * v).toFixed(2) + '%',
    speed_value_per_day:  v => '$' + Number(v).toFixed(1)
  };
  Object.keys(fmt).forEach(id => {
    const el = document.getElementById(id);
    if (el) el.nextElementSibling.textContent = fmt[id](el.value);
  });
  const ml = document.getElementById('max_late');
  ml.nextElementSibling.textContent = parseFloat(ml.value) >= 1 ? 'off' : pct(parseFloat(ml.value), 0);
}

/* ---------- rendering ---------- */
function renderEngine(data) {
  const rows = data.policies.single_mode_detail.map(p => {
    const mode = Object.keys(p.mode_mix)[0];
    return { mode: mode, avg_cost: p.avg_cost, late: p.expected_late_rate };
  });
  const profile = DECISION.mode_profile;
  const a = data.assumptions;
  const value = data.median_order_value;
  const penalty = a.late_penalty_fixed + a.late_penalty_rate * value;

  const detail = rows.map(r => {
    const prof = profile[r.mode] || {};
    const transit = prof.mean_transit || 0;
    const freight = a.freight_cost[r.mode] || 0;
    const delay = r.late * penalty;
    const holding = a.holding_rate_per_day * value * transit;
    const speed = a.speed_value_per_day * transit;
    return Object.assign(r, {
      freight: freight, delay: delay, holding: holding, speed: speed,
      total: freight + delay + holding + speed, transit: transit
    });
  }).sort((x, y) => x.total - y.total);

  const best = detail[0];
  document.querySelector('#engine-table tbody').innerHTML = detail.map(r => `
    <tr ${r.mode === best.mode ? 'style="background:var(--good-sf)"' : ''}>
      <td><strong>${r.mode}</strong></td>
      <td class="num">${pct(r.late)}</td>
      <td class="num">${money(r.freight)}</td>
      <td class="num">${money(r.delay)}</td>
      <td class="num">${money(r.holding)}</td>
      <td class="num">${money(r.speed)}</td>
      <td class="num"><strong>${money(r.total)}</strong></td>
      <td class="num">${r.transit.toFixed(2)} d</td>
      <td>${r.mode === best.mode
            ? '<span class="badge good">Lowest expected cost</span>'
            : '<span class="note">+' + money(r.total - best.total) + ' vs best</span>'}</td>
    </tr>`).join('');
  document.getElementById('median-value').textContent = money(value);
}

function renderPolicies(data) {
  const P = data.policies.policies;
  const baseline = P.reduce((a, b) => a.avg_cost > b.avg_cost ? a : b);

  document.querySelector('#policy-table tbody').innerHTML = P.map((p, i) => `
    <tr ${i === 0 ? 'style="background:var(--good-sf)"' : ''}>
      <td><strong>${p.policy}</strong></td>
      <td class="num">${money(p.avg_cost)}</td>
      <td class="num">${pct(p.expected_late_rate)}</td>
      <td class="num">${p.saving_pct > 0
          ? '<span class="badge good">' + p.saving_pct.toFixed(1) + '%</span>'
          : '<span class="badge mute">baseline</span>'}</td>
      <td style="font-size:12.5px">${p.complexity}</td>
    </tr>`).join('');

  document.querySelector('#policy-desc tbody').innerHTML = P.map(p => `
    <tr><td><strong>${p.policy}</strong></td>
        <td class="wrap" style="font-size:12.5px">${p.description}</td>
        <td style="font-size:12px">${Object.entries(p.mode_mix)
            .map(([m, s]) => m + ' ' + pct(s, 0)).join(', ')}</td></tr>`).join('');

  draw('c-policies', [{
    type: 'scatter', mode: 'markers+text',
    x: P.map(p => p.avg_cost), y: P.map(p => 100 * p.expected_late_rate),
    text: P.map(p => p.policy.split(' - ')[0]), textposition: 'top center',
    marker: { size: 14, color: P.map((p, i) => i === 0 ? PALETTE.green : PALETTE.accent), opacity: .8 },
    customdata: P.map(p => p.policy),
    hovertemplate: '%{customdata}<br>Cost $%{x:.2f}/order<br>Late %{y:.1f}%<extra></extra>'
  }], { height: 340, xaxis: { title: 'Expected cost per order ($)' },
        yaxis: { title: 'Expected late rate (%)' }, margin: { t: 28 } });
}

function renderBreakEven(data) {
  document.querySelector('#breakeven-table tbody').innerHTML = data.break_even.map(b => `
    <tr>
      <td><strong>${b.from_mode} &rarr; ${b.to_mode}</strong></td>
      <td class="num">${money(b.incremental_freight)}</td>
      <td class="num">${pct(b.observed_p_from)}</td>
      <td class="num">${pct(b.observed_p_to)}</td>
      <td class="num">${b.break_even_p === null ? '&mdash;' : pct(b.break_even_p)}</td>
      <td class="num">${money(b.cost_from)}</td>
      <td class="num">${money(b.cost_to)}</td>
      <td class="wrap" style="font-size:12.5px">
        ${b.worthwhile ? '<span class="badge good">Worthwhile</span> '
                       : '<span class="badge mute">Not worthwhile</span> '}${b.verdict}</td>
    </tr>`).join('');
}

function refresh(preset) {
  syncOutputs();
  const body = preset
    ? { preset: preset, constraints: currentConstraints() }
    : { assumptions: currentAssumptions(), constraints: currentConstraints() };

  fetch('/api/scenario', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  })
  .then(r => r.json())
  .then(data => {
    if (data.error) { console.error(data.error); return; }
    if (preset) applyAssumptionsToControls(data.assumptions);
    renderEngine(data);
    renderPolicies(data);
    renderBreakEven(data);
  })
  .catch(err => console.error('scenario request failed', err));
}

function applyAssumptionsToControls(a) {
  document.querySelectorAll('input.freight').forEach(el => {
    if (a.freight_cost[el.dataset.mode] !== undefined) el.value = a.freight_cost[el.dataset.mode];
  });
  ['late_penalty_fixed', 'late_penalty_rate', 'holding_rate_per_day', 'speed_value_per_day']
    .forEach(id => { const el = document.getElementById(id); if (el && a[id] !== undefined) el.value = a[id]; });
  syncOutputs();
}

/* ---------- static charts ---------- */
function renderDegeneracy(d) {
  const mix = k => {
    const m = d.degeneracy[k];
    return { labels: Object.keys(m), values: Object.values(m).map(v => 100 * v) };
  };
  const deg = mix('degenerate_mix'), full = mix('full_mix');
  barH('c-degenerate', deg.labels, deg.values,
    { fmt: v => v.toFixed(1) + '%', xtitle: 'Share of orders (%)', left: 130,
      height: 220, color: PALETTE.grey, suffix: '%' });
  barH('c-full-mix', full.labels, full.values,
    { fmt: v => v.toFixed(1) + '%', xtitle: 'Share of orders (%)', left: 130,
      height: 220, color: PALETTE.accent, suffix: '%' });
}

function renderSensitivity(d) {
  Object.values(d.sensitivity).forEach((s, i) => {
    const modes = [...new Set(s.rows.map(r => r.dominant_mode))];
    lineChart('c-sens-' + (i + 1), s.rows.map(r => r.value), [
      { name: 'Avg cost/order', values: s.rows.map(r => r.avg_cost), color: PALETTE.navy },
      { name: 'Expected late rate (%)',
        values: s.rows.map(r => 100 * r.expected_late_rate), color: PALETTE.red, axis: 'y2' }
    ], { height: 250, xtitle: s.parameter, ytitle: 'Cost ($)',
         extra: { yaxis2: { title: 'Late (%)', overlaying: 'y', side: 'right',
                            gridcolor: 'transparent' } } });
  });
}

function renderMatrix() {
  const x = document.getElementById('matrix-x').value;
  fetch('/api/risk-matrix?x=' + encodeURIComponent(x))
    .then(r => r.json())
    .then(data => {
      if (data.error) return;
      const colours = { 'Prioritise': PALETTE.red, 'Watch / manage': PALETTE.amber,
                        'Protect': PALETTE.accent, 'Routine': PALETTE.grey };
      const groups = Object.keys(colours).map(q => ({
        name: q, color: colours[q],
        x: data.points.filter(p => p.quadrant === q).map(p => p.x),
        y: data.points.filter(p => p.quadrant === q).map(p => p.y)
      })).filter(g => g.x.length);

      scatter('c-matrix', groups, {
        xtitle: data.x_label, ytitle: 'Predicted late probability', height: 480,
        extra: {
          shapes: [
            { type: 'line', x0: data.value_median, x1: data.value_median,
              y0: 0, y1: 1, line: { color: PALETTE.navy, width: 1.5, dash: 'dot' } },
            { type: 'line', x0: Math.min(...data.points.map(p => p.x)),
              x1: Math.max(...data.points.map(p => p.x)),
              y0: data.risk_median, y1: data.risk_median,
              line: { color: PALETTE.navy, width: 1.5, dash: 'dot' } }
          ]
        }
      });

      document.getElementById('matrix-summary').innerHTML = data.summary.map(s => `
        <div class="stat-row">
          <span class="k"><span class="badge" style="background:${colours[s.quadrant]}22;color:${colours[s.quadrant]}">${s.quadrant}</span></span>
          <span class="v">${s.orders.toLocaleString()} &middot; ${pct(s.avg_risk)}</span>
        </div>`).join('');
    })
    .catch(err => console.error('risk matrix failed', err));
}

/* ---------- init ---------- */
function initDecision(d) {
  DECISION = d;
  renderDegeneracy(d);
  renderSensitivity(d);
  renderMatrix();

  document.querySelectorAll('input.assume, #max_late').forEach(el => {
    el.addEventListener('input', syncOutputs);
    el.addEventListener('change', () => refresh(null));
  });
  document.querySelectorAll('button.preset').forEach(btn => {
    btn.addEventListener('click', () => refresh(btn.dataset.preset));
  });
  document.getElementById('reset-assume').addEventListener('click', () => {
    applyAssumptionsToControls(d.assumptions);
    document.getElementById('max_late').value = 1;
    refresh(null);
  });
  document.getElementById('matrix-x').addEventListener('change', renderMatrix);

  refresh(null);
}
