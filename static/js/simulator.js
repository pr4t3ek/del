/* Delivery Risk & Shipping Decision Simulator.
   Every number shown is computed by the trained model and the decision engine via
   /api/predict - nothing is hardcoded. */

let FIELD_DEFS = [], PROFILE = {}, LAST = null;

function collectInputs() {
  const inputs = {};
  document.querySelectorAll('.sim-input').forEach(el => {
    inputs[el.dataset.field] = el.type === 'range' ? parseFloat(el.value) : el.value;
  });
  return inputs;
}

function collectAssumptions() {
  const a = {};
  document.querySelectorAll('.sim-assume').forEach(el => { a[el.id] = parseFloat(el.value); });
  return a;
}

function syncOutputs() {
  const fmt = {
    late_penalty_fixed:   v => '$' + Number(v).toFixed(0),
    late_penalty_rate:    v => (100 * v).toFixed(0) + '%',
    speed_value_per_day:  v => '$' + Number(v).toFixed(1),
    holding_rate_per_day: v => (100 * v).toFixed(2) + '%'
  };
  document.querySelectorAll('.sim-assume').forEach(el => {
    el.nextElementSibling.textContent = fmt[el.id] ? fmt[el.id](el.value) : el.value;
  });
  document.querySelectorAll('input.sim-input[type=range]').forEach(el => {
    const f = el.dataset.field;
    const v = parseFloat(el.value);
    el.nextElementSibling.textContent =
      f.indexOf('Rate') >= 0 ? (100 * v).toFixed(0) + '%'
      : (f.indexOf('Price') >= 0 || f.indexOf('Total') >= 0) ? money(v, 0)
      : v;
  });
}

/* The promised window is a deterministic lookup on the mode, so keep it in sync unless the
   user explicitly unlocks it to model a promise redesign. */
function syncPromise() {
  const override = document.getElementById('promise-override').checked;
  const slider = document.getElementById('promised-days');
  const badge = document.getElementById('promise-auto');
  slider.disabled = !override;
  badge.textContent = override ? 'manual' : 'auto';
  badge.className = 'badge ' + (override ? 'warn' : 'mute');
  if (!override) {
    const modeEl = document.querySelector('.sim-input[data-field="Shipping Mode"]');
    const prof = modeEl ? PROFILE[modeEl.value] : null;
    if (prof) slider.value = prof.promised_days;
  }
  document.getElementById('promised-out').textContent = slider.value;
}

function render(data) {
  LAST = data;
  const p = data.p_late;

  const fill = document.getElementById('risk-fill');
  fill.style.width = Math.max(4, p * 100).toFixed(1) + '%';
  fill.textContent = pct(p, 0);
  fill.className = 'fill ' + riskClass(p);

  const band = document.getElementById('risk-band');
  band.textContent = data.risk_band + ' risk';
  band.className = 'badge ' + (p < 0.33 ? 'good' : (p < 0.66 ? 'warn' : 'bad'));

  const cmp = data.comparison;
  const rec = cmp.rows.find(r => r.mode === cmp.recommended);
  document.getElementById('rec-mode').textContent = cmp.recommended || '-';
  document.getElementById('rec-cost').textContent = rec ? money(rec.expected_total_cost) : '-';
  document.getElementById('pred-profit').textContent =
    data.predicted_profit !== null && data.predicted_profit !== undefined
      ? money(data.predicted_profit) : 'n/a';

  const current = cmp.rows.find(r => r.mode === data.selected_mode);
  const sub = document.getElementById('rec-saving');
  if (rec && current && rec.mode !== current.mode) {
    sub.innerHTML = '<span class="badge good">saves ' +
      money(current.expected_total_cost - rec.expected_total_cost) + ' vs ' + current.mode + '</span>';
  } else {
    sub.textContent = 'current mode is already optimal';
  }
  document.getElementById('rec-sub').textContent =
    cmp.constrained ? 'best feasible under your constraint' : 'under current assumptions';

  /* risk drivers */
  document.getElementById('risk-drivers').innerHTML = data.risk_drivers.map((d, i) => `
    <div style="margin-bottom:10px">
      <strong>${i + 1}. ${d.title}</strong>
      <div class="note" style="margin-top:2px">${d.detail}</div>
    </div>`).join('');

  /* recommended action */
  const action = document.getElementById('rec-action');
  if (rec && current && rec.mode !== current.mode) {
    const saving = current.expected_total_cost - rec.expected_total_cost;
    action.innerHTML = `<p><strong>Recommended action: switch from ${current.mode} to
      ${rec.mode}.</strong></p>
      <p>Under the selected cost assumptions this lowers expected total cost by
      ${money(saving)} per order (${money(current.expected_total_cost)} to
      ${money(rec.expected_total_cost)}), with model-implied late risk moving from
      ${pct(current.p_late)} to ${pct(rec.p_late)}.</p>
      <p class="note">A decision recommendation under stated assumptions &mdash; not causal
      proof that switching will produce this outcome.</p>`;
  } else if (rec) {
    action.innerHTML = `<p><strong>Recommended action: keep ${rec.mode}.</strong></p>
      <p>No alternative mode lowers expected total cost for this order at the current
      assumptions. Expected cost is ${money(rec.expected_total_cost)}.</p>
      <p class="note">A decision recommendation under stated assumptions.</p>`;
  }

  /* comparison table */
  document.querySelector('#mode-table tbody').innerHTML = cmp.rows.map(r => `
    <tr ${r.mode === cmp.recommended ? 'style="background:var(--good-sf)"' : ''}>
      <td><strong>${r.mode}</strong>${r.mode === data.selected_mode
          ? ' <span class="badge info">current</span>' : ''}</td>
      <td class="num">${pct(r.p_late)}</td>
      <td class="num">${money(r.freight)}</td>
      <td class="num">${money(r.expected_delay_cost)}</td>
      <td class="num">${money(r.holding_cost)}</td>
      <td class="num">${money(r.speed_cost)}</td>
      <td class="num"><strong>${money(r.expected_total_cost)}</strong></td>
      <td>${!r.feasible ? '<span class="badge bad">Excluded</span> ' + r.infeasible_reason
            : (r.mode === cmp.recommended ? '<span class="badge good">Recommended</span>'
               : '<span class="note">+' + money(r.expected_total_cost -
                   cmp.recommended_cost) + '</span>')}</td>
    </tr>`).join('');

  const modes = cmp.rows.map(r => r.mode);
  barH('c-sim-risk', modes, cmp.rows.map(r => 100 * r.p_late),
    { fmt: v => v.toFixed(1) + '%', xtitle: 'Model-implied P(late) (%)', left: 130,
      height: 250, suffix: '%',
      color: cmp.rows.map(r => r.p_late < 0.33 ? PALETTE.green
                             : (r.p_late < 0.66 ? PALETTE.amber : PALETTE.red)) });
  barH('c-sim-cost', modes, cmp.rows.map(r => r.expected_total_cost),
    { fmt: v => money(v, 0), xtitle: 'Expected total cost ($)', left: 130, height: 250,
      color: cmp.rows.map(r => r.mode === cmp.recommended ? PALETTE.green : PALETTE.accent) });
}

function predict() {
  syncOutputs();
  syncPromise();
  fetch('/api/predict', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ inputs: collectInputs(), assumptions: collectAssumptions() })
  })
  .then(r => r.json())
  .then(data => { if (!data.error) render(data); else console.error(data.error); })
  .catch(err => console.error('predict failed', err));
}

/* ---------- what-if ---------- */
function populateWhatIfValues() {
  const field = document.getElementById('whatif-field').value;
  const def = FIELD_DEFS.find(f => f.name === field);
  const sel = document.getElementById('whatif-value');
  sel.innerHTML = (def ? def.options : []).map(o => `<option value="${o}">${o}</option>`).join('');
}

function runWhatIf() {
  if (!LAST) return;
  const field = document.getElementById('whatif-field').value;
  const value = document.getElementById('whatif-value').value;
  const base = collectInputs();
  const modified = Object.assign({}, base);
  modified[field] = value;

  fetch('/api/predict', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ inputs: modified, assumptions: collectAssumptions() })
  })
  .then(r => r.json())
  .then(data => {
    if (data.error) return;
    const before = LAST.p_late, after = data.p_late;
    const deltaP = (after - before) * 100;
    const cBefore = LAST.comparison.recommended_cost;
    const cAfter = data.comparison.recommended_cost;
    const dir = deltaP > 0.05 ? 'bad' : (deltaP < -0.05 ? 'good' : 'mute');

    document.getElementById('whatif-result').innerHTML = `
      <div class="callout ${dir}" style="margin-top:12px">
        <p><strong>Model-implied scenario change</strong> &mdash;
           ${field}: <span class="mono">${base[field]}</span> &rarr;
           <span class="mono">${value}</span></p>
      </div>
      <div class="table-wrap" style="margin-top:12px"><table class="compact">
        <thead><tr><th>Measure</th><th class="num">Current</th><th class="num">Scenario</th><th class="num">Change</th></tr></thead>
        <tbody>
          <tr><td>Predicted late risk</td>
              <td class="num">${pct(before)}</td><td class="num">${pct(after)}</td>
              <td class="num">${deltaP >= 0 ? '+' : ''}${deltaP.toFixed(1)} pp</td></tr>
          <tr><td>Expected total cost (best mode)</td>
              <td class="num">${money(cBefore)}</td><td class="num">${money(cAfter)}</td>
              <td class="num">${cAfter - cBefore >= 0 ? '+' : ''}${money(cAfter - cBefore)}</td></tr>
          <tr><td>Recommended mode</td>
              <td class="num">${LAST.comparison.recommended}</td>
              <td class="num">${data.comparison.recommended}</td>
              <td class="num">${LAST.comparison.recommended === data.comparison.recommended
                    ? 'unchanged' : 'changed'}</td></tr>
        </tbody>
      </table></div>
      <p class="note" style="margin-top:8px">Perturbing one variable in an observational model
      shows what the model expects for an order with those characteristics. It does not imply
      that changing this variable would cause the difference.</p>`;
  })
  .catch(err => console.error('what-if failed', err));
}

/* ---------- init ---------- */
function initSimulator(fields, profile) {
  FIELD_DEFS = fields; PROFILE = profile;

  document.querySelectorAll('.sim-input').forEach(el => {
    el.addEventListener('input', syncOutputs);
    el.addEventListener('change', predict);
  });
  document.querySelectorAll('.sim-assume').forEach(el => {
    el.addEventListener('input', syncOutputs);
    el.addEventListener('change', predict);
  });
  document.getElementById('promise-override').addEventListener('change', () => { syncPromise(); predict(); });
  document.getElementById('promised-days').addEventListener('input', syncPromise);
  document.querySelectorAll('button.preset').forEach(btn => {
    btn.addEventListener('click', () => {
      const presets = {
        low:    { late_penalty_fixed: 5,  late_penalty_rate: 0.01, speed_value_per_day: 3 },
        medium: { late_penalty_fixed: 15, late_penalty_rate: 0.05, speed_value_per_day: 8 },
        high:   { late_penalty_fixed: 40, late_penalty_rate: 0.15, speed_value_per_day: 15 }
      };
      const p = presets[btn.dataset.preset];
      if (!p) return;
      Object.entries(p).forEach(([k, v]) => {
        const el = document.getElementById(k); if (el) el.value = v;
      });
      predict();
    });
  });
  document.getElementById('whatif-field').addEventListener('change', populateWhatIfValues);
  document.getElementById('whatif-run').addEventListener('click', runWhatIf);

  populateWhatIfValues();
  predict();
}
