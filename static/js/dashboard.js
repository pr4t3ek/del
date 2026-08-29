/* Shared chart helpers.
   All series arrive pre-aggregated from the server - the browser never receives raw rows. */

const PALETTE = {
  navy:   '#16324f',
  accent: '#0f6fc5',
  teal:   '#2a8f8f',
  amber:  '#c98a12',
  red:    '#b3261e',
  green:  '#1a7f4b',
  grey:   '#8b97a8',
  light:  '#a9c7e4'
};
const SERIES = [PALETTE.accent, PALETTE.amber, PALETTE.teal, PALETTE.navy,
                PALETTE.green, PALETTE.red, PALETTE.grey, PALETTE.light];

const LAYOUT = {
  font: { family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif', size: 12, color: '#35414f' },
  paper_bgcolor: '#fff',
  plot_bgcolor: '#fff',
  margin: { l: 62, r: 20, t: 28, b: 52 },
  xaxis: { gridcolor: '#eef1f6', zerolinecolor: '#e2e7ee', automargin: true },
  yaxis: { gridcolor: '#eef1f6', zerolinecolor: '#e2e7ee', automargin: true },
  legend: { orientation: 'h', y: -0.2, font: { size: 11 } },
  hoverlabel: { bgcolor: '#10151c', font: { color: '#fff', size: 12 } }
};
const OPTS = { displayModeBar: false, responsive: true };

function layout(extra) {
  const base = JSON.parse(JSON.stringify(LAYOUT));
  return Object.assign(base, extra || {});
}

function draw(id, traces, extra) {
  const el = document.getElementById(id);
  if (!el) return;
  Plotly.newPlot(el, traces, layout(extra), OPTS);
}

/* Horizontal bar - the default for ranked categorical comparisons, since labels stay
   readable however long they are. */
function barH(id, labels, values, opts) {
  opts = opts || {};
  draw(id, [{
    type: 'bar', orientation: 'h',
    x: values, y: labels,
    marker: { color: opts.color || PALETTE.accent },
    text: values.map(v => opts.fmt ? opts.fmt(v) : v),
    textposition: 'auto',
    hovertemplate: '%{y}: %{x}' + (opts.suffix || '') + '<extra></extra>'
  }], {
    height: opts.height || Math.max(220, labels.length * 26 + 80),
    margin: { l: opts.left || 150, r: 24, t: 14, b: 42 },
    xaxis: { title: opts.xtitle || '', gridcolor: '#eef1f6' },
    yaxis: { automargin: true, autorange: 'reversed' }
  });
}

function barV(id, labels, values, opts) {
  opts = opts || {};
  draw(id, [{
    type: 'bar', x: labels, y: values,
    marker: { color: opts.colors || opts.color || PALETTE.accent },
    text: values.map(v => opts.fmt ? opts.fmt(v) : v),
    textposition: 'auto',
    hovertemplate: '%{x}: %{y}' + (opts.suffix || '') + '<extra></extra>'
  }], {
    height: opts.height || 300,
    yaxis: { title: opts.ytitle || '' },
    xaxis: { title: opts.xtitle || '' }
  });
}

function groupedBar(id, labels, series, opts) {
  opts = opts || {};
  const traces = series.map((s, i) => ({
    type: 'bar', name: s.name, x: labels, y: s.values,
    marker: { color: s.color || SERIES[i % SERIES.length] },
    hovertemplate: '%{x}<br>' + s.name + ': %{y}' + (s.suffix || '') + '<extra></extra>'
  }));
  draw(id, traces, {
    barmode: opts.mode || 'group',
    height: opts.height || 330,
    yaxis: { title: opts.ytitle || '' }
  });
}

function lineChart(id, x, series, opts) {
  opts = opts || {};
  const traces = series.map((s, i) => ({
    type: 'scatter', mode: s.mode || 'lines', name: s.name, x: x, y: s.values,
    line: { color: s.color || SERIES[i % SERIES.length], width: 2, shape: s.shape || 'linear' },
    marker: { size: 5 },
    yaxis: s.axis || 'y',
    hovertemplate: '%{x}<br>' + s.name + ': %{y}' + (s.suffix || '') + '<extra></extra>'
  }));
  draw(id, traces, Object.assign({
    height: opts.height || 320,
    xaxis: { title: opts.xtitle || '' },
    yaxis: { title: opts.ytitle || '' }
  }, opts.extra || {}));
}

function histogram(id, edges, counts, opts) {
  opts = opts || {};
  const centres = [];
  for (let i = 0; i < counts.length; i++) centres.push((edges[i] + edges[i + 1]) / 2);
  const width = edges.length > 1 ? (edges[1] - edges[0]) : 1;
  draw(id, [{
    type: 'bar', x: centres, y: counts, width: width * 0.96,
    marker: { color: opts.color || PALETTE.accent, line: { width: 0 } },
    hovertemplate: '%{x}: %{y} orders<extra></extra>'
  }], {
    height: opts.height || 300, bargap: 0.02,
    xaxis: { title: opts.xtitle || '' },
    yaxis: { title: 'Orders' }
  });
}

function heatmap(id, labels, matrix, opts) {
  opts = opts || {};
  draw(id, [{
    type: 'heatmap', z: matrix, x: labels, y: labels,
    colorscale: [[0, '#b3261e'], [0.5, '#ffffff'], [1, '#0f6fc5']],
    zmid: 0, zmin: -1, zmax: 1,
    hovertemplate: '%{y} / %{x}: %{z:.3f}<extra></extra>',
    colorbar: { thickness: 12, len: 0.85, tickfont: { size: 10 } }
  }], {
    height: opts.height || 560,
    margin: { l: 190, r: 20, t: 16, b: 180 },
    xaxis: { tickangle: -45, tickfont: { size: 10 }, gridcolor: 'transparent' },
    yaxis: { autorange: 'reversed', tickfont: { size: 10 }, gridcolor: 'transparent' }
  });
}

function scatter(id, groups, opts) {
  opts = opts || {};
  const traces = groups.map((g, i) => ({
    type: 'scattergl', mode: 'markers', name: g.name, x: g.x, y: g.y,
    marker: { color: g.color || SERIES[i % SERIES.length], size: 5, opacity: 0.55 },
    hovertemplate: g.name + '<br>' + (opts.xtitle || 'x') + ': %{x}<br>' +
                   (opts.ytitle || 'y') + ': %{y:.3f}<extra></extra>'
  }));
  draw(id, traces, Object.assign({
    height: opts.height || 460,
    xaxis: { title: opts.xtitle || '' },
    yaxis: { title: opts.ytitle || '' }
  }, opts.extra || {}));
}

function pct(v, digits) { return (v * 100).toFixed(digits === undefined ? 1 : digits) + '%'; }
function money(v, digits) {
  return '$' + Number(v).toLocaleString('en-US', {
    minimumFractionDigits: digits === undefined ? 2 : digits,
    maximumFractionDigits: digits === undefined ? 2 : digits
  });
}
function riskClass(p) { return p < 0.33 ? 'low' : (p < 0.66 ? 'med' : 'high'); }
