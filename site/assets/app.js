const $ = id => document.getElementById(id);
const state = { task: 'current_hour', method: 'RFECV', k: null, report: null };
const names = { RFECV: 'RFECV', SelectKBest: 'SelectKBest', LassoCV: 'Lasso CV', 'All Features': '全特徵基準' };
const explanations = {
  RFECV: '曲線是 RFE 各 k 的探索結果；紫色菱形是 RFECV 在內層交叉驗證選出的正式點，兩者分數不可混用。',
  LassoCV: '曲線是 Lasso 路徑實際出現的 k；紫色菱形是每個內層訓練折各自標準化的 Lasso CV 正式點。',
  SelectKBest: '依單一特徵與目標的統計關係選 k，再用相同時間折比較迴歸誤差。',
  'All Features': '14 個編碼後候選欄位全部入模，只有 k=14 一個真實點；下方跨方法圖以虛線顯示其固定參考分數。',
};
const comparisonMethods = [
  { key: 'SelectKBest', label: 'SelectKBest', color: '#66d5f2' },
  { key: 'RFECV', label: 'RFE 路徑', color: '#7fe3b0' },
  { key: 'LassoCV', label: 'Lasso 路徑', color: '#ffc78e' },
  { key: 'All Features', label: '全特徵基準 · k=14', color: '#d4e2ed' },
];
const fmt = (value, digits = 2) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—';
const rowsFor = () => state.report.tasks[state.task].methods[state.method].rows;
const exactFor = () => state.report.tasks[state.task].methods[state.method].exact;
const candidateFor = method => state.report.tasks[state.task].candidates.find(item => item.method === method);

function svgElement(tag, attributes, parent) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  parent.appendChild(node);
  return node;
}

function draw(svgId, rows, key) {
  const svg = $(svgId);
  svg.replaceChildren();
  svg.setAttribute('viewBox', '0 0 560 210');
  const exact = exactFor();
  const formalPoint = exact?.available ? { k: exact.k, value: exact[key] } : null;
  const xs = rows.map(row => row.k).concat(formalPoint ? [formalPoint.k] : []);
  const ys = rows.map(row => row[key]).concat(formalPoint ? [formalPoint.value] : []);
  const left = 54, right = 540, top = 22, bottom = 166;
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const pad = (maxY - minY || Math.abs(maxY) * .08 || 1) * .22;
  const low = minY - pad, high = maxY + pad;
  const xp = k => minX === maxX ? (left + right) / 2 : left + (k - minX) / (maxX - minX) * (right - left);
  const yp = value => bottom - (value - low) / (high - low) * (bottom - top);
  for (let i = 0; i < 4; i++) {
    const y = top + (bottom - top) * i / 3;
    svgElement('line', { x1: left, y1: y, x2: right, y2: y, class: 'grid' }, svg);
    const label = svgElement('text', { x: left - 8, y: y + 4, 'text-anchor': 'end', class: 'tick' }, svg);
    label.textContent = key === 'cv_rmse' ? Math.round(high - (high - low) * i / 3) : fmt(high - (high - low) * i / 3, 2);
  }
  svgElement('line', { x1: left, y1: bottom, x2: right, y2: bottom, class: 'axis' }, svg);
  const ticks = rows.length <= 7 ? rows : rows.filter((_, index) => index % 2 === 0 || index === rows.length - 1);
  ticks.forEach(row => {
    const tick = svgElement('text', { x: xp(row.k), y: bottom + 20, 'text-anchor': 'middle', class: 'tick' }, svg);
    tick.textContent = row.k;
  });
  svgElement('path', { d: rows.map((row, index) => `${index ? 'L' : 'M'}${xp(row.k)},${yp(row[key])}`).join(' '), class: 'series' }, svg);
  const best = state.report.tasks[state.task].methods[state.method].best_path_k;
  rows.forEach(row => {
    if (row.k === best) svgElement('circle', { cx: xp(row.k), cy: yp(row[key]), r: 10, class: 'best-ring' }, svg);
    const point = svgElement('circle', { cx: xp(row.k), cy: yp(row[key]), r: row.k === state.k ? 6 : 4.5,
      class: `dot${row.k === state.k ? ' active' : ''}`, tabindex: '0', role: 'button',
      'aria-label': `探索路徑 k ${row.k}，${key === 'cv_rmse' ? 'RMSE' : 'R 平方'} ${fmt(row[key], key === 'cv_rmse' ? 2 : 3)}` }, svg);
    point.addEventListener('click', () => choose(row.k));
    point.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); choose(row.k); }
    });
  });
  if (formalPoint) {
    const x = xp(formalPoint.k), y = yp(formalPoint.value);
    const diamond = svgElement('path', { d: `M${x},${y - 8} L${x + 8},${y} L${x},${y + 8} L${x - 8},${y} Z`,
      fill: '#d9b7ff', stroke: '#fff', 'stroke-width': 1.5 }, svg);
    svgElement('title', {}, diamond).textContent = `${names[state.method]} 正式選擇 k=${formalPoint.k}；${key}=${fmt(formalPoint.value, key === 'cv_rmse' ? 2 : 3)}`;
  }
}

function choose(k) {
  state.k = k;
  render();
  document.querySelector(`tr[data-k="${k}"]`)?.scrollIntoView({ block: 'nearest' });
}

function selectComparisonPoint(method, k) {
  state.method = method;
  state.k = k;
  $('method').value = method;
  render();
  $('officialStudy').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function drawComparison(svgId, key) {
  const svg = $(svgId);
  svg.replaceChildren();
  svg.setAttribute('viewBox', '0 0 560 232');
  const task = state.report.tasks[state.task];
  const series = comparisonMethods.slice(0, 3).map(item => ({ ...item, rows: task.methods[item.key].rows }));
  const baseline = task.methods['All Features'].rows[0];
  const values = series.flatMap(item => item.rows.map(row => row[key]));
  values.push(baseline[key]);
  for (const item of series.slice(1)) {
    const exact = task.methods[item.key].exact;
    if (exact?.available) values.push(exact[key]);
  }
  const left = 53, right = 530, top = 20, bottom = 179;
  const min = Math.min(...values), max = Math.max(...values);
  const pad = (max - min || 1) * .12;
  const low = min - pad, high = max + pad;
  const xp = k => left + (k - 1) / 13 * (right - left);
  const yp = value => bottom - (value - low) / (high - low) * (bottom - top);
  const markers = series.flatMap(item => item.rows.map(row => ({ method: item.key, k: row.k, value: row[key] })));
  markers.push({ method: 'All Features', k: 14, value: baseline[key] });
  const markerX = (method, k, value) => {
    const nearby = markers.filter(point => point.k === k && Math.abs(yp(point.value) - yp(value)) < 8);
    const index = nearby.findIndex(point => point.method === method && point.value === value);
    return xp(k) + (index - (nearby.length - 1) / 2) * 10;
  };
  for (let i = 0; i < 4; i++) {
    const y = top + (bottom - top) * i / 3;
    svgElement('line', { x1: left, y1: y, x2: right, y2: y, class: 'grid' }, svg);
    const tick = svgElement('text', { x: left - 8, y: y + 4, 'text-anchor': 'end', class: 'tick' }, svg);
    tick.textContent = key === 'cv_rmse' ? Math.round(high - (high - low) * i / 3) : fmt(high - (high - low) * i / 3, 2);
  }
  svgElement('line', { x1: left, y1: bottom, x2: right, y2: bottom, class: 'axis' }, svg);
  [1, 3, 5, 7, 9, 11, 13, 14].forEach(k => {
    svgElement('text', { x: xp(k), y: bottom + 18, 'text-anchor': 'middle', class: 'tick' }, svg).textContent = k;
  });
  svgElement('text', { x: (left + right) / 2, y: 222, 'text-anchor': 'middle', class: 'tick' }, svg).textContent = '編碼後入模特徵數 k';
  const baseLine = svgElement('line', { x1: left, y1: yp(baseline[key]), x2: right, y2: yp(baseline[key]), class: 'comparison-baseline' }, svg);
  svgElement('title', {}, baseLine).textContent = `全特徵基準僅在 k=14 計算：${key === 'cv_rmse' ? 'RMSE' : 'R²'} ${fmt(baseline[key], key === 'cv_rmse' ? 2 : 3)}；虛線只作水平參考`;
  for (const item of series) {
    const path = svgElement('path', { d: item.rows.map((row, index) => `${index ? 'L' : 'M'}${xp(row.k)},${yp(row[key])}`).join(' '),
      class: 'comparison-line', stroke: item.color, 'stroke-dasharray': item.key === 'LassoCV' ? '5 4' : 'none' }, svg);
    svgElement('title', {}, path).textContent = `${item.label}：只連接已實際評估的特徵數`;
    for (const row of item.rows) {
      const point = svgElement('circle', { cx: markerX(item.key, row.k, row[key]), cy: yp(row[key]), r: item.key === state.method && row.k === state.k ? 6 : 4,
        fill: item.color, class: 'comparison-point', tabindex: '0', role: 'button',
        'aria-label': `${item.label}，k=${row.k}，${key === 'cv_rmse' ? 'RMSE' : 'R 平方'} ${fmt(row[key], key === 'cv_rmse' ? 2 : 3)}；點選查看特徵組合` }, svg);
      svgElement('title', {}, point).textContent = `${item.label} · k=${row.k} · ${key === 'cv_rmse' ? 'RMSE' : 'R²'} ${fmt(row[key], key === 'cv_rmse' ? 2 : 3)}\n${row.features.join('、')}`;
      point.addEventListener('click', () => selectComparisonPoint(item.key, row.k));
      point.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectComparisonPoint(item.key, row.k); }
      });
    }
    const exact = task.methods[item.key].exact;
    if (exact?.available) {
      const x = xp(exact.k), y = yp(exact[key]);
      const diamond = svgElement('path', { d: `M${x},${y - 7} L${x + 7},${y} L${x},${y + 7} L${x - 7},${y} Z`,
        fill: item.color, stroke: '#fff', 'stroke-width': 1.5 }, svg);
      svgElement('title', {}, diamond).textContent = `${names[item.key]} 正式選擇 · k=${exact.k} · ${key === 'cv_rmse' ? 'RMSE' : 'R²'} ${fmt(exact[key], key === 'cv_rmse' ? 2 : 3)}`;
    }
  }
  const basePoint = svgElement('circle', { cx: markerX('All Features', 14, baseline[key]), cy: yp(baseline[key]), r: state.method === 'All Features' ? 6 : 5,
    fill: '#d4e2ed', class: 'comparison-point', tabindex: '0', role: 'button',
    'aria-label': `全特徵基準，唯一真實點 k=14，${key === 'cv_rmse' ? 'RMSE' : 'R 平方'} ${fmt(baseline[key], key === 'cv_rmse' ? 2 : 3)}` }, svg);
  svgElement('title', {}, basePoint).textContent = '全特徵基準只計算 k=14；虛線不是其他 k 的結果';
  basePoint.addEventListener('click', () => selectComparisonPoint('All Features', 14));
  basePoint.addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectComparisonPoint('All Features', 14); }
  });
}

function renderComparison() {
  const legend = $('comparisonLegend');
  legend.replaceChildren();
  for (const item of comparisonMethods) {
    const button = document.createElement('button');
    button.type = 'button';
    button.setAttribute('aria-current', String(state.method === item.key));
    const swatch = document.createElement('span');
    swatch.className = `comparison-swatch${item.key === 'All Features' ? ' baseline' : ''}`;
    swatch.style.setProperty('--swatch', item.color);
    button.append(swatch, document.createTextNode(item.label));
    button.addEventListener('click', () => selectComparisonPoint(item.key, item.key === 'All Features' ? 14 : state.report.tasks[state.task].methods[item.key].best_path_k));
    legend.appendChild(button);
  }
  drawComparison('compareRmseChart', 'cv_rmse');
  drawComparison('compareR2Chart', 'cv_r2');
  const tbody = $('comparisonRows');
  tbody.replaceChildren();
  for (const item of comparisonMethods) {
    const formal = candidateFor(item.key);
    if (!formal) continue;
    const row = document.createElement('tr');
    if (state.method === item.key) row.className = 'active';
    for (const value of [names[item.key], String(formal.k), fmt(formal.cv_rmse), fmt(formal.cv_r2, 3), formal.features.join('、')]) {
      const cell = document.createElement('td');
      cell.textContent = value;
      row.appendChild(cell);
    }
    row.addEventListener('click', () => selectComparisonPoint(item.key, formal.k));
    tbody.appendChild(row);
  }
}

function render() {
  if (!state.report) return;
  const task = state.report.tasks[state.task];
  const method = task.methods[state.method];
  const rows = method.rows;
  if (!rows.length) return;
  if (!rows.some(row => row.k === state.k)) state.k = method.best_path_k ?? rows[0].k;
  const selected = rows.find(row => row.k === state.k);
  const count = $('featureCount');
  count.replaceChildren();
  rows.forEach(row => {
    const option = document.createElement('option');
    option.value = row.k;
    option.textContent = `${row.k} 個特徵`;
    count.appendChild(option);
  });
  count.value = String(state.k);
  $('methodText').textContent = explanations[state.method];
  $('rowCount').textContent = `${rows.length} 組已計算結果`;
  $('metricK').textContent = `${selected.k} 個特徵`;
  $('metricRmse').textContent = fmt(selected.cv_rmse);
  $('metricR2').textContent = fmt(selected.cv_r2, 3);
  $('insightK').textContent = selected.k;
  $('features').textContent = selected.features.join(' · ');
  const formal = candidateFor(state.method);
  const exact = method.exact;
  $('caution').textContent = exact?.available
    ? `目前是逐 k 探索路徑。${names[state.method]} 正式選出 k=${exact.k}，其驗證 RMSE 為 ${fmt(exact.cv_rmse)}；不要把兩種分數混為一談。`
    : selected.k === method.best_path_k ? '★ 這是此路徑的最低驗證 RMSE 組合。' : '金色列標示此路徑的最低驗證 RMSE 組合。';
  $('methodConclusionLabel').textContent = formal ? '此方法的正式驗證選擇' : '此方法目前沒有正式候選';
  $('methodConclusionTitle').textContent = formal ? `${names[state.method]} · ${formal.k} 個欄位` : `${names[state.method]} · 無有效組合`;
  $('methodConclusionFeatures').textContent = formal?.features.join(' · ') ?? '—';
  $('methodCvRmse').textContent = formal ? fmt(formal.cv_rmse) : '—';
  $('methodCvR2').textContent = formal ? fmt(formal.cv_r2, 3) : '—';
  $('methodTestMae').textContent = formal ? fmt(formal.test.mae) : '—';
  $('methodTestRmse').textContent = formal ? fmt(formal.test.rmse) : '—';
  $('methodTestR2').textContent = formal ? fmt(formal.test.r2, 3) : '—';
  const baseline = candidateFor('All Features');
  const delta = formal && baseline ? formal.test.rmse - baseline.test.rmse : null;
  $('methodComparison').textContent = formal && baseline
    ? `留出期 RMSE 比全特徵基準${delta >= 0 ? '高' : '低'} ${fmt(Math.abs(delta))} 輛；此值只作結果檢查，不用來重新選方案。`
    : '此方法未產生可比較的正式候選。';
  const winner = task.winner;
  $('overallConclusionTitle').textContent = winner ? `${names[winner.method]} · ${winner.k} 個欄位` : '沒有有效推薦';
  $('overallConclusionFeatures').textContent = winner?.features.join(' · ') ?? '—';
  $('overallEvidence').textContent = winner
    ? `三折時間驗證 RMSE ${fmt(winner.cv_rmse)}，R² ${fmt(winner.cv_r2, 3)}；留出期 RMSE ${fmt(winner.test.rmse)}、R² ${fmt(winner.test.r2, 3)}。`
    : '請檢查分析資料與設定。';
  $('overallLimit').textContent = winner
    ? `營運日留出期 R² ${fmt(winner.functioning_test?.r2, 3)}；停駛時段 ${winner.nonfunctioning_test_count} 筆，負租借量預測 ${winner.negative_prediction_count} 筆。氣象／營運資訊可得性是條件式前提；留出期曾在選題時看過。`
    : '—';
  draw('rmseChart', rows, 'cv_rmse');
  draw('r2Chart', rows, 'cv_r2');
  const body = $('rows');
  body.replaceChildren();
  rows.forEach(row => {
    const tr = document.createElement('tr');
    tr.dataset.k = row.k;
    tr.className = [row.k === state.k ? 'active' : '', row.k === method.best_path_k ? 'recommended' : ''].filter(Boolean).join(' ');
    for (const [value, cls] of [[row.k, ''], [row.features.join('、'), 'feature-cell'], [fmt(row.cv_rmse), ''], [fmt(row.cv_r2, 3), '']]) {
      const td = document.createElement('td');
      td.textContent = value;
      if (cls) td.className = cls;
      tr.appendChild(td);
    }
    const status = document.createElement('td');
    if (row.k === method.best_path_k) {
      const badge = document.createElement('span');
      badge.className = 'best-badge';
      badge.textContent = '★ 路徑最低';
      status.appendChild(badge);
    } else if (state.method === 'All Features') {
      status.textContent = '單點基準';
    }
    tr.appendChild(status);
    tr.tabIndex = 0;
    tr.addEventListener('click', () => choose(row.k));
    tr.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); choose(row.k); }
    });
    body.appendChild(tr);
  });
  renderComparison();
}

function setMode(mode) {
  const official = mode === 'official';
  for (const id of ['officialHero', 'officialControls', 'officialStudy', 'officialLower', 'officialComparison', 'officialConclusion']) $(id).hidden = !official;
  $('customHero').hidden = official;
  $('customView').hidden = official;
  $('officialMode').setAttribute('aria-pressed', String(official));
  $('customMode').setAttribute('aria-pressed', String(!official));
}

function setSource(sourceType) {
  const file = sourceType === 'file';
  $('fileSourceField').hidden = !file;
  $('urlSourceField').hidden = file;
  $('fileSource').setAttribute('aria-pressed', String(file));
  $('urlSource').setAttribute('aria-pressed', String(!file));
  $('customStatus').textContent = file ? '請選擇本機 CSV；分析引擎尚未啟用。' : '請輸入允許跨來源讀取的 CSV URL；分析引擎尚未啟用。';
}

$('problem').addEventListener('change', event => { state.task = event.target.value; state.k = null; render(); });
$('method').addEventListener('change', event => { state.method = event.target.value; state.k = null; render(); });
$('featureCount').addEventListener('change', event => choose(Number(event.target.value)));
$('officialMode').addEventListener('click', () => setMode('official'));
$('customMode').addEventListener('click', () => setMode('custom'));
$('fileSource').addEventListener('click', () => setSource('file'));
$('urlSource').addEventListener('click', () => setSource('url'));
$('customFile').addEventListener('change', event => {
  $('customStatus').textContent = event.target.files.length
    ? `已選擇「${event.target.files[0].name}」；分析引擎尚未啟用。` : '尚未選擇 CSV。';
});
$('customUrl').addEventListener('input', event => {
  $('customStatus').textContent = event.target.value.trim() ? '已輸入 CSV URL；尚未讀取或分析。' : '尚未輸入 CSV URL。';
});
document.querySelectorAll('.glass,.hero').forEach(element => element.addEventListener('pointermove', event => {
  const box = element.getBoundingClientRect();
  element.style.setProperty('--x', `${event.clientX - box.left}px`);
  element.style.setProperty('--y', `${event.clientY - box.top}px`);
}));

fetch('data/results.json').then(response => {
  if (!response.ok) throw new Error(`結果資料載入失敗（HTTP ${response.status}）`);
  return response.json();
}).then(report => {
  state.report = report;
  $('statRows').textContent = (report.dataset_summary?.rows ?? 8760).toLocaleString();
  $('statRaw').textContent = report.dataset_summary?.raw_candidate_features ?? 12;
  $('statEncoded').textContent = report.dataset_summary?.encoded_features ?? report.tasks.current_hour.encoded_feature_count;
  $('statTasks').textContent = Object.keys(report.tasks).length;
  render();
}).catch(error => {
  $('methodText').textContent = error.message;
  $('rowCount').textContent = '資料載入失敗';
});
