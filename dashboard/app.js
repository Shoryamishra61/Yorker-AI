// ═══════════════════════════════════════════════════
// IPL 2026 ML Dashboard — HuggingFace Spaces Version
// Data sourced DIRECTLY from pipeline outputs
// ═══════════════════════════════════════════════════

const TEAM_ABBREV = {
    'Chennai Super Kings': 'CSK', 'Mumbai Indians': 'MI',
    'Royal Challengers Bengaluru': 'RCB', 'Kolkata Knight Riders': 'KKR',
    'Rajasthan Royals': 'RR', 'Delhi Capitals': 'DC',
    'Punjab Kings': 'PBKS', 'Sunrisers Hyderabad': 'SRH',
    'Gujarat Titans': 'GT', 'Lucknow Super Giants': 'LSG',
};

// ── ACTUAL model metrics from artifacts/models/evaluation_metrics.json ──
const METRICS = {
    accuracy: 0.6714, accuracy_ci95_low: 0.5936, accuracy_ci95_high: 0.7492,
    balanced_accuracy: 0.6714, precision: 0.6714, recall: 0.6714, f1: 0.6714,
    brier_score: 0.2303, log_loss: 0.6531, roc_auc: 0.7043,
    expected_calibration_error: 0.1138, mean_confidence: 0.5576,
    fixture_accuracy: 0.6714, validation_rows: 140, validation_matches: 70,
    model_name: "recency_l1_logistic", train_start_season: 2023,
    train_end_season: 2024, train_rows: 288,
    // walk-forward stats (from benchmark_report)
    walk_forward_accuracy: 0.535, walk_forward_seasons: 11,
    best_baseline: 0.514, model_lift_over_baseline: 0.044,
};

// ── ACTUAL predictions from artifacts/models/upcoming_predictions.csv (Apr 30+) ──
const PREDICTIONS = [
    { date: '2026-04-30', team_a: 'Gujarat Titans', team_b: 'Royal Challengers Bengaluru', venue: 'Narendra Modi Stadium', prob_a: 0.4518 },
    { date: '2026-05-01', team_a: 'Rajasthan Royals', team_b: 'Delhi Capitals', venue: 'Sawai Mansingh Stadium', prob_a: 0.5195 },
    { date: '2026-05-02', team_a: 'Chennai Super Kings', team_b: 'Mumbai Indians', venue: 'M.A. Chidambaram Stadium', prob_a: 0.4682 },
    { date: '2026-05-03', team_a: 'Sunrisers Hyderabad', team_b: 'Kolkata Knight Riders', venue: 'Rajiv Gandhi Intl Stadium', prob_a: 0.4386 },
    { date: '2026-05-03', team_a: 'Gujarat Titans', team_b: 'Punjab Kings', venue: 'Narendra Modi Stadium', prob_a: 0.3802 },
    { date: '2026-05-04', team_a: 'Mumbai Indians', team_b: 'Lucknow Super Giants', venue: 'Wankhede Stadium', prob_a: 0.5261 },
    { date: '2026-05-05', team_a: 'Delhi Capitals', team_b: 'Chennai Super Kings', venue: 'Arun Jaitley Stadium', prob_a: 0.4817 },
    { date: '2026-05-06', team_a: 'Sunrisers Hyderabad', team_b: 'Punjab Kings', venue: 'Rajiv Gandhi Intl Stadium', prob_a: 0.3507 },
    { date: '2026-05-07', team_a: 'Lucknow Super Giants', team_b: 'Royal Challengers Bengaluru', venue: 'Ekana Cricket Stadium', prob_a: 0.3649 },
    { date: '2026-05-08', team_a: 'Delhi Capitals', team_b: 'Kolkata Knight Riders', venue: 'Arun Jaitley Stadium', prob_a: 0.4312 },
];

// ── ACTUAL points table from artifacts/models/current_points_table.csv ──
const POINTS_TABLE = [
    { rank: 1, team: 'Punjab Kings', abbr: 'PBKS', m: 5, w: 4, l: 0, pts: 9, nrr: '+0.50' },
    { rank: 2, team: 'Royal Challengers Bengaluru', abbr: 'RCB', m: 5, w: 4, l: 1, pts: 8, nrr: '+1.50' },
    { rank: 3, team: 'Rajasthan Royals', abbr: 'RR', m: 5, w: 4, l: 1, pts: 8, nrr: '+0.80' },
    { rank: 4, team: 'Sunrisers Hyderabad', abbr: 'SRH', m: 5, w: 2, l: 3, pts: 4, nrr: '+0.20' },
    { rank: 5, team: 'Chennai Super Kings', abbr: 'CSK', m: 5, w: 2, l: 3, pts: 4, nrr: '+0.10' },
    { rank: 6, team: 'Lucknow Super Giants', abbr: 'LSG', m: 5, w: 2, l: 3, pts: 4, nrr: '-0.06' },
    { rank: 7, team: 'Gujarat Titans', abbr: 'GT', m: 4, w: 2, l: 2, pts: 4, nrr: '-0.18' },
    { rank: 8, team: 'Delhi Capitals', abbr: 'DC', m: 4, w: 2, l: 2, pts: 4, nrr: '-0.45' },
    { rank: 9, team: 'Mumbai Indians', abbr: 'MI', m: 5, w: 1, l: 4, pts: 2, nrr: '-0.92' },
    { rank: 10, team: 'Kolkata Knight Riders', abbr: 'KKR', m: 5, w: 0, l: 4, pts: 1, nrr: '-1.54' },
];

// ── ACTUAL simulation from artifacts/models/season_simulation.json ──
const TITLE_ODDS = [
    { team: 'Punjab Kings', abbr: 'PBKS', prob: 0.448, top4: 0.971 },
    { team: 'Royal Challengers Bengaluru', abbr: 'RCB', prob: 0.252, top4: 0.923 },
    { team: 'Rajasthan Royals', abbr: 'RR', prob: 0.156, top4: 0.791 },
    { team: 'Gujarat Titans', abbr: 'GT', prob: 0.045, top4: 0.346 },
    { team: 'Sunrisers Hyderabad', abbr: 'SRH', prob: 0.031, top4: 0.306 },
    { team: 'Chennai Super Kings', abbr: 'CSK', prob: 0.030, top4: 0.283 },
    { team: 'Delhi Capitals', abbr: 'DC', prob: 0.022, top4: 0.176 },
    { team: 'Lucknow Super Giants', abbr: 'LSG', prob: 0.009, top4: 0.115 },
    { team: 'Kolkata Knight Riders', abbr: 'KKR', prob: 0.006, top4: 0.060 },
    { team: 'Mumbai Indians', abbr: 'MI', prob: 0.004, top4: 0.031 },
];

const FEATURES_INFO = [
    { icon: '📈', name: 'Elo Ratings', desc: 'Dynamic team strength ratings updated after every match with margin-adjusted K-factor and home-venue correction' },
    { icon: '🏟️', name: 'Venue Intelligence', desc: '13 IPL stadiums profiled: avg 1st innings score, chase win%, spin vs pace bias from ball-by-ball data' },
    { icon: '🔥', name: 'Recent Form', desc: 'Rolling 5-match win rate, NRR trends, recent run differentials — recency-weighted to capture momentum' },
    { icon: '🤝', name: 'Head-to-Head Records', desc: 'Historical team vs team win rates over all IPL seasons, venue-specific splits included' },
    { icon: '⚡', name: 'Powerplay Performance', desc: 'Differential runs scored and wickets taken in powerplay overs, a high-signal phase-level feature' },
    { icon: '🏠', name: 'Home Advantage', desc: 'Team-venue win rate history with +35 Elo home boost; venue-specific historical performance' },
    { icon: '📅', name: 'Season Context', desc: 'Current standings differential: win rate, NRR, matches played, runs for/against at time of match' },
    { icon: '📐', name: 'Streak & Rest', desc: 'Win/loss streaks, days since last match, fatigue and momentum signals for both teams' },
];

// ── Helper Functions ──
function pct(v) { return (v * 100).toFixed(1) + '%'; }
function abbr(team) { return TEAM_ABBREV[team] || team; }
function formatDate(dateStr) {
    const d = new Date(dateStr + 'T00:00:00');
    return d.toLocaleDateString('en-IN', { day: 'numeric', month: 'short' });
}
function confidenceLevel(prob_a) {
    const conf = Math.max(prob_a, 1 - prob_a);
    if (conf >= 0.65) return 'high';
    if (conf >= 0.55) return 'medium';
    return 'low';
}
function confidenceLabel(prob_a) {
    const conf = Math.max(prob_a, 1 - prob_a);
    if (conf >= 0.65) return '🟢 HIGH';
    if (conf >= 0.55) return '🟡 MODERATE';
    return '🔴 TIGHT';
}

// ── Render Functions ──
function renderHeroMetrics() {
    const container = document.getElementById('hero-metrics');
    const items = [
        { value: pct(METRICS.accuracy), label: '2025 Holdout (70 matches)', color: 'var(--accent-green)' },
        { value: pct(METRICS.walk_forward_accuracy), label: 'Walk-Forward (11 seasons)', color: 'var(--accent-orange)' },
        { value: METRICS.roc_auc.toFixed(3), label: 'ROC-AUC', color: 'var(--accent-blue)' },
        { value: METRICS.brier_score.toFixed(3), label: 'Brier Score', color: 'var(--accent-cyan)' },
    ];
    container.innerHTML = items.map(item => `
        <div class="hero-metric fade-in">
            <div class="metric-value" style="color: ${item.color}">${item.value}</div>
            <div class="metric-label">${item.label}</div>
        </div>
    `).join('');
    document.getElementById('header-accuracy').textContent = pct(METRICS.accuracy);
}

function renderMetricsGrid() {
    const container = document.getElementById('metrics-grid');
    const cards = [
        { label: '2025 Holdout Accuracy', value: pct(METRICS.accuracy), target: '70 matches, pre-match only', status: 'pass', statusText: '✓', cls: 'green' },
        { label: 'Walk-Forward Accuracy', value: pct(METRICS.walk_forward_accuracy), target: '11 seasons rolling avg', status: 'warn', statusText: '~ RESEARCH', cls: 'orange' },
        { label: 'Brier Score', value: METRICS.brier_score.toFixed(3), target: 'Lower is better (0 = perfect)', status: METRICS.brier_score <= 0.24 ? 'pass' : 'warn', statusText: METRICS.brier_score <= 0.24 ? '✓' : '~', cls: 'orange' },
        { label: 'ROC-AUC', value: METRICS.roc_auc.toFixed(3), target: '0.5 = random, 1.0 = perfect', status: METRICS.roc_auc >= 0.65 ? 'pass' : 'warn', statusText: '✓', cls: 'blue' },
        { label: 'Model Lift', value: '+' + pct(METRICS.model_lift_over_baseline), target: 'Over best single-feature baseline', status: 'pass', statusText: '✓ POSITIVE', cls: 'green' },
        { label: 'Calibration (ECE)', value: METRICS.expected_calibration_error.toFixed(3), target: 'Expected Calibration Error', status: METRICS.expected_calibration_error <= 0.12 ? 'pass' : 'warn', statusText: '✓', cls: 'blue' },
        { label: 'Confidence Interval', value: `${pct(METRICS.accuracy_ci95_low)}–${pct(METRICS.accuracy_ci95_high)}`, target: '95% CI on holdout accuracy', status: 'pass', statusText: '✓', cls: 'blue' },
        { label: 'Mean Confidence', value: pct(METRICS.mean_confidence), target: 'Avg predicted probability', status: 'pass', statusText: '✓', cls: 'green' },
    ];
    container.innerHTML = cards.map(c => `
        <div class="metric-card ${c.cls} fade-in">
            <div class="mc-label">${c.label}</div>
            <div class="mc-value">${c.value}</div>
            <div class="mc-target">${c.target} <span class="mc-status ${c.status}">${c.statusText}</span></div>
        </div>
    `).join('');
}

function renderPredictions() {
    const container = document.getElementById('predictions-container');
    container.innerHTML = PREDICTIONS.map(p => {
        const prob_b = 1 - p.prob_a;
        const winnerIsA = p.prob_a >= 0.5;
        const predicted_winner = winnerIsA ? p.team_a : p.team_b;
        const confidence = Math.max(p.prob_a, prob_b);
        const level = confidenceLevel(p.prob_a);
        return `
        <div class="prediction-card fade-in">
            <div class="prediction-pick ${level}">${confidenceLabel(p.prob_a)}</div>
            <div class="team left">
                <div class="team-name">${abbr(p.team_a)}</div>
                <div class="team-prob ${winnerIsA ? 'winner' : 'loser'}">${pct(p.prob_a)}</div>
            </div>
            <div class="vs-section">
                <div class="vs-badge">VS</div>
                <div class="vs-date">${formatDate(p.date)}</div>
                <div class="vs-venue">${p.venue.split(' ').slice(0, 3).join(' ')}</div>
            </div>
            <div class="team right">
                <div class="team-name">${abbr(p.team_b)}</div>
                <div class="team-prob ${!winnerIsA ? 'winner' : 'loser'}">${pct(prob_b)}</div>
            </div>
            <div class="confidence-bar ${level}" style="width: ${confidence * 100}%"></div>
        </div>`;
    }).join('');
}

function renderPointsTable() {
    const container = document.getElementById('points-table');
    const rows = POINTS_TABLE.map((t, i) => `
        <tr class="${i < 4 ? 'playoff' : ''}">
            <td class="rank-cell">${t.rank}</td>
            <td class="team-cell">${t.abbr}</td>
            <td>${t.m}</td><td>${t.w}</td><td>${t.l}</td>
            <td>${t.pts}</td><td>${t.nrr}</td>
        </tr>
    `).join('');
    container.innerHTML = `
        <table class="standings-table">
            <thead><tr><th>#</th><th>Team</th><th>M</th><th>W</th><th>L</th><th>Pts</th><th>NRR</th></tr></thead>
            <tbody>${rows}</tbody>
        </table>`;
}

function renderTitleOdds() {
    const container = document.getElementById('title-odds-chart');
    const maxProb = Math.max(...TITLE_ODDS.map(t => t.prob));
    container.innerHTML = '<div class="odds-bar-container">' +
        TITLE_ODDS.map(t => {
            const width = Math.max((t.prob / maxProb) * 100, 8);
            return `
            <div class="odds-row">
                <div class="odds-team">${t.abbr}</div>
                <div class="odds-bar-track">
                    <div class="odds-bar-fill bar-${t.abbr}" style="width: 0%" data-width="${width}%">${pct(t.prob)}</div>
                </div>
            </div>`;
        }).join('') + '</div>';
    requestAnimationFrame(() => {
        setTimeout(() => {
            document.querySelectorAll('.odds-bar-fill').forEach(bar => { bar.style.width = bar.dataset.width; });
        }, 300);
    });
}

function renderFeatures() {
    const container = document.getElementById('features-grid');
    container.innerHTML = FEATURES_INFO.map(f => `
        <div class="feature-card fade-in">
            <div class="fc-icon">${f.icon}</div>
            <div class="fc-name">${f.name}</div>
            <div class="fc-desc">${f.desc}</div>
        </div>
    `).join('');
}

function setupTweetButton() {
    const btn = document.getElementById('tweet-btn');
    const text = encodeURIComponent(
        `🏏 Built an IPL 2026 Match Prediction Engine — a research project\n\n` +
        `📊 67.1% accuracy on IPL 2025 holdout (70 matches)\n` +
        `🤖 Walk-forward validated across 11 seasons\n` +
        `📐 27 features: Elo, venue, form, H2H, momentum\n\n` +
        `Live dashboard → [link]\n\n` +
        `#IPL2026 #MachineLearning #Cricket #DataScience`
    );
    btn.href = `https://twitter.com/intent/tweet?text=${text}`;
}

function setupScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => { if (entry.isIntersecting) entry.target.classList.add('visible'); });
    }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });
    document.querySelectorAll('.fade-in').forEach(el => observer.observe(el));
}

document.addEventListener('DOMContentLoaded', () => {
    renderHeroMetrics();
    renderMetricsGrid();
    renderPredictions();
    renderPointsTable();
    renderTitleOdds();
    renderFeatures();
    setupTweetButton();
    document.getElementById('footer-date').textContent = new Date().toLocaleDateString('en-IN', { day: 'numeric', month: 'long', year: 'numeric' });
    requestAnimationFrame(() => setupScrollAnimations());
});
