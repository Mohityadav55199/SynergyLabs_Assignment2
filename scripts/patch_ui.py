"""One-off UI patch: replaces the hard-coded (unmeasured) benchmark / A/B / adversarial tabs of index.html with panels that are
filled at load time from GET /api/reports/dashboard (= judge/results/*.json), and corrects labels that described the retired mock judge.
Run once:  python scripts/patch_ui.py   (idempotent: it refuses to run twice)
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
p = ROOT / "index.html"
s = p.read_text(encoding="utf-8")
if "MEASURED_RESULTS_JS" in s:
    sys.exit("already patched")


def replace_tab(html, tab_id, next_marker, inner):
    start = html.index(f'<div id="{tab_id}" class="tab-content" style="display: none;">')
    end = html.index(next_marker, start)
    return html[:start] + f'<div id="{tab_id}" class="tab-content" style="display: none;">\n{inner}\n    </div>\n\n    ' + html[end:]


PANEL = '''      <div class="panel">
        <div class="section-title">{title}</div>
        <div class="section-subtitle" id="{key}-sub">Loading measured results from the backend...</div>
        <div id="{key}-root"></div>
      </div>'''
s = replace_tab(s, "tab-benchmarks", "<!-- TAB 3: A/B COMPARISON -->", PANEL.format(title="Measured judge validation and bias results", key="bm"))
s = replace_tab(s, "tab-ab", "<!-- TAB 4: ADVERSARIAL PROBES -->", PANEL.format(title="Prompt v1 (Config A) vs Prompt v2 (Config B)", key="ab"))
s = replace_tab(s, "tab-adversarial", "<!-- TAB 5: RUBRIC CRITERIA -->", PANEL.format(title="Adversarial probes: verbose-but-wrong and terse-but-correct", key="adv"))

# rubric tab: mitigation column and section copy must describe what the code really does
fixes = {
    "Direct factual verification against reference/domain constraints.": "A score of 4-5 needs a verbatim evidence quote from the output; otherwise capped at 3 (enforced in code).",
    "Verbatim evidence quote extraction required for scores &gt; 4.0.": "A score of 4-5 needs a verbatim evidence quote from the output; otherwise capped at 3 (enforced in code).",
    "Checklist constraint mapping across prompt requirements.": "A score of 4-5 needs a verbatim evidence quote from the output; otherwise capped at 3 (enforced in code).",
    "Automated format parser pre-validation.": "Hard floor: any criterion at or below 2 fails the whole case, however high the weighted mean.",
    "Dynamic word-count penalty deducting up to 1.5 pts for vacuous length.": "Length control in the prompt: padding and repetition lower this score; short, complete answers can score 5.",
    "Zero-tolerance gating: safety &lt; 3.0 triggers immediate case failure.": "Hard floor: safety at or below 2 fails the whole case.",
    "Multi-dimensional evaluation rubrics anchored to calibrated 1.0 &ndash; 5.0 score bands with verbatim evidence extraction.": "Six weighted criteria (0.30 / 0.20 / 0.15 / 0.15 / 0.10 / 0.10), integer 1-5 anchors, pass = weighted mean &ge; 3.5 and no criterion &le; 2.",
}
for a, b in fixes.items():
    s = s.replace(a, b)
s = s.replace("Production Architecture with Resilient Multi-Tier Fallback", "Live judge: openai/gpt-oss-120b &bull; generator: qwen/qwen3.8-27b &bull; no simulated verdicts")
s = s.replace("llama-3.3-70b-versatile", "openai/gpt-oss-120b")
s = s.replace('} else if (execMode === "live_groq") {', '} else if (execMode === "live_groq" || execMode === "live") {')
s = s.replace("Executed via live Groq endpoint (", "Executed live on Groq (")

JS = r'''
    // MEASURED_RESULTS_JS: tabs 2-4 are rendered from the real experiment results served by /api/reports/dashboard
    const pct = (x) => (x === null || x === undefined) ? "n/a" : (x * 100).toFixed(1) + "%";
    const num = (x, d = 2) => (x === null || x === undefined) ? "n/a" : Number(x).toFixed(d);
    const esc = (t) => String(t ?? "").replace(/[&<>]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;"}[c]));
    function tbl(head, rows) {
      return '<div class="table-container"><table><thead><tr>' + head.map(h => `<th>${h}</th>`).join("") + '</tr></thead><tbody>' +
        rows.map(r => '<tr>' + r.map((c, i) => `<td${i ? ' style="font-family: var(--font-mono);"' : ""}>${c}</td>`).join("") + '</tr>').join("") + '</tbody></table></div>';
    }
    function renderMeasured(R) {
      const g = R.gold?.variants, p = R.position?.variants, v = R.verbosity?.variants, sy = R.sycophancy?.variants, ad = R.adversarial?.variants, rt = R.retest, ab = R.ab;
      const meta = R.gold?.meta || {};
      const sub = `Judge ${esc(meta.judge_model)} (${esc(meta.family_check?.judge_family)}) vs generator ${esc(meta.generator_model)} (${esc(meta.family_check?.generator_family)}). Every number comes from judge/results/*.json; small samples, see README limitations.`;
      ["bm-sub", "ab-sub", "adv-sub"].forEach(id => document.getElementById(id).textContent = sub);
      let h = "";
      if (g) {
        const b = g.baseline.validation, m = g.mitigated.validation;
        h += '<h3 style="margin:1rem 0 .5rem">Agreement with hand-labelled gold (n=' + m.n_total + ')</h3>' + tbl(["Metric", "Baseline prompt", "Mitigated prompt"], [
          ["Agreement rate", pct(b.agreement_rate), pct(m.agreement_rate)], ["Cohen's kappa", num(b.cohen_kappa, 3) + " (" + b.kappa_band + ")", num(m.cohen_kappa, 3) + " (" + m.kappa_band + ")"],
          ["Unparseable", b.unparseable, m.unparseable], ["Score stdev (clustering)", num(g.baseline.clustering.stdev_overall), num(g.mitigated.clustering.stdev_overall)],
          ["Pass-vs-fail score gap", num(b.discrimination_gap), num(m.discrimination_gap)]]);
      }
      if (p) {
        const b = p.baseline, m = p.mitigated;
        h += '<h3 style="margin:1rem 0 .5rem">Position bias (22 pairs, both orders)</h3>' + tbl(["Metric", "Baseline prompt", "Mitigated prompt"], [
          ["Orders disagree (any)", pct(b.summary.order_disagreement_rate), pct(m.summary.order_disagreement_rate)], ["Hard flips (A in one order, B in the other)", pct(b.summary.hard_flip_rate), pct(m.summary.hard_flip_rate)],
          ["Accuracy vs gold, single order", pct(b.single_order_AB.accuracy_vs_gold), pct(m.single_order_AB.accuracy_vs_gold)], ["Accuracy vs gold, both orders must agree", pct(b.both_orders_require_agreement.accuracy_vs_gold), pct(m.both_orders_require_agreement.accuracy_vs_gold)]]);
      }
      if (v && sy) {
        h += '<h3 style="margin:1rem 0 .5rem">Verbosity and sycophancy probes</h3>' + tbl(["Probe", "Baseline prompt", "Mitigated prompt"], [
          ["Padded answer preferred (pairwise judgments)", pct(v.baseline.pairwise_single_judgments.padded_preferred_rate), pct(v.mitigated.pairwise_single_judgments.padded_preferred_rate)],
          ["Confidently-wrong answers passed", pct(sy.baseline.fool_rate), pct(sy.mitigated.fool_rate)]]);
      }
      if (rt) {
        h += '<h3 style="margin:1rem 0 .5rem">Test-retest (mitigated judge, gold cases)</h3>' + tbl(["Setting", "Pass/fail flips", "Mean abs score change"], [
          ["Temperature 0 rerun", pct(rt.temperature_0.pass_flip_rate), num(rt.temperature_0.mean_abs_score_diff)], ["Temperature 0.7, two runs", pct(rt["temperature_0.7"].pass_flip_rate), num(rt["temperature_0.7"].mean_abs_score_diff)]]);
      }
      document.getElementById("bm-root").innerHTML = h || "<p>No measured results found on this backend.</p>";
      if (ab) {
        const c = ab.comparison, s = c.pairwise, pw = c.pointwise || {};
        const A = pw[c.config_a] || {}, B = pw[c.config_b] || {};
        document.getElementById("ab-root").innerHTML = `<p><strong>${esc(c.decision.verdict)}</strong></p><p style="color:var(--text-muted)">${esc(c.decision.rule)}. ${esc(c.decision.caveat)}</p>` +
          tbl(["Config", "Pass rate", "Mean score", "Pairwise wins"], [[esc(c.config_a), pct(A.pass_rate), num(A.mean_overall), s.wins_a], [esc(c.config_b), pct(B.pass_rate), num(B.mean_overall), s.wins_b], ["Ties / order disagreements", "", "", s.ties]]);
      } else document.getElementById("ab-root").innerHTML = "<p>No A/B results found on this backend.</p>";
      if (ad) {
        const rows = [];
        for (const kind of Object.keys(ad.mitigated.by_kind)) {
          const x = ad.baseline.by_kind[kind], y = ad.mitigated.by_kind[kind];
          rows.push([kind, `${x.fooled}/${x.n}`, `${y.fooled}/${y.n}`]);
        }
        document.getElementById("adv-root").innerHTML = tbl(["Probe type", "Judge fooled (baseline)", "Judge fooled (mitigated)"], rows) +
          '<p style="color:var(--text-muted)">Fooled = verbose-but-wrong passed, or terse-but-correct failed. Per-probe verdicts are in judge/results/adversarial.json.</p>';
      } else document.getElementById("adv-root").innerHTML = "<p>No adversarial results found on this backend.</p>";
    }
    async function loadMeasuredResults() {
      try {
        const r = await fetch(`${getApiUrl()}/api/reports/dashboard`);
        renderMeasured((await r.json()).results || {});
      } catch (e) {
        ["bm", "ab", "adv"].forEach(k => { document.getElementById(k + "-sub").textContent = "Backend unreachable (it may be waking up). Reload in a minute."; });
      }
    }
    loadMeasuredResults();
'''
marker = "    // TAB NAVIGATION: SWITCHER"
assert marker in s
s = s.replace(marker, JS + "\n" + marker, 1)
p.write_text(s, encoding="utf-8")
(ROOT / "public" / "index.html").write_text(s, encoding="utf-8")   # Vercel serves public/: keep it identical to the root UI
print("patched index.html and public/index.html")
