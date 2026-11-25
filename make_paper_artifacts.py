"""
Generate every number, table and figure used in the camera-ready paper from
the raw result files, so nothing in the paper is transcribed by hand.

Writes into $AVSC_PAPER_DIR/generated/ (default ../paper/generated/):
  numbers.tex        \\newcommand macros for numbers quoted in the text
  tab_*.tex          tables
and figures into $AVSC_PAPER_DIR (default ../paper/).

Usage: make_paper_artifacts.py
"""
import csv
import glob
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from avsc.fast_samplers import csr_from_cache

PAPER = os.environ.get("AVSC_PAPER_DIR", os.path.join(HERE, "..", "paper"))
GEN = os.path.join(PAPER, "generated")
os.makedirs(GEN, exist_ok=True)
GRAPHS = ["jazz", "pgp", "celegans", "power", "com-dblp", "com-amazon",
          "roadnet-ca", "com-youtube", "as-skitter"]
DISPLAY = {"jazz": "jazz", "pgp": "PGP", "celegans": "C.\\ elegans", "power": "power",
           "com-dblp": "com-DBLP", "com-amazon": "com-Amazon", "roadnet-ca": "roadNet-CA",
           "com-youtube": "com-Youtube", "as-skitter": "as-Skitter"}
CITE = {"jazz": "gleiser2003jazz", "pgp": "boguna2004models", "celegans": "duch2005community",
        "power": "watts1998collective", "com-dblp": "yang2012defining",
        "com-amazon": "yang2012defining", "roadnet-ca": "leskovec2009community",
        "com-youtube": "yang2012defining", "as-skitter": "leskovec2005graphs"}

MACROS = {}


def macro(name, value):
    assert name.isalpha(), name
    MACROS[name] = value


def thousands(x):
    return f"{int(x):,}".replace(",", "{,}")


def ground_truth():
    table = {}
    times = {}
    with open(os.path.join(HERE, "tools_gt", "ground_truth_kcliques.csv")) as f:
        for row in csv.DictReader(f):
            assert row["tools_agree_exactly"] == "True"
            table[(row["graph"], int(row["k"]))] = int(row["count"])
            times[(row["graph"], int(row["k"]))] = float(row["kclist_wall_s"])
    return table, times


def graph_stats():
    stats = {}
    for name in GRAPHS:
        g = csr_from_cache(os.path.join(HERE, "data", "csr", name + ".npz"))
        g.build()
        wedges = int(np.sum(g.degree * (g.degree - 1) // 2))
        stats[name] = {"n": g.n, "m": g.m, "W": wedges, "maxdeg": int(g.degree.max())}
    return stats


from decimal import ROUND_HALF_UP, Decimal


def prop3(count, n):
    """Proportion count/n to three decimals, exact, rounding half up."""
    return str((Decimal(int(count)) / Decimal(int(n))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def dec3(x):
    """Three decimals, rounding half up on the shortest decimal repr of x."""
    return str(Decimal(repr(float(x))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def floor1(x):
    """One decimal, rounded DOWN: for 'at least' statements."""
    return f"{math.floor(x * 10 + 1e-9) / 10:.1f}"


def ceil1(x):
    """One decimal, rounded UP: for 'within' / 'at most' statements."""
    return f"{math.ceil(x * 10 - 1e-9) / 10:.1f}"


def prop4(count, n):
    """Proportion count/n to four decimals, exact for n = 80, rounding half up."""
    return str((Decimal(int(count)) / Decimal(int(n))).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def row_count(row):
    return int(round(row["coverage"] * row["runs"]))


def power_of_ten(x):
    exponent = int(math.floor(math.log10(x)))
    mantissa = x / 10 ** exponent
    if abs(mantissa - 1.0) < 1e-12:
        return f"\\ensuremath{{10^{{{exponent}}}}}"
    return f"\\ensuremath{{{mantissa:g}\\times10^{{{exponent}}}}}"


def short_count(x):
    """Exact below 10^5; otherwise three significant figures with k/M/G."""
    x = int(x)
    if x < 100_000:
        return thousands(x)
    for scale, unit in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if x >= scale:
            value = x / scale
            digits = 2 - int(math.floor(math.log10(value)))
            return f"{value:.{max(digits, 0)}f}{unit}"


def compact(x):
    if x == 0:
        return "0"
    if x < 1_000_000:
        return thousands(x)
    exponent = int(math.floor(math.log10(x)))
    mantissa = x / 10 ** exponent
    return f"${mantissa:.2f}\\times10^{{{exponent}}}$"


def table_datasets(stats, gt):
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{2.2pt}",
             r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"graph & $n$ & $M$ & $\theta_3$ & $\theta_4$ & $\theta_5$ & $\theta_6$ & $\rho_{\mathrm{w}}$ \\",
             r"\midrule"]
    for i, name in enumerate(GRAPHS):
        s = stats[name]
        rho = s["W"] / (3 * gt[(name, 3)])
        cells = [f"{DISPLAY[name]}~\\cite{{{CITE[name]}}}", short_count(s["n"]), short_count(s["m"])]
        cells += [short_count(gt[(name, k)]) for k in (3, 4, 5, 6)]
        cells.append(f"{rho:.2f}")
        lines.append(" & ".join(cells) + r" \\")
        if i == 3:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{The nine graphs. $\theta_k$ is the exact number of $k$-cliques, "
              r"computed independently by kClist~\cite{danisch2018kclist} and "
              r"Pivoter~\cite{jain2020pivoter}, which agree exactly on all 36 values (shown to three "
              r"significant figures, with k, M, G for $10^3$, $10^6$, $10^9$; exact values are in the released "
              r"data); "
              r"$\rho_{\mathrm{w}} = W/(3\theta_3)$ is the concentration ratio of the "
              r"triangle (wedge) sampler. Graphs are simple and undirected, with self-loops "
              r"and duplicate edges removed at load time; the five large graphs are from "
              r"SNAP~\cite{leskovec2014snap}.}",
              r"\label{tab:datasets}", r"\end{table}"]
    return "\n".join(lines) + "\n"


ARMS = ["betting-eb", "betting-hoeffding", "prpl-eb", "prpl-hoeffding"]
MONITOR_HEAD = {"betting-eb": r"\bet-EB", "betting-hoeffding": r"\bet-H",
                "prpl-eb": r"\prpl-EB", "prpl-hoeffding": r"\prpl-H", "wald-peeking": "Wald"}


def human(x):
    """Sample counts: exact below 10^4, else three significant figures with k/M."""
    if x != x:
        return "--"
    if x < 1e4:
        return thousands(round(x))
    scale, unit = (1e3, "k") if x < 1e6 else (1e6, "M")
    value = x / scale
    digits = 2 - int(math.floor(math.log10(value)))
    return f"{value:.{max(digits, 0)}f}{unit}"


def human_pair(mean, sd):
    """Mean and SD in the same unit as the mean."""
    if mean < 1e4:
        return thousands(round(mean)), thousands(round(sd))
    scale, unit = (1e3, "k") if mean < 1e6 else (1e6, "M")
    digits = max(0, 2 - int(math.floor(math.log10(mean / scale))))
    if sd > 0:
        digits = max(digits, 1 - int(math.floor(math.log10(sd / scale))))
    return f"{mean / scale:.{digits}f}{unit}", f"{sd / scale:.{digits}f}{unit}"


def t_cell(row):
    if row["stopped"] == 0:
        text = "--"
    else:
        mean, sd = human_pair(row["T_mean"], row["T_sd"] if row["stopped"] > 1 else 0.0)
        text = f"{mean}\\,$\\pm$\\,{sd}"
    notes = []
    if row["capped"]:
        notes.append(f"c{int(row['capped'])}")
    if row["empty"]:
        notes.append(f"e{int(row['empty'])}")
    if notes:
        text += "$^{\\mathrm{" + ",".join(notes) + "}}$"
    return text


def lookup(summary, keys):
    match = summary
    for column, value in keys.items():
        match = match[match[column] == value]
    assert len(match) == 1, (keys, len(match))
    return match.iloc[0]


def triangle_complete(summary):
    expected = len(GRAPHS) * 2 * 5
    return len(summary) == expected and bool((summary["runs"] == 80).all())


def table_triangle_samples(frame, summary):
    import analysis_cr
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lrr|rrrr|r|rr}", r"\toprule",
             r"& & & \multicolumn{4}{c|}{anytime-valid (samples to certify)} & not valid & "
             r"\multicolumn{2}{c}{speedup} \\",
             r"graph & $\rho$ & $\epsilon$ & \bet-EB & \bet-H & \prpl-EB & \prpl-H & Wald & "
             r"\bet & \prpl \\", r"\midrule"]
    speed_bet = analysis_cr.paired_speedup(frame, ["graph", "eps"], "betting-hoeffding", "betting-eb")
    speed_prpl = analysis_cr.paired_speedup(frame, ["graph", "eps"], "prpl-hoeffding", "prpl-eb")
    for eps in (0.10, 0.30):
        for name in sorted(GRAPHS, key=lambda g: lookup(summary, {"graph": g, "eps": eps, "monitor": "prpl-eb"})["rho"]):
            cells = [DISPLAY[name], f"{lookup(summary, {'graph': name, 'eps': eps, 'monitor': 'prpl-eb'})['rho']:.2f}",
                     f"{eps:.2f}"]
            for monitor in ARMS + ["wald-peeking"]:
                cells.append(t_cell(lookup(summary, {"graph": name, "eps": eps, "monitor": monitor})))
            for table in (speed_bet, speed_prpl):
                match = table[(table["graph"] == name) & (table["eps"] == eps)]
                cells.append(f"{match.iloc[0]['ratio_of_means']:.2f}$\\times$" if len(match) and match.iloc[0]["pairs"] == 80 else "--")
            lines.append(" & ".join(cells) + r" \\")
        if eps == 0.10:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Samples to certify relative width $\epsilon$ (mean $\pm$ standard deviation "
              r"over the runs that stopped; $\NumRuns$ runs per cell, all on shared sample streams). "
              r"Superscripts count runs that reached the sample cap (c) or whose interval became empty (e). "
              r"Speedups are ratios of mean stopping times within a matched pair, "
              r"\bet-H/\bet-EB and \prpl-H/\prpl-EB, shown when all runs of both arms stopped. "
              r"Wald is the naive peeking rule, which is not anytime-valid (coverage in "
              r"Table~\ref{tab:coverage}).}",
              r"\label{tab:samples}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def coverage_cell(row, flag_low):
    text = prop3(row_count(row), row["runs"])
    if flag_low and row["cov_high"] < 0.90:
        text = f"\\textbf{{{text}}}$^{{*}}$"
    return text


def table_triangle_coverage(summary):
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{lr|rrrr|r}", r"\toprule",
             r"graph & $\epsilon$ & \bet-EB & \bet-H & \prpl-EB & \prpl-H & Wald \\", r"\midrule"]
    for eps in (0.10, 0.30):
        for name in sorted(GRAPHS, key=lambda g: lookup(summary, {"graph": g, "eps": eps, "monitor": "prpl-eb"})["rho"]):
            cells = [DISPLAY[name], f"{eps:.2f}"]
            for monitor in ARMS + ["wald-peeking"]:
                cells.append(coverage_cell(lookup(summary, {"graph": name, "eps": eps, "monitor": monitor}), True))
            lines.append(" & ".join(cells) + r" \\")
        if eps == 0.10:
            lines.append(r"\midrule")
    flagged = int((summary["cov_high"] < 0.90).sum())
    if flagged:
        flag_text = r"$^{*}$: the Wilson $95\%$ interval lies entirely below $0.90$."
    else:
        flag_text = r"No cell, for any monitor, has its Wilson $95\%$ interval entirely below $0.90$."
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Coverage over all $\NumRuns$ runs per cell (stopped, capped and empty runs "
              r"all count; target $1-\alpha = 0.90$). " + flag_text + "}",
              r"\label{tab:coverage}", r"\end{table}"]
    return "\n".join(lines) + "\n"


EPS_WORD = {0.10: "Ten", 0.30: "Thirty"}
PRED_CELLS = os.path.join(HERE, "results_cr", "prediction_cells.csv")
PRED_CURVE = os.path.join(HERE, "results_cr", "prediction_curve.csv")


def f_rho(rho):
    return rho ** 2 / (4.0 * (rho - 1.0))


def prediction_cells(summary):
    """Surrogate-predicted PrPl stopping times for every observed cell (cached)."""
    import pandas as pd
    from prpl_prediction import predicted_stop
    if os.path.exists(PRED_CELLS):
        return pd.read_csv(PRED_CELLS)
    rows = []
    for _, r in summary[summary["monitor"] == "prpl-eb"].iterrows():
        nu = 1.0 / r["rho"]
        rows.append({"graph": r["graph"], "eps": r["eps"], "rho": r["rho"],
                     "pred_h": predicted_stop(nu, r["eps"], "hoeffding", 3_000_000_000),
                     "pred_eb": predicted_stop(nu, r["eps"], "eb", 400_000_000)})
    table = pd.DataFrame(rows)
    table.to_csv(PRED_CELLS, index=False)
    return table


def prediction_curve():
    """Surrogate-predicted PrPl ratio on a grid of rho (cached)."""
    import pandas as pd
    from prpl_prediction import predicted_stop
    if os.path.exists(PRED_CURVE):
        return pd.read_csv(PRED_CURVE)
    rows = []
    for eps in (0.10, 0.30):
        for rho in np.geomspace(1.5, 300.0, 40):
            nu = 1.0 / rho
            h = predicted_stop(nu, eps, "hoeffding", 3_000_000_000)
            e = predicted_stop(nu, eps, "eb", 400_000_000)
            rows.append({"eps": eps, "rho": rho, "ratio": h / e})
    table = pd.DataFrame(rows)
    table.to_csv(PRED_CURVE, index=False)
    return table


PRED_LIMIT = os.path.join(HERE, "results_cr", "prediction_eps_limit.csv")
LIMIT_EPS = (0.3, 0.1, 0.03, 0.01, 0.003)
LIMIT_GRAPHS = ("com-amazon", "roadnet-ca")


def prediction_limit(summary):
    """Surrogate PrPl ratio as eps shrinks, for two graphs (cached)."""
    import pandas as pd
    from prpl_prediction import predicted_stop
    if os.path.exists(PRED_LIMIT):
        return pd.read_csv(PRED_LIMIT)
    rows = []
    for name in LIMIT_GRAPHS:
        rho = float(summary[(summary["graph"] == name) & (summary["monitor"] == "prpl-eb")]["rho"].iloc[0])
        for eps in LIMIT_EPS:
            h = predicted_stop(1.0 / rho, eps, "hoeffding", 3_000_000_000)
            e = predicted_stop(1.0 / rho, eps, "eb", 400_000_000)
            rows.append({"graph": name, "rho": rho, "eps": eps, "ratio": h / e, "f": f_rho(rho)})
    table = pd.DataFrame(rows)
    table.to_csv(PRED_LIMIT, index=False)
    return table


def limit_macros(summary):
    table = prediction_limit(summary)
    for name, word in (("com-amazon", "Amazon"), ("roadnet-ca", "Road")):
        part = table[table["graph"] == name].set_index("eps")
        macro(f"NumLimit{word}Rho", f"{part['rho'].iloc[0]:.2f}")
        macro(f"NumLimit{word}F", f"{part['f'].iloc[0]:.2f}")
        macro(f"NumLimit{word}Coarse", f"{part.loc[0.3, 'ratio']:.2f}")
        macro(f"NumLimit{word}Fine", f"{part.loc[0.003, 'ratio']:.2f}")


def ratio_table(frame, summary, numerator, denominator):
    import analysis_cr
    table = analysis_cr.paired_speedup(frame, ["graph", "eps"], numerator, denominator)
    rho = summary[summary["monitor"] == "prpl-eb"].set_index(["graph", "eps"])["rho"]
    table["rho"] = [rho[(g, e)] for g, e in zip(table["graph"], table["eps"])]
    assert (table["pairs"] == 80).all()
    return table.sort_values(["eps", "rho"])


def triangle_macros(frame, summary, cells):
    from scipy.stats import binomtest, spearmanr
    valid = summary[summary["monitor"] != "wald-peeking"]
    assert (summary["capped"] == 0).all() and (summary["empty"] == 0).all()
    worst = valid.loc[valid["coverage"].idxmin()]
    macro("NumCovMinValid", prop4(row_count(worst), worst["runs"]))
    macro("NumCovHighMinValid", dec3(valid["cov_high"].min()))
    for monitor, word in (("betting-eb", "BetEB"), ("betting-hoeffding", "BetH"),
                          ("prpl-eb", "PrplEB"), ("prpl-hoeffding", "PrplH"), ("wald-peeking", "Wald")):
        rows = frame[frame["monitor"] == monitor]
        covered = int(rows["covered"].sum())
        ci = binomtest(covered, len(rows)).proportion_ci(0.95, method="wilson")
        macro(f"NumCovPooled{word}", prop3(covered, len(rows)))
        macro(f"NumCovPooledLow{word}", dec3(ci.low))
        macro(f"NumCovPooledHigh{word}", dec3(ci.high))
    wald = summary[summary["monitor"] == "wald-peeking"]
    macro("NumWaldCellsBelow", str(int((wald["coverage"] < 0.90).sum())))
    macro("NumWaldCells", str(len(wald)))
    low_row = wald.loc[wald["coverage"].idxmin()]
    macro("NumWaldCovMin", prop3(row_count(low_row), low_row["runs"]))
    high_row = wald.loc[wald["coverage"].idxmax()]
    macro("NumWaldCovMax", prop3(row_count(high_row), high_row["runs"]))
    macro("NumMeanOutsideMax", ceil1(100 * valid["mean_outside"].max()))
    macro("NumTotalRuns", thousands(len(frame) // 5))
    pairs = {"Prpl": ("prpl-hoeffding", "prpl-eb"), "Bet": ("betting-hoeffding", "betting-eb"),
             "Constr": ("prpl-eb", "betting-eb"), "Avsc": ("prpl-hoeffding", "betting-eb"),
             "Price": ("betting-eb", "wald-peeking")}
    for word, (num, den) in pairs.items():
        table = ratio_table(frame, summary, num, den)
        macro(f"Num{word}RatioMin", f"{table['ratio_of_means'].min():.2f}")
        macro(f"Num{word}RatioMax", f"{table['ratio_of_means'].max():.2f}")
        for eps, eps_word in EPS_WORD.items():
            part = table[table["eps"] == eps]
            macro(f"Num{word}RatioMax{eps_word}", f"{part['ratio_of_means'].max():.1f}")
            macro(f"Num{word}SpearmanAbs{eps_word}", f"{abs(spearmanr(part['rho'], part['ratio_of_means']).correlation):.3f}")
    prpl = ratio_table(frame, summary, "prpl-hoeffding", "prpl-eb")
    below = prpl[prpl["ratio_of_means"] < 1.0]
    above = prpl[prpl["ratio_of_means"] > 1.0]
    assert below["rho"].max() < above["rho"].min()
    macro("NumCrossRhoLow", f"{below['rho'].max():.2f}")
    macro("NumCrossRhoHigh", f"{above['rho'].min():.2f}")
    observed = summary.set_index(["graph", "eps", "monitor"])["T_mean"]
    dev_h = [abs(observed[(c.graph, c.eps, "prpl-hoeffding")] / c.pred_h - 1) for c in cells.itertuples()]
    dev_eb = [abs(observed[(c.graph, c.eps, "prpl-eb")] / c.pred_eb - 1) for c in cells.itertuples()]
    macro("NumPredDevMaxH", ceil1(100 * max(dev_h)))
    macro("NumPredDevMaxEB", ceil1(100 * max(dev_eb)))
    ratios_obs = [observed[(c.graph, c.eps, "prpl-hoeffding")] / observed[(c.graph, c.eps, "prpl-eb")] for c in cells.itertuples()]
    ratios_pred = [c.pred_h / c.pred_eb for c in cells.itertuples()]
    macro("NumPredRatioDevMax", ceil1(100 * max(abs(o / p - 1) for o, p in zip(ratios_obs, ratios_pred))))
    big = cells[cells["rho"] > 100]
    macro("NumKappaBig", floor1(min((observed[(c.graph, c.eps, 'prpl-hoeffding')] / observed[(c.graph, c.eps, 'prpl-eb')]) / f_rho(c.rho) for c in big.itertuples())))


def figure_pipeline():
    """Compact AVSC pipeline strip (no data; deterministic)."""
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    plt.rcParams.update({"font.size": 6, "font.family": "serif", "mathtext.fontset": "dejavuserif",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fill, edge, ink, accent = "#f0efec", "#6f6e69", "#1f1f1e", "#2a78d6"
    fig = plt.figure(figsize=(3.33, 1.08))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")
    boxes = [
        (0.5, "Sampler", ["wedge or edge", r"$X_i\in[0,B]$", r"$\mathbb{E}X_i = c\,\theta(G)$"]),
        (26.5, "Conf. seq.", ["betting or", "closed form;", "EB or H. bets"]),
        (52.5, "Certified?", [r"rel. width $\leq\epsilon$,", "checked every", r"$\Delta_t$ samples"]),
        (78.0, "Report", [r"$[L,U]$, midpoint", r"miss prob. $\leq\alpha$", r"rel. err. $\leq\frac{\epsilon}{2-\epsilon}$"]),
    ]
    width, bottom, height = 21.5, 8.0, 20.5
    for x, title, lines in boxes:
        ax.add_patch(FancyBboxPatch((x, bottom), width, height, boxstyle="round,pad=0.25,rounding_size=1.2",
                                    linewidth=0.6, edgecolor=edge, facecolor=fill))
        ax.text(x + width / 2, bottom + height - 3.6, title, ha="center", va="center",
                fontsize=6.4, fontweight="bold", color=ink)
        for i, line in enumerate(lines):
            ax.text(x + width / 2, bottom + height - 9.0 - 4.6 * i, line, ha="center", va="center",
                    fontsize=6.0, color=ink)
    for x0, x1 in ((0.5, 26.5), (26.5, 52.5), (52.5, 78.0)):
        ax.add_patch(FancyArrowPatch((x0 + width + 0.5, bottom + height / 2), (x1 - 0.5, bottom + height / 2),
                                     arrowstyle="-|>", mutation_scale=6, linewidth=0.8, color=accent))
    ax.add_patch(FancyArrowPatch((63.25, bottom - 0.2), (11.25, bottom - 0.2), connectionstyle="arc3,rad=-0.12",
                                 arrowstyle="-|>", mutation_scale=6, linewidth=0.8, color=accent))
    ax.text(37.25, 1.6, r"no: draw $\Delta_t$ more samples", ha="center", va="center", fontsize=6.0, color=ink)
    fig.savefig(os.path.join(PAPER, "fig_pipeline.pdf"), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def figure_rho(frame, summary):
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    blue, orange, gray = "#2a78d6", "#eb6834", "#6f6e69"
    plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "font.family": "serif",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.33, 2.35))
    curve = prediction_curve()
    grid = np.geomspace(1.5, 300.0, 200)
    ax.plot(grid, f_rho(grid), color=gray, lw=1.0, ls="--", label=r"$f(\rho)$, leading order")
    for eps, style in ((0.10, "-"), (0.30, ":")):
        part = curve[curve["eps"] == eps]
        ax.plot(part["rho"], part["ratio"], color=blue, lw=1.0, ls=style,
                label=fr"PrPl prediction, $\epsilon={eps:.2f}$")
    for (num, den, color, marker, name) in (("prpl-hoeffding", "prpl-eb", blue, "o", "PrPl"),
                                            ("betting-hoeffding", "betting-eb", orange, "s", "Bet")):
        table = ratio_table(frame, summary, num, den)
        for eps, filled in ((0.10, True), (0.30, False)):
            part = table[table["eps"] == eps]
            ax.scatter(part["rho"], part["ratio_of_means"], s=16, marker=marker, linewidths=0.8,
                       facecolors=color if filled else "white", edgecolors=color, zorder=3,
                       label=fr"{name} H/EB observed, $\epsilon={eps:.2f}$")
    ax.axhline(1.0, color=gray, lw=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"concentration ratio $\rho = B/\mu$")
    ax.set_ylabel("samples, Hoeffding / EB")
    ax.grid(True, which="major", color="#e6e5e0", lw=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=5.6, frameon=False, loc="upper left", handlelength=2.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(PAPER, "fig_rho.pdf"), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


BASELINES = os.path.join(HERE, "results_cr", "baselines_triangles.csv")


def load_baselines():
    import pandas as pd
    table = pd.read_csv(BASELINES)
    for column in ("covered", "met_target", "feasible"):
        table[column] = table[column].astype(str) == "True"
    return table


def table_and_macros_baselines(summary):
    table = load_baselines()
    budget = table[table["method"] == "fixed-budget-hoeffding"]
    oracle = table[table["method"] == "fixed-n-eb-oracle"]
    assert len(budget) == len(oracle) == len(GRAPHS) * 2 * 80
    feasible = budget[budget["feasible"]]
    macro("NumFBNoBudget", str(int((~budget["feasible"]).sum())))
    macro("NumFBRuns", thousands(len(budget)))
    macro("NumFBCovFeasible", prop3(int(feasible["covered"].sum()), len(feasible)))
    macro("NumFBMetPct", f"{100 * feasible['met_target'].mean():.0f}")
    macro("NumOracleCov", prop3(int(oracle["covered"].sum()), len(oracle)))
    for eps, word in EPS_WORD.items():
        part = oracle[oracle["eps"] == eps]
        macro(f"NumOracleMetPct{word}", f"{100 * part['met_target'].mean():.0f}")
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{lr|rrr|r}", r"\toprule",
             r"& & \multicolumn{3}{c|}{fixed budget (Hoeffding)} & oracle $n$ \\",
             r"graph & $\epsilon$ & no budget & met & mean $n$ & met \\", r"\midrule"]
    for eps in (0.10, 0.30):
        order = sorted(GRAPHS, key=lambda g: lookup(summary, {"graph": g, "eps": eps, "monitor": "prpl-eb"})["rho"])
        for name in order:
            b = budget[(budget["graph"] == name) & (budget["eps"] == eps)]
            o = oracle[(oracle["graph"] == name) & (oracle["eps"] == eps)]
            f = b[b["feasible"]]
            mean_n = human(float(f["n_used"].mean())) if len(f) else "--"
            met = f"{int(f['met_target'].sum())}/{len(f)}" if len(f) else "--"
            cells = [DISPLAY[name], f"{eps:.2f}", f"{int((~b['feasible']).sum())}/80", met, mean_n,
                     f"{int(o['met_target'].sum())}/80"]
            lines.append(" & ".join(cells) + r" \\")
        if eps == 0.10:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Fixed-sample baselines, $\NumRuns$ runs per cell. \emph{Fixed budget}: a "
              r"$100$-sample pilot sets the classical Hoeffding budget for width $\epsilon$; ``no budget'' "
              r"counts runs whose pilot saw no closed wedge, so no budget could be computed; ``met'' counts "
              r"runs, among those with a budget, whose interval reached width $\epsilon$. \emph{Oracle $n$}: "
              r"a fixed-$n$ empirical-Bernstein interval~\cite{maurer2009empirical} at the median "
              r"\bet-EB stopping time of the cell, which no real method knows in advance; ``met'' as before. "
              r"Coverage of both is near $1$ (text).}",
              r"\label{tab:baselines}", r"\end{table}"]
    write("tab_baselines.tex", "\n".join(lines) + "\n")


CLIQUE_GRAPHS = ["jazz", "pgp", "celegans", "power", "com-dblp", "com-amazon"]
CLIQUE_KS = [3, 4, 5, 6]
EXACT_VAR = os.path.join(HERE, "results_cr", "exact_edge_sampler_variance.csv")
CLIQUE_PRED = os.path.join(HERE, "results_cr", "prediction_cliques.csv")


def fmt_ratio(x):
    """Concentration ratios: one decimal below 10^6, scientific above."""
    if x >= 1e6:
        return compact(x)
    return f"{x:,.1f}".replace(",", "{,}")


def clique_complete(summary):
    return len(summary) == len(CLIQUE_GRAPHS) * len(CLIQUE_KS) * 2 * 5 and bool((summary["runs"] == 80).all())


def compact_count(row):
    if row["stopped"] == 0:
        return "cap"
    text = human(row["T_mean"])
    if row["capped"]:
        text += f"$^{{\\mathrm{{c{int(row['capped'])}}}}}$"
    return text


def clique_predictions(summary):
    """Surrogate PrPl-EB stopping times for clique cells, using exact Var(Y) (cached)."""
    import pandas as pd
    from prpl_prediction import predicted_stop
    if os.path.exists(CLIQUE_PRED):
        return pd.read_csv(CLIQUE_PRED)
    exact = pd.read_csv(EXACT_VAR)
    rows = []
    for _, e in exact.iterrows():
        nu = float(e["nu"])
        for eps in (0.10, 0.30):
            pred = predicted_stop(nu, eps, "eb", 400_000_000, sigma2=float(e["var_Y"]))
            rows.append({"graph": e["graph"], "k": int(e["k"]), "eps": eps, "pred_eb": pred,
                         "rho": 1.0 / nu, "rho_eff": float(e["rho_eff"])})
    table = pd.DataFrame(rows)
    table.to_csv(CLIQUE_PRED, index=False)
    return table


def table_and_macros_cliques(frame, summary):
    import pandas as pd
    exact = pd.read_csv(EXACT_VAR).set_index(["graph", "k"])
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lrrr|rrrr|rrrr}", r"\toprule",
             r"& & & & \multicolumn{4}{c|}{$\epsilon=0.10$} & \multicolumn{4}{c}{$\epsilon=0.30$} \\",
             r"graph & $k$ & $\rho$ & $\rho_{\mathrm{eff}}$ & \bet-EB & \prpl-H & cov. & Wald & "
             r"\bet-EB & \prpl-H & cov. & Wald \\", r"\midrule"]
    valid = summary[summary["monitor"] != "wald-peeking"]
    for gi, name in enumerate(CLIQUE_GRAPHS):
        for k in CLIQUE_KS:
            e = exact.loc[(name, k)]
            cells = [DISPLAY[name] if k == CLIQUE_KS[0] else "", str(k), fmt_ratio(1.0 / float(e["nu"])),
                     fmt_ratio(float(e["rho_eff"]))]
            for eps in (0.10, 0.30):
                key = {"graph": name, "k": k, "eps": eps}
                bet = lookup(summary, dict(key, monitor="betting-eb"))
                hoef = lookup(summary, dict(key, monitor="prpl-hoeffding"))
                arms = valid[(valid["graph"] == name) & (valid["k"] == k) & (valid["eps"] == eps)]
                worst = arms.loc[arms["coverage"].idxmin()]
                wald = lookup(summary, dict(key, monitor="wald-peeking"))
                cells += [compact_count(bet), compact_count(hoef), prop3(row_count(worst), worst["runs"]),
                          prop3(row_count(wald), wald["runs"])]
            lines.append(" & ".join(cells) + r" \\")
        if gi < len(CLIQUE_GRAPHS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{$k$-clique study with exact ground truth for every cell ($\NumRuns$ runs per cell; "
              r"sample cap $\NumCapClique$). $\rho = B/\mu$ uses the sampler's range; "
              r"$\rho_{\mathrm{eff}} = 1+\Var(Y)/\E[Y]^2$ is computed exactly from per-edge clique counts and "
              r"is the concentration ratio of a $0/1$ sampler with the same relative variance. "
              r"\bet-EB and \prpl-H: mean samples to certify over runs that stopped (``cap'': no run stopped; "
              r"superscript c: runs that reached the cap). cov.: the lowest coverage among the four "
              r"anytime-valid sequences, over all runs. Wald: coverage of the naive peeking rule.}",
              r"\label{tab:cliques}", r"\end{table*}"]
    write("tab_cliques.tex", "\n".join(lines) + "\n")
    worst_valid = valid.loc[valid["coverage"].idxmin()]
    macro("NumCliqueCovMinValid", prop4(row_count(worst_valid), worst_valid["runs"]))
    wald_all = summary[summary["monitor"] == "wald-peeking"]
    wald_min = wald_all.loc[wald_all["coverage"].idxmin()]
    macro("NumCliqueWaldCovMin", prop3(row_count(wald_min), wald_min["runs"]))
    macro("NumCliqueWaldMinGraph", DISPLAY[wald_min["graph"]])
    macro("NumCliqueWaldMinK", str(int(wald_min["k"])))
    macro("NumCliqueWaldMinEps", f"{wald_min['eps']:.2f}")
    macro("NumCliqueWaldCellsBelow", str(int((wald_all["coverage"] < 0.90).sum())))
    macro("NumCliqueCells", str(len(wald_all)))
    bet = summary[summary["monitor"] == "betting-eb"].set_index(["graph", "k", "eps"])
    hoef = summary[summary["monitor"] == "prpl-hoeffding"].set_index(["graph", "k", "eps"])
    only_eb = [key for key in bet.index if bet.loc[key, "stopped"] == 80 and hoef.loc[key, "stopped"] == 0]
    none = [key for key in bet.index if all(lookup(summary, {"graph": key[0], "k": key[1], "eps": key[2], "monitor": m})["stopped"] == 0 for m in ARMS)]
    macro("NumCliqueOnlyEB", str(len(only_eb)))
    macro("NumCliqueNoneStop", str(len(none)))
    capped_runs = frame[(frame["monitor"] != "wald-peeking") & (frame["outcome"] == "cap")]
    macro("NumCliqueCappedRuns", thousands(len(capped_runs)))
    macro("NumCliqueCappedCov", prop3(int(capped_runs["covered"].sum()), len(capped_runs)) if len(capped_runs) else "--")
    preds = clique_predictions(summary)
    devs = []
    for p in preds.itertuples():
        row = lookup(summary, {"graph": p.graph, "k": p.k, "eps": p.eps, "monitor": "prpl-eb"})
        if row["stopped"] == 80 and p.pred_eb:
            devs.append(abs(row["T_mean"] / p.pred_eb - 1))
    cap_rows = []
    log_term = math.log(2.0 / 0.10)
    for name, k in (("com-amazon", 4), ("com-amazon", 3)):
        e = exact.loc[(name, k)]
        for eps, word in EPS_WORD.items():
            row = lookup(summary, {"graph": name, "k": k, "eps": eps, "monitor": "betting-eb"})
            predicted = 4.0 * log_term * (1.0 / float(e["nu"])) / eps
            cap_rows.append(abs(row["T_mean"] / predicted - 1))
            if k == 4:
                macro(f"NumCapRegimePred{word}", human(predicted))
                macro(f"NumCapRegimeObs{word}", human(row["T_mean"]))
    macro("NumCapRegimeDevMax", ceil1(100 * max(cap_rows)))
    macro("NumAmazonFourRho", fmt_ratio(1.0 / float(exact.loc[("com-amazon", 4)]["nu"])))
    macro("NumAmazonFourRhoEff", fmt_ratio(float(exact.loc[("com-amazon", 4)]["rho_eff"])))
    macro("NumCliquePredCells", str(len(devs)))
    macro("NumCliquePredDevMax", ceil1(100 * max(devs)))


TIMING_RUNS = os.path.join(HERE, "results_cr", "timing_runs.csv")
TIMING_BUILD = os.path.join(HERE, "results_cr", "timing_build.csv")
TIMED = ["com-dblp", "com-amazon", "roadnet-ca", "com-youtube", "as-skitter"]


def seconds(x):
    if x < 1:
        return f"{x:.2f}"
    if x < 10:
        return f"{x:.1f}"
    return f"{x:.0f}"


def table_and_macros_wallclock(summary):
    import pandas as pd
    runs = pd.read_csv(TIMING_RUNS)
    build = pd.read_csv(TIMING_BUILD).set_index("graph")
    assert (runs["outcome"] == "stopped").all() and len(runs) == len(TIMED) * 2 * 10 * 5
    runs["work"] = runs["sampling_s_at_T"] + runs["update_s"]
    med = runs.groupby(["graph", "eps", "monitor"]).agg(work=("work", "median"),
                                                       update=("update_s", "median")).reset_index()
    med["total"] = med["work"] + med["graph"].map(build["build_s_median"])
    total = med.set_index(["graph", "eps", "monitor"])["total"]
    update = med.set_index(["graph", "eps", "monitor"])["update"]
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lr|r|rrrr|r}", r"\toprule",
             r"graph & $\epsilon$ & build & \bet-EB & \prpl-EB & \bet-H & \prpl-H & $\frac{\text{\prpl-H}}{\text{\bet-EB}}$ \\",
             r"\midrule"]
    ratios = {}
    for eps in (0.10, 0.30):
        order = sorted(TIMED, key=lambda g: lookup(summary, {"graph": g, "eps": eps, "monitor": "prpl-eb"})["rho"])
        for name in order:
            cells = [DISPLAY[name], f"{eps:.2f}", seconds(build.loc[name, "build_s_median"])]
            for monitor in ("betting-eb", "prpl-eb", "betting-hoeffding", "prpl-hoeffding"):
                cells.append(seconds(total[(name, eps, monitor)]))
            ratio = total[(name, eps, "prpl-hoeffding")] / total[(name, eps, "betting-eb")]
            ratios[(name, eps)] = ratio
            cells.append(f"{ratio:.2f}$\\times$")
            lines.append(" & ".join(cells) + r" \\")
        if eps == 0.10:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Wall-clock seconds to certify width $\epsilon$, \emph{including} building the "
              r"sampler (median of $10$ runs per cell, one process, no other experiment running). "
              r"\emph{build}: degrees, wedge prefix sums and sorted edge keys (median of $5$ builds), "
              r"included in every total. Last column: total-time ratio of the closed-form Hoeffding "
              r"sequence to AVSC's default.}",
              r"\label{tab:wallclock}", r"\end{table}"]
    write("tab_wallclock.tex", "\n".join(lines) + "\n")
    macro("NumWallRatioMax", f"{max(ratios.values()):.1f}")
    small = [ratios[(g, e)] for g in ("com-dblp", "com-amazon") for e in (0.10, 0.30)]
    macro("NumWallRatioSmallMin", f"{min(small):.2f}")
    macro("NumWallRatioSmallMax", f"{max(small):.2f}")
    macro("NumBuildMax", seconds(build["build_s_median"].max()))
    macro("NumBuildMin", seconds(build["build_s_median"].min()))
    macro("NumBetUpdateMax", seconds(update[[k for k in update.index if k[2] == "betting-eb"]].max()))
    prpl_vs_bet = [total[(g, e, "prpl-eb")] / total[(g, e, "betting-eb")] for g in TIMED for e in (0.10, 0.30)]
    macro("NumWallPrplOverBetMin", f"{min(prpl_vs_bet):.2f}")
    macro("NumWallPrplOverBetMax", f"{max(prpl_vs_bet):.2f}")
    sk = ("as-skitter", 0.10)
    macro("NumSkitterBetEBSeconds", seconds(total[sk + ("betting-eb",)]))
    macro("NumSkitterPrplHSeconds", seconds(total[sk + ("prpl-hoeffding",)]))


CLIQUE_COST = os.path.join(HERE, "results_cr", "clique_cost.csv")


def prep_text(x):
    """Preprocessing time: milliseconds below 10 ms, else seconds."""
    if x < 0.01:
        return f"{x * 1e3:.0f}\\,ms"
    return f"{seconds(x)}\\,s"


def table_and_macros_cost():
    import pandas as pd
    cost = pd.read_csv(CLIQUE_COST)
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lr|rr|rr|rr}", r"\toprule",
             r"& & & & \multicolumn{2}{c|}{range inflation} & \multicolumn{2}{c}{sample inflation} \\",
             r"graph & $k$ & prep. & per sample & $B_{\min}$ & $B_{\Delta}$ & $B_{\min}$ & $B_{\Delta}$ \\",
             r"\midrule"]
    sample_infl = {}
    for i, row in cost.iterrows():
        exact, mindeg, maxdeg = float(row["B_exact"]), float(row["B_mindeg"]), float(row["B_maxdeg"])
        per_sample = float(row["per_sample_s"])
        per_text = f"{per_sample * 1e9:.0f}\\,ns" if per_sample < 1e-6 else f"{per_sample * 1e6:.1f}\\,$\\mu$s"
        cells = [DISPLAY[row["graph"]], str(int(row["k"])), prep_text(float(row["preprocessing_s_median"])), per_text,
                 f"{mindeg / exact:.2f}$\\times$", f"{maxdeg / exact:.2f}$\\times$"]
        for label in ("mindeg", "maxdeg"):
            if int(row["stopped_exact"]) == int(row["runs_exact"]) and int(row[f"stopped_{label}"]) == int(row[f"runs_{label}"]):
                ratio = float(row[f"T_mean_{label}"]) / float(row["T_mean_exact"])
                sample_infl[(row["graph"], int(row["k"]), label)] = ratio
                cells.append(f"{ratio:.2f}$\\times$")
            else:
                cells.append("cap")
        lines.append(" & ".join(cells) + r" \\")
        if i in (2, 5):
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Cost of the $k$-clique edge sampler. \emph{prep.}: one-off preprocessing that "
              r"lists $\CN(e)$ for every edge and computes the exact range $B$ (median of $3$ builds); "
              r"\emph{per sample}: drawing time. Range inflation of two bounds that avoid the per-edge "
              r"pass: $B_{\min}=M\binom{\max_e\min(\deg u,\deg v)-1}{k-2}$ ($O(M)$) and "
              r"$B_{\Delta}=M\binom{\Delta}{k-2}$ ($O(n)$). Sample inflation: mean \bet-EB samples to "
              r"certify $\epsilon=0.30$ with that bound over the exact one, on the same streams ($80$ runs; "
              r"``cap'': no run certified within the cap, recorded after $5$ runs by a rule fixed in advance).}",
              r"\label{tab:cost}", r"\end{table}"]
    write("tab_cost.tex", "\n".join(lines) + "\n")
    macro("NumPrepMax", prep_text(cost["preprocessing_s_median"].max()))
    macro("NumPrepMin", prep_text(cost["preprocessing_s_median"].min()))
    macro("NumMaxdegInflMax", f"{max(v for (g, k, l), v in sample_infl.items() if l == 'maxdeg'):.1f}")
    macro("NumMindegInflMax", f"{max(v for (g, k, l), v in sample_infl.items() if l == 'mindeg'):.2f}")
    a4 = cost[(cost["graph"] == "com-amazon") & (cost["k"] == 4)].iloc[0]
    macro("NumAmazonFourMindegRange", f"{float(a4['B_mindeg']) / float(a4['B_exact']):.2f}")
    macro("NumAmazonFourMaxdegRange", f"{float(a4['B_maxdeg']) / float(a4['B_exact']):.2f}")
    macro("NumAmazonFourMindegSample", f"{sample_infl[('com-amazon', 4, 'mindeg')]:.2f}")
    macro("NumAmazonFourMaxdegSample", f"{sample_infl[('com-amazon', 4, 'maxdeg')]:.2f}")


def write(name, text):
    os.makedirs(GEN, exist_ok=True)
    with open(os.path.join(GEN, name), "w") as f:
        f.write(text)


def write_macros():
    lines = ["% Generated by code/make_paper_artifacts.py -- do not edit by hand."]
    for name in sorted(MACROS):
        lines.append(f"\\newcommand{{\\{name}}}{{{MACROS[name]}}}")
    write("numbers.tex", "\n".join(lines) + "\n")


def main():
    gt, gt_times = ground_truth()
    stats = graph_stats()
    write("tab_datasets.tex", table_datasets(stats, gt))
    macro("NumGTMaxSeconds", f"{max(gt_times.values()):.1f}")
    macro("NumMaxEdges", f"{stats['as-skitter']['m'] / 1e6:.1f}")
    import run_cr_grid
    import run_cr_cliques
    assert run_cr_grid.RUNS == run_cr_cliques.RUNS
    macro("NumRuns", str(run_cr_grid.RUNS))
    macro("NumCapTri", power_of_ten(run_cr_grid.SAMPLE_CAP))
    macro("NumCapClique", power_of_ten(run_cr_cliques.SAMPLE_CAP))
    import analysis_cr
    frame, summary = analysis_cr.triangle_summary()
    if len(summary) and triangle_complete(summary):
        write("tab_samples.tex", table_triangle_samples(frame, summary))
        write("tab_coverage.tex", table_triangle_coverage(summary))
        cells = prediction_cells(summary)
        triangle_macros(frame, summary, cells)
        limit_macros(summary)
        figure_rho(frame, summary)
        figure_pipeline()
        if os.path.exists(BASELINES):
            table_and_macros_baselines(summary)
    else:
        print("triangle grid incomplete: tables not written")
        if "--preview" in sys.argv and len(summary):
            print(summary.to_string())
    if len(summary) and triangle_complete(summary) and os.path.exists(TIMING_RUNS):
        table_and_macros_wallclock(summary)
    if os.path.exists(CLIQUE_COST):
        table_and_macros_cost()
    clique_frame, clique_sum = analysis_cr.clique_summary()
    if len(clique_sum) and clique_complete(clique_sum):
        table_and_macros_cliques(clique_frame, clique_sum)
        misses = clique_frame[(clique_frame["monitor"] == "wald-peeking") & (~clique_frame["covered"])]
        rare = misses[misses["graph"].isin(["celegans", "com-amazon"]) & (misses["k"] >= 5)]
        below = rare["U_y"].astype(float) < rare["nu"].astype(float)
        macro("NumWaldRareMisses", str(len(rare)))
        macro("NumWaldRareBelow", str(int(below.sum())))
        if len(summary) and triangle_complete(summary):
            for eps, word in EPS_WORD.items():
                wedge = lookup(summary, {"graph": "com-amazon", "eps": eps, "monitor": "betting-eb"})
                edge = lookup(clique_sum, {"graph": "com-amazon", "k": 3, "eps": eps, "monitor": "betting-eb"})
                assert wedge["stopped"] == 80 and edge["stopped"] == 80
                macro(f"NumAmazonWedgeT{word}", human(wedge["T_mean"]))
                macro(f"NumAmazonEdgeT{word}", human(edge["T_mean"]))
                macro(f"NumAmazonWedgeCov{word}", prop3(row_count(wedge), wedge["runs"]))
                macro(f"NumAmazonEdgeCov{word}", prop3(row_count(edge), edge["runs"]))
    else:
        print("clique grid incomplete: clique table not written")
    write_macros()
    print("wrote", sorted(os.listdir(GEN)))


main()
