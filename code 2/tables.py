"""Write the LaTeX tables of Section VI-C and Appendix B-B from results/."""
import numpy as np
import pandas as pd

import common as C

OUT = C.RESULTS / "tables"
NAMES = {"wine_red": "Wine (red)", "cahousing": "Calif.\\ housing"}
SHORT = {"wine_red": "Wine", "cahousing": "Calif."}
NOISE = {"output": "Out.", "objective": "Obj."}
ATTACK = {"full": "Full", "restricted": "Restr."}


def header(caption, label, spec, columns, colsep="4pt"):
    return [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
            r"\footnotesize", rf"\setlength{{\tabcolsep}}{{{colsep}}}", rf"\begin{{tabular}}{{{spec}}}",
            r"\toprule", columns + r" \\", r"\midrule"]


def footer(lines):
    lines[-1] = r"\bottomrule"
    return "\n".join(lines + [r"\end{tabular}", r"\end{table}"]) + "\n"


def certificate_table(bounds, threat, caption, label):
    b = bounds[bounds.threat == threat].copy()
    b["GDP"] = C.tv_gaussian(b.U0)
    b["TV"] = C.tv_gaussian(b.U3)
    b["ratio"] = b.U3 / b.U0
    g = b.groupby(["dataset", "mechanism", "k"]).agg(
        GDP=("GDP", "mean"), TV=("TV", "mean"), ratio=("ratio", "mean"),
        geo=("geometry_share", "mean"), valid=("geometry_share", "count"), LU=("L_over_U3", "min"))
    lines = header(caption, label, "lcccccc",
                   r"Noise & $k$ & GDP & TV & $U_3/U_0$ & Geom. & $\min L/U_3$")
    for name in C.DATASETS:
        lines += [rf"\multicolumn{{7}}{{c}}{{\textbf{{{NAMES[name]}}}}} \\", r"\midrule"]
        for mechanism in C.MECHANISMS:
            for k in sorted(b.k.unique()):
                r = g.loc[(name, mechanism, k)]
                if r.valid == 0:
                    geo = "--"
                else:
                    geo = f"{100 * r.geo:.1f}\\%" + ("$^\\dagger$" if r.valid < len(C.SEEDS) else "")
                first = NOISE[mechanism] if k == 1 else "    "
                lines.append(f"{first} & {k} & {r.GDP:.3f} & {r.TV:.3f} & {r.ratio:.3f} & {geo} & {r.LU:.3f} \\\\")
            lines.append(r"\addlinespace" if mechanism == "output" else r"\midrule")
    return footer(lines)


def downstream_table(d, pairs, caption, label, colsep="3pt"):
    g = d.groupby(["dataset", "threat", "mechanism", "k"]).mean(numeric_only=True) * 100
    lines = header(caption, label, "lllcccc",
                   r"Design & Attack & Noise & $k=1$ & $k=2$ & $k=4$ & $k=8$", colsep)
    for name in C.DATASETS:
        for threat in C.THREATS:
            for mechanism in C.MECHANISMS:
                cells = ["/".join(f"{g.loc[(name, threat, mechanism, k), col]:.0f}" for col in pairs)
                         for k in (1, 2, 4, 8)]
                design = SHORT[name] if threat == "full" and mechanism == "output" else ""
                attack = ATTACK[threat] if mechanism == "output" else ""
                lines.append(f"{design} & {attack} & {NOISE[mechanism]} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    return footer(lines)


def utility_table(u):
    g = u.groupby(["dataset", "mechanism", "mu"]).mean(numeric_only=True).reset_index()
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Test MSE on transformed labels, means over five splits. "
             r"Private MSE is the exact noise expectation.}",
             r"\label{tab:utility}", r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}",
             r"\begin{tabular}{llcccccc}", r"\toprule",
             r" & & & & \multicolumn{4}{c}{Private, $\mu=$} \\", r"\cmidrule(lr){5-8}",
             r"Design & Noise & Nonpr. & Const. & 0.25 & 0.5 & 1 & 2 \\", r"\midrule"]
    fmt = lambda v: f"{v:.4f}".lstrip("0")
    for name in C.DATASETS:
        for mechanism in C.MECHANISMS:
            s = g[(g.dataset == name) & (g.mechanism == mechanism)].sort_values("mu")
            cells = [fmt(s.clean_mse.iloc[0]), fmt(s.constant_mse.iloc[0])] + [fmt(v) for v in s.private_mse]
            first = SHORT[name] if mechanism == "output" else ""
            lines.append(f"{first} & {NOISE[mechanism]} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bounds = pd.read_csv(C.RESULTS / "bounds.csv")
    down = pd.read_csv(C.RESULTS / "downstream.csv")
    util = pd.read_csv(C.RESULTS / "utility.csv")
    acc = down.groupby("dataset").accuracy.mean() * 100
    tables = {
        "real_design_full.tex": certificate_table(
            bounds, "full",
            r"Real-design certificates at $\mu=1$ under full-domain label replacements. GDP: group "
            r"baseline; TV: $2\Phi(U_3/2)-1$; $U_3/U_0$: certified displacement relative to group "
            r"privacy; Geom.: geometry share. Means over five splits; $\min L/U_3$ is the minimum.",
            "tab:real-design"),
        "real_design_restricted.tex": certificate_table(
            bounds, "restricted",
            r"Restricted replacements $|y_i'-y_i|\le0.1$ at $\mu=1$. GDP: group baseline under the "
            r"bounded-change adjacency, $2\Phi(0.1k\mu/2)-1$; $U_3/U_0$ is relative to $0.1k\mu$. "
            r"Geom.\ averages over splits with a positive improvement and is omitted where there is "
            r"none; $^\dagger$four of five splits. Means over five splits; $\min L/U_3$ is the minimum.",
            "tab:restricted"),
        "downstream.tex": downstream_table(
            down, ("certified_U0", "certified_U3"),
            r"Percentage of test points whose majority above-threshold decision is certified against "
            r"every admissible $k$-label attack at $\mu=1$ (group privacy / ours), using the direct "
            rf"Gaussian tradeoff conversion. Majority-decision accuracy is ${acc['wine_red']:.1f}\%$ on "
            rf"Wine and ${acc['cahousing']:.1f}\%$ on California housing. Means over five splits.",
            "tab:downstream"),
        "downstream_correct.tex": downstream_table(
            down, ("certified_correct_U0", "certified_correct_U3"),
            r"Percentage of test points that are both correctly classified by the majority decision "
            r"and certified (group privacy / ours), direct tradeoff conversion, $\mu=1$.",
            "tab:downstream-correct"),
        "downstream_tv.tex": downstream_table(
            down, ("tv_certified_U0", "tv_certified_U3"),
            r"Ablation: certified percentage using the TV conversion of "
            r"Corollary~\ref{cor:operational-poisoning}, $|p(z)-1/2|>\tau_k$ (group privacy / ours), "
            r"$\mu=1$.",
            "tab:downstream-tv"),
        "downstream_u2.tex": downstream_table(
            down, ("certified_U0", "certified_U2", "certified_U3"),
            r"Downstream ablation: certified percentage with the displacement cap $U_0$ (group "
            r"privacy), $U_2$ (heterogeneous magnitudes and label reach), and $U_3$ (with geometry), "
            r"shown as $U_0/U_2/U_3$; direct tradeoff conversion, $\mu=1$, means over five splits.",
            "tab:downstream-u2", colsep="2.5pt"),
        "utility.tex": utility_table(util),
    }
    for fname, text in tables.items():
        (OUT / fname).write_text(text)
    print("minimum L/U3:", round(bounds.L_over_U3.min(), 4))
    print("tables written to", OUT)


if __name__ == "__main__":
    main()
