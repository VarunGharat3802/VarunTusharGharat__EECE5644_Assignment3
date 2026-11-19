"""Generate LaTeX tables from the experiment JSON outputs."""
import json
import os
import numpy as np

os.makedirs("tables", exist_ok=True)

# ---------------- Q1 ----------------
r1 = json.load(open("q1_results.json"))
P_grid = r1["P_grid"]
runs = r1["runs"]
Ns = sorted(runs, key=int)

lines = [r"\begin{tabular}{rccc}", r"\toprule",
         r"Training samples $N$ & Selected $P$ & Final training NLL & Test $\hat P(\text{error})$ \\",
         r"\midrule"]
for n in Ns:
    d = runs[n]
    lines.append(f"{int(n):,} & {d['best_P']} & {d['train_nll']:.4f} & {d['test_p_err']:.4f} \\\\")
lines += [r"\midrule",
          f"\\multicolumn{{3}}{{l}}{{Theoretically optimal (true-pdf MAP)}} & {r1['p_err_opt']:.4f} \\\\",
          r"\bottomrule", r"\end{tabular}"]
open("tables/q1_results.tex", "w").write("\n".join(lines))

head = " & ".join(str(p) for p in P_grid)
lines = [r"\begin{tabular}{r" + "c" * len(P_grid) + "}", r"\toprule",
         r"$N$ \textbackslash{} $P$ & " + head + r" \\", r"\midrule"]
for n in Ns:
    cv = runs[n]["cv_err"]
    best = int(np.argmin(cv))
    cells = [(f"\\textbf{{{v:.3f}}}" if i == best else f"{v:.3f}") for i, v in enumerate(cv)]
    lines.append(f"{int(n):,} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
open("tables/q1_cv.tex", "w").write("\n".join(lines))
open("tables/q1_opt.tex", "w").write(f"{r1['p_err_opt']:.4f}%")
m, e = f"{r1['grad_check']:.1e}".split("e")
open("tables/q1_grad.tex", "w").write(f"${m}\\times10^{{{int(e)}}}$%")

# ---------------- Q2 ----------------
if os.path.exists("q2_results.json"):
    r2 = json.load(open("q2_results.json"))
    open("tables/q2_d12.tex", "w").write(f"{r2['d12']:.2f}%")
    open("tables/q2_eig.tex", "w").write(f"{r2['avg_eig_sum']:.2f}%")
    orders, R = r2["orders"], r2["R"]
    lines = [r"\begin{tabular}{r" + "c" * len(orders) + "cc}", r"\toprule",
             r"$N$ \textbackslash{} order & " + " & ".join(map(str, orders))
             + r" & Mean & Mode \\", r"\midrule"]
    for n, cnt in r2["counts"].items():
        sel = np.array(r2["selections"][n])
        mode = int(np.bincount(sel).argmax())
        cells = []
        for M, c in zip(orders, cnt):
            pct = 100.0 * c / R
            s = f"{pct:.0f}\\%" if c else "--"
            cells.append(f"\\textbf{{{s}}}" if M == mode else s)
        lines.append(f"{int(n):,} & " + " & ".join(cells)
                     + f" & {sel.mean():.2f} & {mode} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open("tables/q2_rates.tex", "w").write("\n".join(lines))
print("tables written")
