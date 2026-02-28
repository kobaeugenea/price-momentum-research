"""
Plot cumulative returns for all 10 equal-weighted momentum deciles.

Data source : Fama-French "10 Portfolios Formed on Prior Return (12-2)"
              https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

Decile 1 (Lo PRIOR) = weakest past return → red
Decile 10 (Hi PRIOR) = strongest past return → green
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils import parse_ff_monthly_section, compute_stats, stats_label


FF_FILE = "10_Portfolios_Prior_12_2.csv"
START   = "1990"
END     = "2025"

ff = parse_ff_monthly_section(FF_FILE, "Average Equal Weighted Returns -- Monthly")
ff = ff.loc[START:END]

decile_cols = ff.columns.tolist()          # Lo PRIOR … Hi PRIOR
colors      = cm.RdYlGn(np.linspace(0, 1, len(decile_cols)))

fig, ax = plt.subplots(figsize=(14, 8))

for col, color in zip(decile_cols, colors):
    ret = ff[col].dropna()
    cum = (1 + ret).cumprod()
    cum /= cum.iloc[0]
    s = compute_stats(ret)
    ax.plot(cum.index, cum.values, color=color, linewidth=1.5,
            label=stats_label(col, s))

ax.set_title(
    "Momentum Deciles — Equal-Weighted Returns\n"
    "10 Portfolios Formed on Prior 12-2 (Fama-French)"
)
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Return (normalized to 1.0)")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
