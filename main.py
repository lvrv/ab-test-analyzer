import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(42)

n = 8000
p_A, p_B = 0.10, 0.115
conv_A = np.random.binomial(1, p_A, size=n)
conv_B = np.random.binomial(1, p_B, size=n)
rev_A = np.where(conv_A == 1, np.random.lognormal(2.5, 0.6, n), 0.0)
rev_B = np.where(conv_B == 1, np.random.lognormal(2.55, 0.6, n), 0.0)

df = pd.DataFrame({
    "group": ["A"] * n + ["B"] * n,
    "converted": np.concatenate([conv_A, conv_B]),
    "revenue": np.concatenate([rev_A, rev_B]),
})

summary = df.groupby("group").agg(
    users=("converted", "size"),
    conversions=("converted", "sum"),
    conversion_rate=("converted", "mean"),
    revenue_sum=("revenue", "sum"),
    arpu=("revenue", "mean"),
    aov=("revenue", lambda s: s[s > 0].mean() if (s > 0).any() else 0),
).reset_index()

cont = pd.crosstab(df["group"], df["converted"])
chi2, p_chi2, _, _ = stats.chi2_contingency(cont)

revA = df.loc[df.group == "A", "revenue"]
revB = df.loc[df.group == "B", "revenue"]
t_stat, p_ttest = stats.ttest_ind(revA, revB, equal_var=False)

sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))
sns.barplot(x="group", y="conversion_rate", data=summary)
plt.title("Conversion Rate by Group")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "conversion_rate.png", dpi=140)
plt.close()

plt.figure(figsize=(6, 4))
sns.barplot(x="group", y="arpu", data=summary)
plt.title("ARPU by Group")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "arpu.png", dpi=140)
plt.close()

report = {
    "summary": summary.to_dict(orient="records"),
    "tests": {
        "conversion_chi2": {"chi2": float(chi2), "p_value": float(p_chi2)},
        "arpu_ttest": {"t_stat": float(t_stat), "p_value": float(p_ttest)},
    },
}
with open(OUTPUT_DIR / "report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("✅ A/B анализ выполнен. Артефакты в:", OUTPUT_DIR)