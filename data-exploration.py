import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

ptc = pd.read_csv("PTcourse_data.csv")
pts = pd.read_csv("PTstaff_data.csv")
umc = pd.read_csv("UMDIOcourse_data.csv")

'''
Section 1: Data exploration on raw data
'''

print(ptc.columns)
print(pts.columns)
print(umc.columns)

print(ptc.head(5))
print(pts.head(5))
print(umc.head(5))

pts["round_avg"] = pts.average_rating.apply(lambda x: round(x, 2))
counts = pts.round_avg.value_counts().sort_index()

fig, axs = plt.subplots(1, 2)

pts.round_avg.plot.hist(bins=20)
axs[0].set_xlabel("Avg rating")
axs[0].set_ylabel("N professors")
axs[0].set_title("Avg rating vs professors")

axs[0].scatter(ptc["credits"], ptc["average_gpa"], alpha=0.5)
axs[0].set_xlabel("Credits")
axs[0].set_ylabel("Avg GPA")
axs[0].set_title("Credits vs Avg GPA")

print(stats.pearsonr(ptc["credits"], ptc["average_gpa"]).pvalue)
# Test correlation between credits and avg gpa
df = ptc[["credits", "average_gpa"]].dropna()
stats.pearsonr(df["credits"], df["average_gpa"])

plt.show()
