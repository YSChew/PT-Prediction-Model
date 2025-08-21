import pandas as pd
from scipy import stats

review_df = pd.read_csv("review_data.csv")
# Quick analysis on data to find relevant features
avg_exp_grade = review_df.groupby("professor_id")["expected_grade"].mean()
avg_courseGPA = review_df.groupby("professor_id")["course_avgGPA"].mean()
avg_courselvl = review_df.groupby("professor_id")["course_lvl"].mean()
avg_credits = review_df.groupby("professor_id")["credits"].mean()
pivot1 = pd.crosstab(review_df["review_score"], review_df["expected_grade"])
print(stats.chi2_contingency(pivot1).pvalue)
pivot2 = pd.crosstab(review_df["review_score"], review_df["course_avgGPA"])
print(stats.chi2_contingency(pivot2).pvalue)
pivot3 = pd.crosstab(review_df["review_score"], review_df["school"])
print(stats.chi2_contingency(pivot3).pvalue)
pivot4 = pd.crosstab(review_df["review_score"], review_df["credits"])
print(stats.chi2_contingency(pivot4).pvalue)
pivot5 = pd.crosstab(review_df["review_score"], review_df["expected_grade"] - review_df["course_avgGPA"])
print(stats.chi2_contingency(pivot5).pvalue)