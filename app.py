import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df  = pd.read_csv("./student_performance.csv")
df.info()
print(df.head())
print(df.describe())
print(df["weekly_self_study_hours"].head())
print(df["attendance_percentage"].head())
print(df["class_participation"].head())
print(df["total_score"].head())


fig, axs = plt.subplots(3,2, figsize = (10,10))
sns.boxplot(data=df, y = "weekly_self_study_hours", ax= axs[0,0], color= "lightcoral")
sns.boxplot(data=df, y = "attendance_percentage", ax= axs[0,1], color = "gold")
sns.boxplot(data=df, y = "class_participation", ax= axs[1,0], color = "lightgreen")
sns.boxplot(data=df, y = "total_score", ax= axs[1,1], color = "lightblue")
sns.countplot(data=df, x = "grade", ax= axs[2,0], color = "plum")
fig.delaxes(axs[2, 1])
plt.tight_layout()
plt.show()


df["grade"] = df["grade"].map({"A": 1, "B": 2, "C": 3, "D": 4, "F": 5}).astype(int)


numeric_cols = df.drop(columns = ["student_id"])
corr = numeric_cols.corr()
print(corr)
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.suptitle("Correlation between each numeric feature and grade")
plt.show()


