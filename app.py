import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif


# 1️⃣ Load and Inspect Data
df  = pd.read_csv("./student_performance.csv")
df.info()
print(df.head())
print(df.describe())
print(df["weekly_self_study_hours"].head())
print(df["attendance_percentage"].head())
print(df["class_participation"].head())
print(df["total_score"].head())


# 2️⃣ Data Cleaning
print(df.isnull().sum())
print(df.dtypes)


# 3️⃣ Feature Engineering
df["grade"] = df["grade"].map({"A": 1, "B": 2, "C": 3, "D": 4, "F": 5}).astype(int)
study_per_day = df["weekly_self_study_hours"] / 7
df["study_per_day"] = study_per_day
print(df.head())



# 4️⃣ Exploratory Data Analysis (EDA)
fig, axs = plt.subplots(3,2, figsize = (10,10))
sns.boxplot(data=df, y="weekly_self_study_hours", ax=axs[0,0], color="lightcoral")
sns.boxplot(data=df, y="study_per_day", ax=axs[0,1], color="moccasin")
sns.boxplot(data=df, y="attendance_percentage", ax=axs[1,0], color="gold")
sns.boxplot(data=df, y="class_participation", ax=axs[1,1], color="lightgreen")
sns.boxplot(data=df, y="total_score", ax=axs[2,0], color="lightblue")
sns.countplot(data=df, x="grade", ax=axs[2,1], color="plum")
plt.tight_layout()
#plt.show()

numeric_cols = df.drop(columns = ["student_id"])
corr = numeric_cols.corr()
print(corr)
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.suptitle("Correlation between each numeric feature and grade")
#plt.show()

df_sample = df.sample(n=100, random_state=42)  # 10k rows instead of 1M
X_sample = df_sample[['weekly_self_study_hours', 'study_per_day', 'attendance_percentage', 'class_participation', 'total_score']]
y_sample = df_sample['grade']
mi_scores = mutual_info_classif(X_sample, y_sample, discrete_features=False, random_state=42)
print(f"miutual information of random 1000 samples : {mi_scores}")

fig, axs = plt.subplots(2,2, figsize = (10,10))
sns.scatterplot(data=df_sample, x="weekly_self_study_hours",y="total_score",  ax=axs[0,0], hue="grade")
sns.scatterplot(data=df_sample, x="study_per_day",y="total_score",  ax=axs[0,1], hue="grade")
sns.scatterplot(data=df_sample, x="attendance_percentage", y="total_score", ax=axs[1,0], hue="grade")
sns.scatterplot(data=df_sample, x="class_participation",y="total_score",  ax=axs[1,1], hue="grade")
plt.tight_layout()
plt.show()

