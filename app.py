import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



# 1️⃣ 
df  = pd.read_csv("./student_performance.csv")
df.info()
print(df.head())
print(df.describe())
print(df["weekly_self_study_hours"].head())
print(df["attendance_percentage"].head())
print(df["class_participation"].head())
print(df["total_score"].head())


# 2️⃣ 
print(df.isnull().sum())
print(df.dtypes)


# 3️⃣ 
df["grade"] = df["grade"].map({"A": 1, "B": 2, "C": 3, "D": 4, "F": 5}).astype(int)
study_per_day = df["weekly_self_study_hours"] / 7
df["study_per_day"] = study_per_day
print(df.head())

df["active_students"] = (
    (df["weekly_self_study_hours"] >= 15) &
    (df["class_participation"] >= 7)
)
print(df.head())
df["active_students"] = df["active_students"].astype(int)


# 4️⃣ 
fig, axs = plt.subplots(3,2, figsize = (10,10))
sns.boxplot(data=df, y="weekly_self_study_hours", ax=axs[0,0], color="lightcoral")
sns.boxplot(data=df, y="study_per_day", ax=axs[0,1], color="moccasin")
sns.boxplot(data=df, y="attendance_percentage", ax=axs[1,0], color="gold")
sns.boxplot(data=df, y="class_participation", ax=axs[1,1], color="lightgreen")
sns.boxplot(data=df, y="total_score", ax=axs[2,0], color="lightblue")
sns.countplot(data=df, x="grade", ax=axs[2,1], color="plum")
plt.tight_layout()
plt.show()

numeric_cols = df.drop(columns = ["student_id"])
corr = numeric_cols.corr()
print(corr)
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.suptitle("Correlation between each numeric feature and grade")
plt.show()

df_sample = numeric_cols.sample(n=100, random_state=42)  # 10k rows instead of 1M
X_sample = df_sample[['weekly_self_study_hours', 'study_per_day', 'attendance_percentage', 'class_participation', 'total_score', 'active_students']]
y_sample = df_sample['grade']
mi_scores = mutual_info_classif(X_sample, y_sample, discrete_features=False, random_state=42)
print(f"miutual information of random 1000 samples : {mi_scores}")

fig, axs = plt.subplots(2,2, figsize = (10,10))
sns.scatterplot(data=df_sample, x="weekly_self_study_hours",y="total_score",  ax=axs[0,0], hue="grade")
sns.scatterplot(data=df_sample, x="study_per_day",y="total_score",  ax=axs[0,1], hue="grade")
sns.scatterplot(data=df_sample, x="attendance_percentage", y="total_score", ax=axs[1,0], hue="grade")
sns.scatterplot(data=df_sample, x="class_participation",y="total_score",  ax=axs[1,1], hue="grade")
plt.tight_layout()
#plt.show()


# 6️.1 
print("features without attendance_percentage and class_participation")
df1 = df_sample.drop(columns = ["attendance_percentage", "class_participation"])
X = df1.drop(columns = ["grade"])
y = df1["grade"]
X_train,Y_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(Y_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"RandomForestClassifier Accuracy after feature selection: {accuracy}")
model = AdaBoostClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)     
y_pred = model.predict(Y_test)
accuracy1 = accuracy_score(y_test, y_pred)
print(f"AdaBoostClassifier Accuracy after feature selection: {accuracy1}")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(Y_test)
accuracy2 = accuracy_score(y_test, y_pred)
print(f"LogisticRegression Accuracy after feature selection: {accuracy2}")
compare_result = pd.DataFrame({
    "model":["RandomForestClassifier","AdaBoostClassifier","LogisticRegression"],
    "accuracy":[accuracy,accuracy1,accuracy2]
})


# 6️.2
print("all features")
df2 = df_sample
X = df2.drop(columns = ["grade"])
y = df2["grade"]
X_train,Y_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(Y_test)
accuracy3 = accuracy_score(y_test, y_pred)
print(f"RandomForestClassifier Accuracy after feature selection: {accuracy3}")
model = AdaBoostClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)     
y_pred = model.predict(Y_test)
accuracy4 = accuracy_score(y_test, y_pred)
print(f"XGBoostClassifier Accuracy after feature selection: {accuracy4}")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(Y_test)
accuracy5 = accuracy_score(y_test, y_pred)
print(f"LogisticRegression Accuracy after feature selection: {accuracy5}")
compare_result1 = pd.DataFrame({
    "model":["RandomForestClassifier","AdaBoostClassifier","LogisticRegression"],
    "accuracy":[accuracy3,accuracy4,accuracy5]
})


# 6️.3
fig, axes = plt.subplots(1, 2, figsize=(12,6)) 
sns.barplot(data=compare_result, x="model", y="accuracy", hue="model", ax=axes[0])
axes[0].set_title("Accuracy after Feature Selection")
for p in axes[0].patches:
    height = p.get_height()
    axes[0].annotate(f'{height:.2f}', 
                     (p.get_x() + p.get_width() / 2., height),
                     ha='center', va='bottom')
sns.barplot(data=compare_result1, x="model", y="accuracy", hue="model", ax=axes[1])
axes[1].set_title("Accuracy before Feature Selection")
for p in axes[1].patches:
    height = p.get_height()
    axes[1].annotate(f'{height:.2f}', 
                     (p.get_x() + p.get_width() / 2., height),
                     ha='center', va='bottom')
plt.tight_layout()
plt.show()


