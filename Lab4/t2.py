import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import OneHotEncoder
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]
pd_data = pd.read_csv("adult.data", names=columns, na_values="?", skipinitialspace=True)

types = [
    ('age', float),
    ('workclass', 'U32'),
    ('fnlwgt', float),
    ('education', 'U32'),
    ('education_num', float),
    ('marital_status', 'U32'),
    ('occupation', 'U32'),
    ('relationship', 'U32'),
    ('race', 'U32'),
    ('sex', 'U16'),
    ('capital_gain', float),
    ('capital_loss', float),
    ('hours_per_week', float),
    ('native_country', 'U32'),
    ('income', 'U8')
]

np_data = np.genfromtxt("adult.data", delimiter=",", missing_values="?", autostrip=True, dtype=types, encoding="UTF-8",)

for column in columns:
    pd_data.fillna({column: pd_data[column].mode()[0]}, inplace=True)

for name in np_data.dtype.names:
    na = np_data[name][np_data[name] != "?"]
    unique_values, counts = np.unique(na, return_counts=True)
    most_used = np.argmax(counts)
    value = unique_values[most_used]
    np_data[name][np_data[name] == "?"] = value

pd_standardized = pd_data.copy()
for column in pd_standardized.columns:
    if pd.api.types.is_numeric_dtype(pd_standardized[column]):
        mean = pd_standardized[column].mean()
        std = pd_standardized[column].std()
        if std != 0:
            pd_standardized[column] = ((pd_standardized[column] - mean)/std)
        else:
            pd_standardized[column] = 0

np_standardized = np_data.copy()
for name in np_data.dtype.names:
    if np.issubdtype(np_data[name].dtype, np.number):
        col = np_data[name]
        mean = col.mean()
        std = col.std()
        if std != 0:
            np_standardized[name] = ((col - mean)/std)
        else:
            np_standardized[name] = 0

attribute = pd_standardized["hours-per-week"]
plt.figure(figsize=(12, 6))
plt.title("Гістограма pandas")
sns.histplot(attribute, bins=10)
plt.xlabel("hours per week")
plt.ylabel("Кількість")
plt.show()

attribute = np_standardized["hours_per_week"]
plt.figure(figsize=(12, 6))
plt.title("Гістограма numpy")
sns.histplot(attribute, bins=10)
plt.xlabel("hours per week")
plt.ylabel("Кількість")
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(data=pd_data, x="age", y="hours-per-week")
plt.show()

x = np_data["age"]
y = np_data["hours_per_week"]
plt.figure(figsize=(12, 6))
plt.scatter(x, y)
plt.show()

x = pd_standardized["age"]
y = pd_standardized["education-num"]
pearson_corr, pearson_p = pearsonr(x, y)
spearman_corr, spearman_p = spearmanr(x, y)
print(f"Коефіцієнт Пірсона: {pearson_corr:.4f}, p-value: {pearson_p:.4f}")
print(f"Коефіцієнт Спірмена: {spearman_corr:.4f}, p-value: {spearman_p:.4f}")

x = np_standardized["age"]
y = np_standardized["education_num"]
pearson_corr, pearson_p = pearsonr(x, y)
spearman_corr, spearman_p = spearmanr(x, y)
print(f"Коефіцієнт Пірсона: {pearson_corr:.4f}, p: {pearson_p:.4f}")
print(f"Коефіцієнт Спірмена: {spearman_corr:.4f}, p: {spearman_p:.4f}")

data = pd_data[["workclass"]]
encoder = OneHotEncoder(sparse_output=False)
encoded_data = encoder.fit_transform(data)
feature_names = encoder.get_feature_names_out(['workclass'])
print("names =", feature_names)
print(encoded_data)

category = np_data[["workclass"]].reshape(-1, 1)
encoder = OneHotEncoder(sparse_output=False)
encoded_data = encoder.fit_transform(data)
feature_names = encoder.get_feature_names_out(['workclass'])
print("names =", feature_names)
print(encoded_data)

table = pd_data.pivot_table(
    index="workclass",
    columns="native-country",
    values="hours-per-week",
    aggfunc="mean"
)

plt.figure(figsize=(12, 6))
sns.heatmap(table, annot=True)
plt.title('Hours per week')
plt.xlabel('Country')
plt.ylabel('Workclass')
plt.show()