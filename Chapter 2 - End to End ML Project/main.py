from pathlib import Path

from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
from zlib import crc32
import numpy as np
from sklearn.impute import SimpleImputer


def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok =True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio *2**32

# this whole method can be replaced with train_test_split
def split_data_with_id_hash(data,test_ratio,id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_,test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

# can be used to stratify
def straitify(data):
    splitter = StratifiedShuffleSplit(n_splits=10,test_size=0.2,random_state=42)
    strat_splits = []
    for train_index, test_index in splitter.split(data, data["income_cat"]):
        strat_train_set_n = data.iloc[train_index]
        strat_test_set_m = data.iloc[test_index]
        strat_splits.append([strat_train_set_n,strat_test_set_m])
    return strat_splits



housing = load_housing_data()
#housing.hist(bins=50, figsize=(12,8))
#plt.show()

#housing.head()
#housing.info()
#housing["ocean_proximity"].value_counts()
#housing.describe()

#housing_with_id = housing
#housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
#train_set, test_set = split_data_with_id_hash(housing_with_id,0.2,"id")


housing["income_cat"] = pd.cut(housing["median_income"], bins=[0,1.5,3.0,4.5,6,np.inf],labels=[1,2,3,4,5])

# to show the example from 58
#housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
#plt.xlabel("Income Category")
#plt.ylabel("Number of districts")
#plt.show()

# use train_test_split to get stratified data
strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2, stratify=housing["income_cat"],random_state=42)
#print(strat_test_set["income_cat"].value_counts()/len(strat_test_set))

for set_ in (strat_train_set,strat_test_set):
    set_.drop("income_cat",axis=1, inplace=True)

#housing = strat_train_set.copy()
#housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
#             s=housing["population"]/100, label="population",
#             c="median_house_value", cmap="jet", colorbar=True,
#             legend=True, sharex=False, figsize=(10,7))
#plt.show()
#print(housing)

# !!!!! TO DEBUG !!!!!
#corr_matrix = housing.corr()
#print(corr_matrix["median_house_value"].sort_values(ascending=False))

#attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
#scatter_matrix(housing[attributes],figsize=(12,0));
#plt.show()

housing = strat_train_set.drop("median_house_value",axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

imputer = SimpleImputer(strategy="median")
housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)

print(imputer.statistics_)
print(housing_num.median().values)

X = imputer.transform(housing_num)

