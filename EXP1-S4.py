import pandas as pd

print("Mohd Shafique RB -24BAD074 ")
dataframe = pd.read_csv("marketing_campaign.csv", sep="\t" ,encoding="latin1")
print(dataframe.head())

print(dataframe.info())
print(dataframe.isnull().sum())

import matplotlib.pyplot as mpp

dataframe["Age"] = 2026 - dataframe["Year_Birth"]

mpp.figure()
dataframe["Age"].value_counts().sort_index().plot(kind="bar")
mpp.show()
mpp.figure()
mpp.boxplot(dataframe["Income"].dropna())
mpp.show()
dataframe["Spending"] = dataframe["MntWines"] + dataframe["MntFruits"] + dataframe["MntMeatProducts"] + dataframe["MntFishProducts"] + dataframe["MntSweetProducts"] + dataframe["MntGoldProds"]

mpp.figure()
mpp.boxplot(dataframe["Spending"])
mpp.show()

print(dataframe[["Age","Income","Spending"]].describe())
