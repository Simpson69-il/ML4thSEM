import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("diabetes.csv",encoding="latin1")

print(df.info())
print(df.describe())
print(df.isnull().sum)
plt.figure()
plt.hist(df['Glucose'].dropna())
plt.title("Glucose Level Distribution")
plt.xlabel("Glucose")
plt.ylabel("Frequency")
plt.show()
plt.figure()
plt.hist(df['Age'])
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()
plt.figure()
plt.boxplot(df['Glucose'].dropna())
plt.title("Glucose Boxplot")
plt.ylabel("Glucose")
plt.show()
plt.figure()
plt.boxplot(df['Age'])
plt.title("Age Boxplot")
plt.ylabel("Age")
plt.show()
print(df.corr())
plt.figure()
plt.imshow(df.corr())
plt.colorbar()
plt.title("Correlation Matrix")
plt.show()




