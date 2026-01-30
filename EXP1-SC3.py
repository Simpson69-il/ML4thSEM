import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Housing.csv")

print(df.head())
print(df.columns)
print(df.info())
print(df.isnull().sum())
print(df.describe())
plt.figure()
plt.scatter(df['area'], df['price'])
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Area vs Price")
plt.show()
plt.figure()
plt.scatter(df['bedrooms'], df['price'])
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.title("Bedrooms vs Price")
plt.show()

corr = df.corr(numeric_only=True)

plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.title("Correlation Heatmap")
plt.show()

price_corr = corr['price'].sort_values(ascending=False)

for f,v in price_corr.items():
    if f!="price":
        if v>0.5:
            print(f,"strong")
        elif v>0.3:
            print(f,"moderate")
        elif v>0.1:
            print(f,"weak")

