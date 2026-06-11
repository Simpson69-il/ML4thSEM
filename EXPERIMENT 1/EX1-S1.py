import pandas as pd
import matplotlib.pyplot as mpp
dataframe=pd.read_csv('data.csv', encoding="latin1")
print(dataframe.head())
print(dataframe.tail())
print(dataframe.info())
print(dataframe.describe())
print(dataframe.isnull().sum())

dataframe['Sales'] = dataframe['Quantity'] * dataframe['UnitPrice']
sales_per_product = dataframe.groupby('Description')['Sales'].sum().head(10)


#barhcart
mpp.bar(sales_per_product.index, sales_per_product.values)
mpp.xlabel("Product")
mpp.ylabel("Total Sales")
mpp.title("Sales per Product (Bar Chart)")
mpp.xticks(rotation=90)
mpp.show()
#linechart
mpp.plot(sales_per_product.index, sales_per_product.values, marker='o')
mpp.xlabel("Product")
mpp.ylabel("Total Sales")
mpp.title("Sales per Product (Line Chart)")
mpp.show()
