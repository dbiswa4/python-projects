import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    sales = pd.read_csv("sample-salesv2.csv", parse_dates=['date'])
    print sales.columns
    print sales[:10]
    print sales.head()
    print sales.describe()
    print sales['unit price'].describe()
    print sales.dtypes

    customers = sales[['name', 'ext price', 'date']]
    customers.head()
    customer_group = customers.groupby('name')
    customer_group.size()
    sales_totals = customer_group.sum()
    #sales_totals.sort(by='ext price')
    my_plot = sales_totals.plot(kind='bar')

    my_plot = sales_totals.plot(kind='bar', legend=None, title="Total Sales by Customer")
    my_plot.set_xlabel("Customers")
    my_plot.set_ylabel("Sales ($)")
    fig = my_plot.get_figure()
    fig.savefig("dasdas.png")

    purchase_patterns = sales[['ext price', 'date']]
    print purchase_patterns.head()

    purchase_plot = purchase_patterns['ext price'].hist(bins=20)
    purchase_plot.set_title("Purchase Patterns")
    purchase_plot.set_xlabel("Order Amount($)")
    purchase_plot.set_ylabel("Number of orders")
