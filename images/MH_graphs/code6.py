
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
merged_df = pd.read_csv('path/to/your/csv/file.csv')

# Apply filters
for column, filter_value in filters.items():
    if isinstance(filter_value, tuple):
        merged_df = merged_df[(merged_df[column] >= filter_value[0]) & (merged_df[column] <= filter_value[1])]
    else:
        merged_df = merged_df[merged_df[column].isin(filter_value)]

# Generate the graph
fig, ax = plt.subplots()
if 'Histogram' == "Scatter Plot":
    sns.scatterplot(x='Delivery_Mode', y='NICU_Stay', hue='Delivery_Mode', data=merged_df, ax=ax)
elif 'Histogram' == "Bar Chart":
    merged_df.plot(kind='bar', x='Delivery_Mode', y='NICU_Stay', hue='Delivery_Mode', ax=ax)
elif 'Histogram' == "Histogram":
    sns.histplot(x='Delivery_Mode', hue='Delivery_Mode', data=merged_df, multiple="stack", ax=ax)
elif 'Histogram' == "Pie Chart":
    data = merged_df.groupby('Delivery_Mode')['NICU_Stay'].sum()
    ax.pie(data, labels=data.index, autopct='%1.1f%%', startangle=90, counterclock=False)
    ax.axis('equal')
elif 'Histogram' == "Box Plot":
    sns.boxplot(x='Delivery_Mode', y='NICU_Stay', hue='Delivery_Mode', data=merged_df, ax=ax)
elif 'Histogram' == "Line Graph":
    sns.lineplot(x='Delivery_Mode', y='NICU_Stay', hue='Delivery_Mode', data=merged_df, ax=ax)
ax.set_title('NICU Stay Across Delivery Modes')
plt.show()
