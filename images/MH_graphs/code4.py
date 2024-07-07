
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
if 'Line Graph' == "Scatter Plot":
    sns.scatterplot(x='Time Period', y='Value', hue='Indicator', data=merged_df, ax=ax)
elif 'Line Graph' == "Bar Chart":
    merged_df.plot(kind='bar', x='Time Period', y='Value', hue='Indicator', ax=ax)
elif 'Line Graph' == "Histogram":
    sns.histplot(x='Time Period', hue='Indicator', data=merged_df, multiple="stack", ax=ax)
elif 'Line Graph' == "Pie Chart":
    data = merged_df.groupby('Indicator')['Value'].sum()
    ax.pie(data, labels=data.index, autopct='%1.1f%%', startangle=90, counterclock=False)
    ax.axis('equal')
elif 'Line Graph' == "Box Plot":
    sns.boxplot(x='Time Period', y='Value', hue='Indicator', data=merged_df, ax=ax)
elif 'Line Graph' == "Line Graph":
    sns.lineplot(x='Time Period', y='Value', hue='Indicator', data=merged_df, ax=ax)
ax.set_title('Covid 19 Trends Over Time')
plt.show()
