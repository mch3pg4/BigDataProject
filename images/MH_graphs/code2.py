
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
    sns.scatterplot(x='Low CI', y='High CI', data=merged_df, ax=ax)
elif 'Histogram' == "Bar Chart":
    merged_df.plot(kind='bar', x='Low CI', y='High CI', ax=ax)
elif 'Histogram' == "Histogram":
    merged_df['High CI'].plot(kind='hist', ax=ax)
elif 'Histogram' == "Pie Chart":
    data = merged_df.groupby('Low CI')['High CI'].sum()
    ax.pie(data, labels=data.index, autopct='%1.1f%%', startangle=90, counterclock=False)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
elif 'Histogram' == "Box Plot":
    sns.boxplot(x='Low CI', y='High CI', data=merged_df, ax=ax)
elif 'Histogram' == "Line Graph":
    sns.lineplot(x='Low CI', y='High CI', data=merged_df, ax=ax)
ax.set_title('Low CI vs High CI')
plt.show()
