
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
fig, ax = plt.subplots(figsize=(12, 6))
if 'Box Plot' == "Scatter Plot":
    sns.scatterplot(x='Household_Income', y='Maternal_Age', hue='indicator', data=merged_df, ax=ax)
elif 'Box Plot' == "Bar Chart":
    sns.barplot(x='Household_Income', y='Maternal_Age', hue='indicator', data=merged_df, ax=ax)
elif 'Box Plot' == "Histogram":
    sns.histplot(x='Maternal_Age', hue='indicator', data=merged_df, ax=ax)
elif 'Box Plot' == "Pie Chart":
    data = merged_df.groupby('Household_Income')['Maternal_Age'].sum()
    ax.pie(data, labels=data.index, autopct='%1.1f%%', startangle=90, counterclock=False)
    ax.axis('equal')
elif 'Box Plot' == "Box Plot":
    sns.boxplot(x='Household_Income', y='Maternal_Age', data=merged_df, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
elif 'Box Plot' == "Line Graph":
    sns.lineplot(x='Household_Income', y='Maternal_Age', hue='indicator', data=merged_df, ax=ax, linewidth=2.5)
ax.set_title('Distribution of Maternal Age Across Different Household Income Levels')
plt.show()
