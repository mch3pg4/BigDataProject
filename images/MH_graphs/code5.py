
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
if 'Scatter Plot' == "Scatter Plot":
    sns.scatterplot(x='Maternal_Age', y='Edinburgh_Postnatal_Depression_Scale', hue='Maternal_Age', data=merged_df, ax=ax)
elif 'Scatter Plot' == "Bar Chart":
    merged_df.plot(kind='bar', x='Maternal_Age', y='Edinburgh_Postnatal_Depression_Scale', hue='Maternal_Age', ax=ax)
elif 'Scatter Plot' == "Histogram":
    sns.histplot(x='Maternal_Age', hue='Maternal_Age', data=merged_df, multiple="stack", ax=ax)
elif 'Scatter Plot' == "Pie Chart":
    data = merged_df.groupby('Maternal_Age')['Edinburgh_Postnatal_Depression_Scale'].sum()
    ax.pie(data, labels=data.index, autopct='%1.1f%%', startangle=90, counterclock=False)
    ax.axis('equal')
elif 'Scatter Plot' == "Box Plot":
    sns.boxplot(x='Maternal_Age', y='Edinburgh_Postnatal_Depression_Scale', hue='Maternal_Age', data=merged_df, ax=ax)
elif 'Scatter Plot' == "Line Graph":
    sns.lineplot(x='Maternal_Age', y='Edinburgh_Postnatal_Depression_Scale', hue='Maternal_Age', data=merged_df, ax=ax)
ax.set_title('Scatter Plot of Maternal Age vs EPDS')
plt.show()
