import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO


# Function to generate a downloadable code file
def generate_code_file(code):
    return BytesIO(code.encode())


# Function to generate a downloadable plot file
def generate_plot_file(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return buf


def main():
    st.title("Code Generation Page")

    # Initialize session state lists if they don't exist
    if 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = None
    if 'merged_df' not in st.session_state:
        st.session_state['merged_df'] = None
    if 'code_list' not in st.session_state:
        st.session_state['code_list'] = []
    if 'fig_list' not in st.session_state:
        st.session_state['fig_list'] = []

    # File uploader for CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Store the uploaded file in session state
        st.session_state.uploaded_file = uploaded_file
        st.session_state.merged_df = pd.read_csv(uploaded_file)

    if st.session_state.uploaded_file is not None:
        merged_df = st.session_state.merged_df

        # Display the data
        st.write("Data from the uploaded CSV file:")
        st.write(merged_df)

        # Add column filters
        st.subheader("Filter Data")
        filters = {}
        for column in merged_df.columns:
            if pd.api.types.is_numeric_dtype(merged_df[column]):
                if st.checkbox(f"Filter {column}", key=f"filter_{column}"):
                    min_value, max_value = float(merged_df[column].min()), float(merged_df[column].max())
                    filters[column] = st.slider(f"Select range for {column}", min_value, max_value,
                                                (min_value, max_value))
            elif pd.api.types.is_string_dtype(merged_df[column]):
                if st.checkbox(f"Filter {column}", key=f"filter_{column}"):
                    unique_values = merged_df[column].unique()
                    filters[column] = st.multiselect(f"Select values for {column}", unique_values,
                                                     default=unique_values)

        # Apply filters to the DataFrame
        for column, filter_value in filters.items():
            if isinstance(filter_value, tuple):
                merged_df = merged_df[(merged_df[column] >= filter_value[0]) & (merged_df[column] <= filter_value[1])]
            else:
                merged_df = merged_df[merged_df[column].isin(filter_value)]

        st.write("Filtered Data:")
        st.write(merged_df)

        # Dropdown to select x and y axes
        x_axis = st.selectbox("Select X-axis", merged_df.columns, key="x_axis")
        y_axis = st.selectbox("Select Y-axis", merged_df.columns, key="y_axis")

        #Dropdown to select indicator
        indicator = st.selectbox("Select Indicator", merged_df.columns, key="indicator")

        # Dropdown to select graph type
        graph_type = st.selectbox(
            "Select graph type",
            ("Scatter Plot", "Bar Chart", "Histogram", "Pie Chart", "Box Plot", "Line Graph"),
            key="graph_type"
        )

        # Text input for dynamic graph title
        graph_title = st.text_input("Enter Graph Title", "My Graph", key="graph_title")

        # Prepare a placeholder for the graph
        placeholder = st.empty()

        # Function to generate a graph
        def generate_graph(x, y, indicator, title):
            fig, ax = plt.subplots(figsize=(12, 6))  # Increase figure size for better visualization
            if graph_type == "Scatter Plot":
                sns.scatterplot(x=x, y=y, hue=indicator, data=merged_df, ax=ax)
            elif graph_type == "Bar Chart":
                sns.barplot(x=x, y=y, hue=indicator, data=merged_df, ax=ax)
            elif graph_type == "Histogram":
                sns.histplot(x=y, hue=indicator, data=merged_df, ax=ax)
            elif graph_type == "Pie Chart":
                data = merged_df.groupby(x)[y].sum()
                ax.pie(data, labels=data.index, autopct='%1.1f%%', startangle=90, counterclock=False)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            elif graph_type == "Box Plot":
                sns.boxplot(x=x, y=y, data=merged_df, ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # Rotate x labels for better readability
            elif graph_type == "Line Graph":
                sns.lineplot(x=x, y=y, hue=indicator, data=merged_df, ax=ax, linewidth=2.5)  # Thicker line
            ax.set_title(title)
            placeholder.pyplot(fig)
            return fig

        # Run the user code when the button is clicked
        if st.button("Run Code"):
            fig = generate_graph(x_axis, y_axis, indicator, graph_title)
            if fig:
                code = f"""
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
if '{graph_type}' == "Scatter Plot":
    sns.scatterplot(x='{x_axis}', y='{y_axis}', hue='indicator', data=merged_df, ax=ax)
elif '{graph_type}' == "Bar Chart":
    sns.barplot(x='{x_axis}', y='{y_axis}', hue='indicator', data=merged_df, ax=ax)
elif '{graph_type}' == "Histogram":
    sns.histplot(x='{y_axis}', hue='indicator', data=merged_df, ax=ax)
elif '{graph_type}' == "Pie Chart":
    data = merged_df.groupby('{x_axis}')['{y_axis}'].sum()
    ax.pie(data, labels=data.index, autopct='%1.1f%%', startangle=90, counterclock=False)
    ax.axis('equal')
elif '{graph_type}' == "Box Plot":
    sns.boxplot(x='{x_axis}', y='{y_axis}', data=merged_df, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
elif '{graph_type}' == "Line Graph":
    sns.lineplot(x='{x_axis}', y='{y_axis}', hue='indicator', data=merged_df, ax=ax, linewidth=2.5)
ax.set_title('{graph_title}')
plt.show()
"""
                st.session_state['code_list'].append(code)
                st.session_state['fig_list'].append(fig)
                st.success("Code generated successfully!")
    else:
        st.info("Please upload a CSV file to proceed.")


if __name__ == "__main__":
    main()
