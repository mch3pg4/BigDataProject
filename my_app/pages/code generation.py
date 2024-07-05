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
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf

def main():
    st.title("Code Generation Page")

    # Initialize session state lists if they don't exist
    if 'code_list' not in st.session_state:
        st.session_state['code_list'] = []
    if 'fig_list' not in st.session_state:
        st.session_state['fig_list'] = []

    # File uploader for CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load the data
        merged_df = pd.read_csv(uploaded_file)

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
                    filters[column] = st.slider(f"Select range for {column}", min_value, max_value, (min_value, max_value))
            elif pd.api.types.is_string_dtype(merged_df[column]):
                if st.checkbox(f"Filter {column}", key=f"filter_{column}"):
                    unique_values = merged_df[column].unique()
                    filters[column] = st.multiselect(f"Select values for {column}", unique_values, default=unique_values)

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

        # Dropdown to select graph type
        graph_type = st.selectbox(
            "Select graph type",
            ("Scatter Plot", "Bar Chart", "Histogram", "Pie Chart", "Box Plot"),
            key="graph_type"
        )

        # Text input for dynamic graph title
        graph_title = st.text_input("Enter Graph Title", "My Graph", key="graph_title")

        # Prepare a placeholder for the graph
        placeholder = st.empty()

        # Function to generate a graph
        def generate_graph(x, y, title):
            fig, ax = plt.subplots()
            if graph_type == "Scatter Plot":
                sns.scatterplot(x=x, y=y, data=merged_df, ax=ax)
            elif graph_type == "Bar Chart":
                merged_df.plot(kind='bar', x=x, y=y, ax=ax)
            elif graph_type == "Histogram":
                merged_df[y].plot(kind='hist', ax=ax)
            elif graph_type == "Pie Chart":
                merged_df.set_index(x)[y].plot(kind='pie', ax=ax)
            elif graph_type == "Box Plot":
                sns.boxplot(x=x, y=y, data=merged_df, ax=ax)
            ax.set_title(title)
            placeholder.pyplot(fig)
            return fig

        # Run the user code when the button is clicked
        if st.button("Run Code"):
            fig = generate_graph(x_axis, y_axis, graph_title)
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
fig, ax = plt.subplots()
if '{graph_type}' == "Scatter Plot":
    sns.scatterplot(x='{x_axis}', y='{y_axis}', data=merged_df, ax=ax)
elif '{graph_type}' == "Bar Chart":
    merged_df.plot(kind='bar', x='{x_axis}', y='{y_axis}', ax=ax)
elif '{graph_type}' == "Histogram":
    merged_df['{y_axis}'].plot(kind='hist', ax=ax)
elif '{graph_type}' == "Pie Chart":
    merged_df.set_index('{x_axis}')['{y_axis}'].plot(kind='pie', ax=ax)
elif '{graph_type}' == "Box Plot":
    sns.boxplot(x='{x_axis}', y='{y_axis}', data=merged_df, ax=ax)
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
