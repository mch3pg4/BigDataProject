import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO


# Function to generate a downloadable code file
def generate_code_file(code):
    return BytesIO(code.encode())


# Function to generate a downloadable plot file
def generate_plot_file(fig):
    buf = BytesIO()
    fig.write_image(buf, format="png")
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

        # Function to generate an interactive graph
        def generate_graph(x, y, indicator, title):
            if graph_type == "Scatter Plot":
                fig = px.scatter(merged_df, x=x, y=y, color=indicator, title=title)
            elif graph_type == "Bar Chart":
                fig = px.bar(merged_df, x=x, y=y, color=indicator, title=title)
            elif graph_type == "Histogram":
                fig = px.histogram(merged_df, x=x, color=indicator, title=title)
            elif graph_type == "Pie Chart":
                data = merged_df.groupby(x)[y].sum().reset_index()
                fig = px.pie(data, values=y, names=x, title=title)
            elif graph_type == "Box Plot":
                fig = px.box(merged_df, x=x, y=y, color=indicator, title=title)
            elif graph_type == "Line Graph":
                fig = px.line(merged_df, x=x, y=y, color=indicator, title=title)
            placeholder.plotly_chart(fig)
            return fig

        # Run the user code when the button is clicked
        if st.button("Run Code"):
            fig = generate_graph(x_axis, y_axis, indicator, graph_title)
            if fig:
                code = f"""
import pandas as pd
import plotly.express as px

# Load the data
merged_df = pd.read_csv('path/to/your/csv/file.csv')

# Apply filters
for column, filter_value in filters.items():
    if isinstance(filter_value, tuple):
        merged_df = merged_df[(merged_df[column] >= filter_value[0]) & (merged_df[column] <= filter_value[1])]
    else:
        merged_df = merged_df[merged_df[column].isin(filter_value)]

# Generate the graph
if '{graph_type}' == "Scatter Plot":
    fig = px.scatter(merged_df, x='{x_axis}', y='{y_axis}', color='{indicator}', title='{graph_title}')
elif '{graph_type}' == "Bar Chart":
    fig = px.bar(merged_df, x='{x_axis}', y='{y_axis}', color='{indicator}', title='{graph_title}')
elif '{graph_type}' == "Histogram":
    fig = px.histogram(merged_df, x='{x_axis}', color='{indicator}', title='{graph_title}')
elif '{graph_type}' == "Pie Chart":
    data = merged_df.groupby('{x_axis}')['{y_axis}'].sum().reset_index()
    fig = px.pie(data, values='{y_axis}', names='{x_axis}', title='{graph_title}')
elif '{graph_type}' == "Box Plot":
    fig = px.box(merged_df, x='{x_axis}', y='{y_axis}', color='{indicator}', title='{graph_title}')
elif '{graph_type}' == "Line Graph":
    fig = px.line(merged_df, x='{x_axis}', y='{y_axis}', color='{indicator}', title='{graph_title}')
fig.show()
"""
                st.session_state['code_list'].append(code)
                st.session_state['fig_list'].append(fig)
                st.success("Code generated successfully!")

    else:
        st.info("Please upload a CSV file to proceed.")


if __name__ == "__main__":
    main()
