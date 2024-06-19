import streamlit as st
from streamlit_ace import st_ace
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the Streamlit app
st.title("Streamlit Code Editor with Graph Execution")

# File uploader for CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the data
    merged_df = pd.read_csv(uploaded_file)

    # Display the data
    st.write("Data from the uploaded CSV file:")
    st.write(merged_df)

    # Dropdown to select x and y axes
    x_axis = st.selectbox("Select X-axis", merged_df.columns)
    y_axis = st.selectbox("Select Y-axis", merged_df.columns)

    # Dropdown to select graph type
    graph_type = st.selectbox(
        "Select graph type",
        ("Scatter Plot", "Bar Chart", "Histogram", "Pie Chart", "Box Plot")
    )

    # Add the Ace code editor
    code = st_ace(
        placeholder="Type your code here...",
        language="python",
        theme="monokai",
        keybinding="vscode",
        min_lines=10,
        max_lines=30,
        font_size=14,
        show_gutter=True,
        wrap=True,
        auto_update=True,
        readonly=False,
        key="ace-editor",
    )

    # Display the code from the editor
    st.write("Code from the editor:")
    st.code(code, language="python")

    # Prepare a placeholder for the graph
    placeholder = st.empty()

    # Function to execute user code and generate a graph
    def run_user_code(code, x, y):
        # Prepare a local environment for exec
        local_env = {
            "plt": plt,
            "sns": sns,
            "pd": pd,
            "merged_df": merged_df,
            "x": x,
            "y": y
        }
        try:
            # Execute the user code
            exec(code, local_env)

            # Generate the selected graph
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
            placeholder.pyplot(fig)
        except Exception as e:
            st.error(f"Error executing code: {e}")

    # Run the user code when the button is clicked
    if st.button("Run Code"):
        run_user_code(code, x_axis, y_axis)

else:
    st.info("Please upload a CSV file to proceed.")