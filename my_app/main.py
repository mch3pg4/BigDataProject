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
    def run_user_code(code):
        # Prepare a local environment for exec
        local_env = {
            "plt": plt,
            "sns": sns,
            "pd": pd,
            "merged_df": merged_df
        }
        try:
            # Execute the user code
            exec(code, local_env)

            # Check if a 'generate_graph' function is defined by the user
            if 'generate_graph' in local_env:
                fig, ax = plt.subplots()
                local_env['generate_graph'](ax)
                placeholder.pyplot(fig)
            else:
                st.error("Please define a function named 'generate_graph(ax)' to plot the graph.")
        except Exception as e:
            st.error(f"Error executing code: {e}")

    # Run the user code when the button is clicked
    if st.button("Run Code"):
        run_user_code(code)

else:
    st.info("Please upload a CSV file to proceed.")