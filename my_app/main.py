import streamlit as st
from streamlit_ace import st_ace
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set up the Streamlit app
st.title("Streamlit Code Editor with Graph Execution")

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
    local_env = {}
    try:
        # Execute the user code
        exec(code, {"np": np, "plt": plt}, local_env)

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