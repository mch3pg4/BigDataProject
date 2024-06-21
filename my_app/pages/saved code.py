import streamlit as st
from io import BytesIO

# Set up the Streamlit app for the second page
st.title("Saved Code and Graph")

if 'code' in st.session_state and 'fig' in st.session_state:
    # Display the saved code
    st.write("Saved Code:")
    st.code(st.session_state['code'], language="python")

    # Display the saved graph
    st.write("Saved Graph:")
    st.pyplot(st.session_state['fig'])

    # Function to save code as a .py file
    def save_code_as_file(code, filename="user_code.py"):
        with open(filename, "w") as file:
            file.write(code)
        st.success(f"Code saved as {filename}")

    # Function to save the plot as a .png file
    def save_plot_as_file(fig, filename="plot.png"):
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        st.download_button(
            label="Download plot as PNG",
            data=buf,
            file_name=filename,
            mime="image/png"
        )

    # Button to download the code
    st.download_button(
        label="Download Code as .py",
        data=st.session_state['code'],
        file_name="user_code.py",
        mime="text/x-python"
    )

    # Button to download the plot
    save_plot_as_file(st.session_state['fig'])

else:
    st.info("No saved code or graph found. Please generate and save on the main page.")
