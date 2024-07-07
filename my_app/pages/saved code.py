import streamlit as st
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


def saved_code_page():
    st.title("Saved Code and Graph Page")

    if 'code_list' in st.session_state and 'fig_list' in st.session_state:
        if st.session_state['code_list'] and st.session_state['fig_list']:
            for i, (code, fig) in enumerate(zip(st.session_state['code_list'], st.session_state['fig_list'])):
                st.write(f"Saved Code {i + 1}:")
                st.code(code, language="python")
                st.pyplot(fig)

                # Generate a download button for the code
                code_download = generate_code_file(code)
                st.download_button(
                    label=f"Download Code {i + 1} as .py",
                    data=code_download,
                    file_name=f"code{i + 1}.py",
                    mime="text/x-python"
                )

                # Generate a download button for the plot
                plot_download = generate_plot_file(fig)
                st.download_button(
                    label=f"Download plot {i + 1} as PNG",
                    data=plot_download,
                    file_name=f"plot_{i + 1}.png",
                    mime="image/png"
                )
        else:
            st.info("No saved code or graph found. Please generate and save on the main page.")
    else:
        st.info("No saved code or graph found. Please generate and save on the main page.")


if __name__ == "__main__":
    saved_code_page()
