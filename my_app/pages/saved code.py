import streamlit as st
from io import BytesIO
import plotly.graph_objs as go


# Function to generate a downloadable code file
def generate_code_file(code):
    return BytesIO(code.encode())


# Function to generate a downloadable plot file (static image)
def generate_plot_file(fig):
    buf = BytesIO()
    # Adjust layout to ensure the plot is tightly bounded and properly sized
    fig.update_layout(
        margin=dict(l=80, r=10, t=60, b=60),
        width=900,  # You can adjust the width as needed
        height=700  # You can adjust the height as needed
    )
    fig.write_image(buf, format="png")
    buf.seek(0)
    return buf


# Function to save Plotly figure as an interactive HTML file
def generate_plot_html(fig):
    html_str = fig.to_html()
    buf = BytesIO(html_str.encode('utf-8'))
    buf.seek(0)
    return buf


def saved_code_page():
    st.title("Saved Code and Graph Page")

    if 'code_list' in st.session_state and 'fig_list' in st.session_state:
        if st.session_state['code_list'] and st.session_state['fig_list']:
            for i, (code, fig) in enumerate(zip(st.session_state['code_list'], st.session_state['fig_list'])):
                st.write(f"Saved Code {i + 1}:")
                st.code(code, language="python")

                # Add a toggle switch for static vs interactive plot
                plot_type = st.radio(f"Select Plot Type for Graph {i + 1}", ('Interactive', 'Static Image'),
                                     key=f"plot_type_{i}")

                if plot_type == 'Interactive':
                    st.plotly_chart(fig)
                else:
                    # Generate a static image of the plot with adjusted settings
                    plot_download = generate_plot_file(fig)
                    st.image(plot_download.getvalue())

                # Generate a download button for the code
                code_download = generate_code_file(code)
                st.download_button(
                    label=f"Download Code {i + 1} as .py",
                    data=code_download,
                    file_name=f"user_code_{i + 1}.py",
                    mime="text/x-python"
                )

                # Generate a download button for the plot
                if plot_type == 'Interactive':
                    plot_download = generate_plot_html(fig)
                    st.download_button(
                        label=f"Download Interactive Plot {i + 1} as HTML",
                        data=plot_download,
                        file_name=f"plot_{i + 1}.html",
                        mime="text/html"
                    )
                else:
                    plot_download = generate_plot_file(fig)
                    st.download_button(
                        label=f"Download Static Plot {i + 1} as PNG",
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