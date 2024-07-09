import streamlit as st
import os
import pandas as pd


def load_code_from_file(code_filename):
    """
    Load Python code from a .py file.

    Parameters:
    - code_filename (str): Filename of the Python code file.

    Returns:
    - str: Contents of the Python code file as a string.
    """
    with open(code_filename, 'r') as f:
        code_content = f.read()
    return code_content


def load_html_from_file(html_filename):
    """
    Load HTML content from a file.

    Parameters:
    - html_filename (str): Filename of the HTML file.

    Returns:
    - str: Contents of the HTML file as a string.
    """
    with open(html_filename, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content


def main():
    # Scale sidebar logo to be larger
    st.markdown(
        """<style>
        div[data-testid="stSidebarHeader"] > img, div[data-testid="collapsedControl"] > img {
            height: 5rem;
            width: auto;
        }

        div[data-testid="stSidebarHeader"], div[data-testid="collapsedControl"] > img {
            display: flex;
            align-items: center;
        }
        </style>""", unsafe_allow_html=True
    )
    # Add a logo to the sidebar
    st.logo('images/logo_full.png', icon_image='images/logo.png')

    st.title('💭 Mental Health Page')
    st.write('Welcome to the mental health page.')

    # Directory containing images and metadata file
    image_dir = 'images/MH_graphs'
    html_dir = 'images/MH_graphs/html'
    metadata_file = os.path.join(image_dir, 'metadata.csv')

    # Check if metadata file exists
    if not os.path.exists(metadata_file):
        st.error("Metadata file not found!")
        return

    # Load metadata
    metadata = pd.read_csv(metadata_file)

    # Display each image with title, subheading, and toggle buttons
    for i, row in metadata.iterrows():
        with st.container():
            st.subheader(row['title'])
            st.write(row['subheading'])

            image_path = os.path.join(image_dir, row['filename'])
            code_filename = os.path.join(image_dir, row['code_filename'])
            html_filename = os.path.join(html_dir, row['html_filename'])

            if f"view_{i}" not in st.session_state:
                st.session_state[f"view_{i}"] = "image"

            # Use st.columns to place buttons side by side
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"Show Image {i + 1}", key=f"show_image_{i}_btn"):
                    st.session_state[f"view_{i}"] = "image"
            with col2:
                if st.button(f"Show Code {i + 1}", key=f"show_code_{i}_btn"):
                    st.session_state[f"view_{i}"] = "code"
            with col3:
                if st.button(f"Show Interactive {i + 1}", key=f"show_interactive_{i}_btn"):
                    st.session_state[f"view_{i}"] = "interactive"

            if st.session_state[f"view_{i}"] == "image":
                st.image(image_path, use_column_width=True)
            elif st.session_state[f"view_{i}"] == "code":
                st.code(load_code_from_file(code_filename), language="python")
            elif st.session_state[f"view_{i}"] == "interactive":
                html_content = load_html_from_file(html_filename)
                st.components.v1.html(html_content, height=500, scrolling=True)


if __name__ == "__main__":
    main()
