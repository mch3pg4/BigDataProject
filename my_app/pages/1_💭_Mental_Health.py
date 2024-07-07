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
    st.sidebar.image('../images/logo_full.png', use_column_width=True)

    st.title('💭 Mental Health Page')
    st.write('Welcome to the mental health page.')

    # Directory containing images and metadata file
    image_dir = '../images/MH_graphs'
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

            # Button to toggle between image and code view
            if st.button(f"View Image {i + 1}", key=f"view_image_{i}_btn"):
                st.session_state[f"view_{i}"] = "image"
            if st.button(f"View Code {i + 1}", key=f"view_code_{i}_btn"):
                st.session_state[f"view_{i}"] = "code"

            # Display image or code based on session state
            if st.session_state.get(f"view_{i}") == "image":
                st.image(image_path, use_column_width=True)
            else:
                st.code(load_code_from_file(code_filename), language="python")

if __name__ == "__main__":
    main()
