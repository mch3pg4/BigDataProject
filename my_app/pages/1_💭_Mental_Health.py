import streamlit as st
import os
import pandas as pd


def load_code_from_file(code_filename):
    with open(code_filename, 'r') as f:
        code_content = f.read()
    return code_content


def load_html_from_file(html_filename):
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
    # Introductory section
    st.write("""
           ## Welcome to the Mental Health Page
           This page is dedicated to providing insights into various aspects of mental health through interactive and static graphs. 
           Mental health is a critical part of overall well-being, and understanding the trends, challenges, and data behind it 
           can help in forming better support systems and interventions.

           ### Types of Graphs:
           - **Anxiety and Depression Graphs**: These graphs showcase trends, statistics, and insights related to anxiety and depression.
           - **Mental Health During Pregnancy**: These graphs focus on the psychological well-being of expectant mothers, presenting relevant data and analysis.
       
       
       
       """)

    # Directory containing images and metadata file
    image_dir = 'images/MH_graphs'
    html_dir = 'images/MH_graphs/html'
    metadata_file = 'images/MH_graphs/metadata.csv'

    if not os.path.exists(metadata_file):
        st.error("Metadata file not found!")
        return

    metadata = pd.read_csv(metadata_file)

    anxiety_depression = metadata[metadata['category'] == 'Anxiety and Depression']
    pregnancy = metadata[metadata['category'] == 'Pregnancy']

    tabs = st.tabs(["Anxiety and Depression", "Mental Health During Pregnancy"])

    with tabs[0]:
        st.header("Anxiety and Depression Graphs")
        st.write(
            "These graphs showcase trends, statistics, and insights related to anxiety and depression across the "
            "United States, highlighting the variability and distribution of symptoms over time.")
        display_graphs(anxiety_depression, image_dir, html_dir)

    with tabs[1]:
        st.header("Mental Health During Pregnancy Graphs")
        st.write(
            "These graphs focus on the psychological well-being of expectant mothers across Canada, presenting "
            "relevant data and analysis on maternal age versus EPDS scores, NICU stay across delivery modes, "
            "and maternal age across different household income levels.")
        display_graphs(pregnancy, image_dir, html_dir)


def display_graphs(metadata, image_dir, html_dir):
    for i, row in metadata.iterrows():
        with st.container():
            st.subheader(row['title'])
            st.write(row['subheading'])
            st.write(row['description'])

            image_path = os.path.join(image_dir, row['filename'])
            code_filename = os.path.join(image_dir, row['code_filename'])
            html_filename = os.path.join(html_dir, row['html_filename'])

            if f"view_{i}" not in st.session_state:
                st.session_state[f"view_{i}"] = "image"

            st.markdown(
                """<style>
                .stButton > button {
                    margin-right: 5px;
                }
                </style>""", unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns([1, 1, 1])
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
