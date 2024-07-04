import streamlit as st
from streamlit_image_select import image_select


def main():
    # scale sidebar logo to be larger
    st.markdown(
        """ <style>
    div[data-testid="stSidebarHeader"] > img, div[data-testid="collapsedControl"] > img {
        height: 5rem;
        width: auto;
    }

    div[data-testid="stSidebarHeader"], div[data-testid="stSidebarHeader"] > *,
    div[data-testid="collapsedControl"], div[data-testid="collapsedControl"] > * {
        display: flex;
        align-items: center;
    }
    </style>""", unsafe_allow_html=True
    )
    # add a logo to the sidebar
    st.logo('images/logo_full.png', icon_image='images/logo.png')

    st.title('Welcome to MED ANALYTICS. 📊')

    st.write('The MED ANALYTICS dashboard empowers users to navigate the challenges of COVID-19 through data-driven insights. This platform offersdata visualizations on healthcare facilities, mental health, and vaccination efforts, providing a comprehensive picture of the pandemic\'s impact. By leveraging machine learning, MED ANALYTICS aims to predict future trends and resource needs, supporting proactive decision-making.')

    # COVID-19 at a Glance
    st.subheader('COVID-19 at a Glance')
    st.write('COVID-19 cases in Malaysia as of 1st October 2021: 2,000,000 cases')
    st.write('COVID-19 cases in Malaysia as of 1st October 2021: 2,000,000 cases')
    st.write('COVID-19 cases in Malaysia as of 1st October 2021: 2,000,000 cases')
    st.write('COVID-19 cases in Malaysia as of 1st October 2021: 2,000,000 cases')
    st.write('COVID-19 cases in Malaysia as of 1st October 2021: 2,000,000 cases')

    # show daily cases number, hospitalization, ICU, vaccination in row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader('Daily Cases')
        st.write('1,000')
    with col2:
        st.subheader('Hospitalization')
        st.write('500')
    with col3:
        st.subheader('ICU')
        st.write('100')

    # Select topic
    st.subheader('Select a topic')
    img = image_select(
        label="Clicking on a topic brings you to the topic's respective page.",
        images=["images/btn_images/mentalhealth_btn.png",
                "images/btn_images/vaccination_btn.png", "images/btn_images/facility_btn.png"],
        captions=["Mental Health", "Vaccination", "Healthcare Facility"],
    )

    # center go to page button
    col1, col2, col3, col4, col5 = st.columns(5)
    topic_btn = col3.button('Go to topic')
    if topic_btn:
        if img == "images/btn_images/mentalhealth_btn.png":
            st.switch_page("pages/1_💭_Mental_Health.py")
        elif img == "images/btn_images/vaccination_btn.png":
            st.switch_page("pages/2_💉_Vaccination.py")
        elif img == "images/btn_images/facility_btn.png":
            st.switch_page("pages/3_🏥_Healthcare_Facility.py")


if __name__ == "__main__":
    main()
