import streamlit as st

def show_dashboard():
    st.title('Dashboard')
    st.write('Welcome to the dashboard.')

    # Button to navigate to vaccination page
    if st.button('Go to Vaccination Page', key='dashboard_to_vaccination'):
        st.session_state.page = 'vaccination'
        st.experimental_rerun()  
    # Button to navigate to mental health page
    if st.button('Go to Mental Health Page', key='dashboard_to_mental_health'):
        st.session_state.page = 'mental health'
        st.experimental_rerun() 
    # Button to navigate to healthcare facility page
    if st.button('Go to Healthcare Facility Page', key='dashboard_to_healthcare_facility'):
        st.session_state.page = 'healthcare facility'
        st.experimental_rerun()  

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'dashboard'

# Display the current page
if st.session_state.page == 'dashboard':
    show_dashboard()
elif st.session_state.page == 'vaccination':
    st.empty()
    exec(open("vaccination.py").read())
elif st.session_state.page == 'mental health':
    st.empty()
    exec(open("mental_health.py").read())
elif st.session_state.page == 'healthcare facility':
    st.empty()
    exec(open("healthcare_facility.py").read())
