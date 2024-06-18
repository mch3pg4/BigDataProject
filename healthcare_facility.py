import streamlit as st

def show_facility():
    st.title('Healthcare Facility Page')
    st.write('Welcome to the healthcare facility page.')

    # Button to return to dashboard
    if st.button('Return to Dashboard', key='healthcare_facility_to_dashboard'):
        st.session_state.page = 'dashboard'
        st.experimental_rerun()  # Rerun the app to navigate immediately

# Display the current page
if st.session_state.page == 'healthcare facility':
    show_facility()
elif st.session_state.page == 'dashboard':
    st.empty()
    exec(open("dashboard.py").read())