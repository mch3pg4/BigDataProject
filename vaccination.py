import streamlit as st

def show_vaccination():
    st.title('Vaccination Page')
    st.write('Welcome to the vaccination page.')

    # Button to return to dashboard
    if st.button('Return to Dashboard', key='vaccination_to_dashboard'):
        st.session_state.page = 'dashboard'
        st.experimental_rerun()  # Rerun the app to navigate immediately

# Display the current page
if st.session_state.page == 'vaccination':
    show_vaccination()
elif st.session_state.page == 'dashboard':
    st.empty()
    exec(open("dashboard.py").read())
