import streamlit as st

def show_mental_health():
    st.title('Mental Health Page')
    st.write('Welcome to the mental health page.')

    # Button to return to dashboard
    if st.button('Return to Dashboard', key='mental_health_to_dashboard'):
        st.session_state.page = 'dashboard'
        st.experimental_rerun()  # Rerun the app to navigate immediately

# Display the current page
if st.session_state.page == 'mental health':
    show_mental_health()
elif st.session_state.page == 'dashboard':
    st.empty()
    exec(open("dashboard.py").read())
