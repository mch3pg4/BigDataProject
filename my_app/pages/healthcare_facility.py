import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def load_data():
    # Load the data
    hospital_data = pd.read_csv('filtered_datasets/hospital.csv')
    icu_data = pd.read_csv('filtered_datasets/icu.csv')
    
    # Convert the 'date' columns to datetime
    hospital_data['date'] = pd.to_datetime(hospital_data['date'])
    icu_data['date'] = pd.to_datetime(icu_data['date'])
    
    return hospital_data, icu_data

def preprocess_data(hospital_data, icu_data, year=2020):
    # Filter the data for the selected year (default is 2020)
    hospital_data = hospital_data[hospital_data['date'].dt.year == year]
    icu_data = icu_data[icu_data['date'].dt.year == year]
    
    # Select relevant columns
    hospital_columns = ['date', 'state', 'beds_covid', 'admitted_covid', 'hosp_covid']
    icu_columns = ['date', 'state', 'beds_icu_covid', 'icu_covid', 'vent_covid']
    
    hospital_data = hospital_data[hospital_columns]
    icu_data = icu_data[icu_columns]
    
    # Combine the datasets for correlation analysis
    combined_data = pd.merge(hospital_data, icu_data, on=['date', 'state'])
    
    return hospital_data, icu_data, combined_data

def correlation_matrix_plot(data):
    st.header("Correlation Matrix")
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = data[numeric_columns].corr()
    fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
    st.plotly_chart(fig)
    st.write("This heatmap shows the correlation between various metrics in the dataset.")

def regression_analysis_plot(hospital_data):
    st.header("Regression Analysis: Beds COVID vs. Admitted COVID")
    X = hospital_data[['beds_covid']]
    y = hospital_data['admitted_covid']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_test['beds_covid'], y=y_test, mode='markers', name='Actual', marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=X_test['beds_covid'], y=y_pred, mode='lines', name='Predicted', line=dict(color='red')))
    fig.update_layout(
        xaxis_title='Beds COVID',
        yaxis_title='Admitted COVID',
        title='Regression Analysis: Beds COVID vs. Admitted COVID'
    )
    st.plotly_chart(fig)
    st.write("This plot shows the relationship between the number of COVID-19 beds and the number of admitted COVID-19 patients.")

def feature_importance_plot(icu_data):
    st.header("Feature Importance: ICU Data")
    X = icu_data[['beds_icu_covid', 'vent_covid']]
    y = icu_data['icu_covid']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    feature_importances = model.feature_importances_
    features = X.columns

    fig = px.bar(x=features, y=feature_importances, labels={'x':'Features', 'y':'Importance'}, title='Feature Importance for ICU Data')
    st.plotly_chart(fig)
    st.write("This bar chart shows the importance of different features in predicting the number of ICU COVID-19 patients.")

def main():
    st.markdown(
        """
        <style>
        .logo-container {
            display: flex;
            justify-content: center;
            margin-bottom: 20px;
        }
        .logo-image {
            max-width: 240px; /* Adjust the max-width as needed */
            max-height: 340px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Load logo from desktop
    logo_path = 'images/logo.png'  # Update with your actual file path
    try:
        with open(logo_path, 'rb') as f:
            logo_image = f.read()
            logo_image_base64 = base64.b64encode(logo_image).decode()
            st.markdown(
                f'<div class="logo-container"><img src="data:image/png;base64,{logo_image_base64}" class="logo-image"></div>',
                unsafe_allow_html=True
            )
    except FileNotFoundError:
        st.warning('Logo file not found.')

    # Load background image from desktop
    background_path = 'images/background.jpg'  # Update with your actual file path
    try:
        with open(background_path, 'rb') as f:
            background_image = f.read()
            background_image_base64 = base64.b64encode(background_image).decode()
            st.markdown(
                f"""
                <style>
                .stApp {{
                    background-image: url("data:image/jpg;base64,{background_image_base64}");
                    background-size: cover; /* Options: cover, contain, auto */
                    background-repeat: no-repeat; /* Options: no-repeat, repeat, repeat-x, repeat-y */
                    background-position: center; /* Options: left, right, top, bottom, center */
                    clip-path: ellipse(75% 75% at 50% 50%); /* Example of a shape; adjust as needed */
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
    except FileNotFoundError:
        st.warning('Background file not found.')

    # CSS for sidebar background
    sidebar_bg_path = 'images/sidebar.jpg'  # Update with your actual file path
    try:
        with open(sidebar_bg_path, 'rb') as f:
            sidebar_bg_image = f.read()
            sidebar_bg_image_base64 = base64.b64encode(sidebar_bg_image).decode()
            st.markdown(
                f"""
                <style>
                [data-testid="stSidebar"] {{
                    background-image: url("data:image/jpg;base64,{sidebar_bg_image_base64}");
                    background-size: cover; /* Options: cover, contain, auto */
                    background-repeat: no-repeat; /* Options: no-repeat, repeat, repeat-x, repeat-y */
                    background-position: center; /* Options: left, right, top, bottom, center */
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
    except FileNotFoundError:
        st.warning('Sidebar background file not found.')

    st.title("Healthcare Facility Analysis")

    # Load the data
    hospital_data, icu_data = load_data()
    
    # Sidebar for user input
    st.sidebar.title("Settings")
    year = st.sidebar.slider("Select Year", min_value=2020, max_value=2021, value=2020)
    state = st.sidebar.multiselect("Select State(s)", options=hospital_data['state'].unique(), default=hospital_data['state'].unique())
    
    # Preprocess the data
    hospital_data, icu_data, combined_data = preprocess_data(hospital_data, icu_data, year)
    
    # Filter by state
    if state:
        hospital_data = hospital_data[hospital_data['state'].isin(state)]
        icu_data = icu_data[icu_data['state'].isin(state)]
        combined_data = combined_data[combined_data['state'].isin(state)]
    
    # Display data summary
    st.sidebar.header("Data Summary")
    st.sidebar.write("Total Hospital Records:", hospital_data.shape[0])
    st.sidebar.write("Total ICU Records:", icu_data.shape[0])
    
    # Display correlation matrix
    correlation_matrix_plot(combined_data)
    
    # Display regression analysis plot
    regression_analysis_plot(hospital_data)
    
    # Display feature importance plot
    feature_importance_plot(icu_data)
    
if __name__ == "__main__":
    main()
