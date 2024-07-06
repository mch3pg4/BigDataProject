import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

def load_data():
    # Load the data
    hospital_data = pd.read_csv('filtered_datasets/hospital.csv')
    icu_data = pd.read_csv('filtered_datasets/icu.csv')
    cases_data = pd.read_csv('filtered_datasets/cases_state.csv')
    
    # Convert the 'date' columns to datetime
    hospital_data['date'] = pd.to_datetime(hospital_data['date'])
    icu_data['date'] = pd.to_datetime(icu_data['date'])
    cases_data['date'] = pd.to_datetime(cases_data['date'])
    
    return hospital_data, icu_data, cases_data

def preprocess_data(hospital_data, icu_data, cases_data, start_year=2020, end_year=2024):
    # Filter the data for the selected year range
    hospital_data = hospital_data[(hospital_data['date'].dt.year >= start_year) & (hospital_data['date'].dt.year <= end_year)]
    icu_data = icu_data[(icu_data['date'].dt.year >= start_year) & (icu_data['date'].dt.year <= end_year)]
    cases_data = cases_data[(cases_data['date'].dt.year >= start_year) & (cases_data['date'].dt.year <= end_year)]
    
    return hospital_data, icu_data, cases_data

def line_graph(hospital_data):
    fig = px.line(hospital_data, x='date', y='beds_covid',
                  title='COVID-19 Beds Over Time')
    st.plotly_chart(fig)
    st.write("This line graph shows the number of COVID-19 beds over time.")

def heatmap_plot(hospital_data, icu_data):
    combined_data = pd.merge(hospital_data, icu_data, on=['date', 'state'])
    # Ensure only numeric columns are included
    numeric_columns = combined_data.select_dtypes(
        include=['float64', 'int64']).columns
    correlation_matrix = combined_data[numeric_columns].corr()
    fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto",
                    color_continuous_scale='RdBu_r', title='Correlation Matrix')
    st.plotly_chart(fig)
    st.write("This heatmap plot shows a heatmap showing the correlation matrix between numeric columns from combined hospital and ICU data.")

def bubble_chart(hospital_data):
    fig = px.scatter(hospital_data, x='beds_covid', y='admitted_covid', size='hosp_covid', color='state',
                     title='COVID-19 Beds vs Admitted Patients', labels={'hosp_covid': 'Hospitalized COVID-19'})
    st.plotly_chart(fig)
    st.write("This bubble chart shows the comparison between COVID-19 beds with admitted patients, colored by state and sized by hospitalized COVID-19 patients.")

def histogram(hospital_data):
    fig = px.histogram(hospital_data, x='hosp_covid', nbins=50, title='Distribution of Hospitalized COVID-19 Patients')
    st.plotly_chart(fig)
    st.write("This histogram shows the distribution of hospitalized COVID-19 patients.")

def pie_chart(hospital_data):
    latest_data = hospital_data[hospital_data['date'] == hospital_data['date'].max()]
    fig = px.pie(latest_data, values='beds_covid', names='state', title='Distribution of COVID-19 Beds by State')
    st.plotly_chart(fig)
    st.write("This pie chart shows the distribution of COVID-19 beds by state for the latest date available in the dataset.")

def scatter_plot(hospital_data, cases_data):
    hospital_cases_data = pd.merge(hospital_data, cases_data, on=['date', 'state'])
    hospital_cases_data = hospital_cases_data.dropna(subset=['cases_new', 'admitted_total'])
    
    X = hospital_cases_data[['cases_new']]
    y = hospital_cases_data['admitted_total']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Set up the pipeline with scaling and regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    
    # Set up the parameter grid for GridSearchCV
    param_grid = {
        'regressor__fit_intercept': [True, False]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    st.write(f"Best parameters found: {grid_search.best_params_}")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_test['cases_new'], y=y_test, mode='markers', name='Actual'))
    fig.add_trace(go.Scatter(x=X_test['cases_new'], y=y_pred, mode='lines', name='Predicted', line=dict(dash='dash')))
    
    fig.update_layout(title='Regression Analysis: New Cases vs. Hospital Admissions',
                      xaxis_title='New COVID-19 Cases',
                      yaxis_title='Total Hospital Admissions')
    
    # Display the plot in Streamlit
    st.plotly_chart(fig)
    st.write("This scatter plot shows the relationship between the number of new COVID-19 cases and the number of hospital admissions.")

def bar_chart(icu_data):
    X = icu_data[['beds_icu_covid', 'vent_covid']]
    y = icu_data['icu_covid']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize the RandomForestRegressor
    model = RandomForestRegressor(random_state=42)
    
    # Set up a simplified parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],  # Reduced range
        'max_depth': [None, 10],     # Reduced range
        'min_samples_split': [2, 5], # Reduced range
        'min_samples_leaf': [1, 2]   # Reduced range
    }
    
    # Use RandomizedSearchCV for efficiency
    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1) # Reduced cv folds
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    st.write(f"Best parameters found: {grid_search.best_params_}")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    
    feature_importances = best_model.feature_importances_
    features = X.columns
    fig = go.Figure([go.Bar(x=features, y=feature_importances)])
    fig.update_layout(title='Feature Importance for ICU Data',
                      xaxis_title='Features',
                      yaxis_title='Importance')
    
    st.plotly_chart(fig)
    st.write("This bar chart shows the importance of different features in predicting the number of ICU COVID-19 patients.")

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

    # # Load background image from desktop
    # background_path = 'images/background.jpg'  # Update with your actual file path
    # try:
    #     with open(background_path, 'rb') as f:
    #         background_image = f.read()
    #         background_image_base64 = base64.b64encode(
    #             background_image).decode()
    #         st.markdown(
    #             f"""
    #             <style>
    #             .stApp {{
    #                 background-image: url("data:image/jpg;base64,{background_image_base64}");
    #                 background-size: cover; /* Options: cover, contain, auto */
    #                 background-repeat: no-repeat; /* Options: no-repeat, repeat, repeat-x, repeat-y */
    #                 background-position: center; /* Options: left, right, top, bottom, center */
    #                 clip-path: ellipse(75% 75% at 50% 50%); /* Example of a shape; adjust as needed */
    #             }}
    #             </style>
    #             """,
    #             unsafe_allow_html=True
    #         )
    # except FileNotFoundError:
    #     st.warning('Background file not found.')

    # # CSS for sidebar background
    #sidebar_bg_path = 'images/sidebar.png'  # Update with your actual file path
    # try:
    #      with open(sidebar_bg_path, 'rb') as f:
    #          sidebar_bg_image = f.read()
    #          sidebar_bg_image_base64 = base64.b64encode(
    #              sidebar_bg_image).decode()
    #          st.markdown(
    #              f"""
    #              <style>
    #              [data-testid="stSidebar"] {{
    #                  background-image: url("data:image/jpg;base64,{sidebar_bg_image_base64}");
    #                  background-size: cover; /* Options: cover, contain, auto */
    #                  background-repeat: no-repeat; /* Options: no-repeat, repeat, repeat-x, repeat-y */
    #                  background-position: center; /* Options: left, right, top, bottom, center */
    #              }}
    #              </style>
    #              """,
    #              unsafe_allow_html=True
    #          )
    # except FileNotFoundError:
    #      st.warning('Sidebar background file not found.')

    st.title("🏥 Healthcare Facility Analysis in Malaysia (2020-2024)")

    # Load the data
    hospital_data, icu_data, cases_data = load_data()
    
    # Sidebar for user input
    st.sidebar.title("Settings")
    start_year = st.sidebar.slider("Select Start Year", min_value=2020, max_value=2024, value=2020)
    end_year = st.sidebar.slider("Select End Year", min_value=2020, max_value=2024, value=2024)
    state = st.sidebar.multiselect("Select State(s)", options=hospital_data['state'].unique(), default=hospital_data['state'].unique())
    
    # Preprocess the data
    hospital_data, icu_data, cases_data = preprocess_data(hospital_data, icu_data, cases_data, start_year, end_year)
    
    # Filter by state
    if state:
        hospital_data = hospital_data[hospital_data['state'].isin(state)]
        icu_data = icu_data[icu_data['state'].isin(state)]
        cases_data = cases_data[cases_data['state'].isin(state)]
    
   # Layout the graphs in a vertical manner
    st.header("Graphs")

    st.subheader("COVID-19 Beds Over Time")
    line_graph(hospital_data)
    
    st.subheader("Heatmap of Hospital and ICU Data")
    heatmap_plot(hospital_data, icu_data)

    st.subheader("COVID-19 Beds vs Admitted Patients")
    bubble_chart(hospital_data)
    
    st.subheader("Distribution of Hospitalized COVID-19 Patients")
    histogram(hospital_data)

    st.subheader("Distribution of COVID-19 Beds by State")
    pie_chart(hospital_data)

    st.subheader("Regression Analysis: New Cases vs. Hospital Admissions")
    scatter_plot(hospital_data, cases_data)
    
    st.subheader("Feature Importance for ICU Data")
    bar_chart(icu_data)

if __name__ == "__main__":
    main()

