import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

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
    hospital_data = hospital_data[(hospital_data['date'].dt.year >= start_year) & (
        hospital_data['date'].dt.year <= end_year)]
    icu_data = icu_data[(icu_data['date'].dt.year >= start_year) & (
        icu_data['date'].dt.year <= end_year)]
    cases_data = cases_data[(cases_data['date'].dt.year >= start_year) & (
        cases_data['date'].dt.year <= end_year)]

    return hospital_data, icu_data, cases_data

def line_graph(hospital_data):
    fig = px.line(hospital_data, x='date', y='beds_covid')
    st.plotly_chart(fig)
    st.write("This line graph shows the number of COVID-19 beds over time.")

def bubble_chart(hospital_data):
    fig = px.scatter(hospital_data, x='beds_covid', y='admitted_covid', size='hosp_covid', color='state', labels={'hosp_covid': 'Hospitalized COVID-19'})
    st.plotly_chart(fig)
    st.write("This bubble chart shows the comparison between COVID-19 beds with admitted patients, colored by state and sized by hospitalized COVID-19 patients.")

def icu_availability_chart(icu_data):
    fig = px.line(icu_data, x='date', y=['beds_icu', 'vent_covid'],
                  labels={'value': 'Number Available', 'date': 'Date'})
    fig.update_layout(legend_title_text='Metrics')
    st.plotly_chart(fig)
    st.write("This line chart shows the availability of ICU beds and ventilators over time.")

def monthly_icu_usage_chart(icu_data):
    # Convert the 'date' column to string format for compatibility with Plotly
    icu_data['month'] = icu_data['date'].dt.to_period('M').astype(str)
    
    # Aggregate data by month
    monthly_usage = icu_data.groupby('month').agg({
        'beds_icu': 'sum',
        'vent_covid': 'sum',
        'icu_covid': 'sum'
    }).reset_index()
    
    fig = px.bar(monthly_usage, x='month', y=['beds_icu', 'vent_covid'],
                 labels={'value': 'Number Used', 'month': 'Month'})
    fig.update_layout(barmode='stack')
    
    st.plotly_chart(fig)
    st.write("This bar chart shows the total monthly usage of ICU beds and ventilators.")

def heatmap_plot(hospital_data, icu_data):
    # Merge the hospital and ICU data on 'date' and 'state'
    combined_data = pd.merge(hospital_data, icu_data, on=['date', 'state'])
    
    # Ensure only numeric columns are included
    numeric_columns = combined_data[['beds_icu', 'vent', 'vent_used', 'icu_covid', 'icu_pui', 'icu_noncovid']].dropna()
    
    # Compute the correlation matrix
    correlation_matrix = numeric_columns.corr()
    
    # Create the heatmap
    fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto",
                    color_continuous_scale='RdBu_r')
    st.plotly_chart(fig)
    st.write("This heatmap shows the correlation between ICU metrics (beds, ventilators, usage) and patient categories (COVID, PUI, non-COVID).")

def scatter_plot(hospital_data, cases_data):
    hospital_cases_data = pd.merge(
        hospital_data, cases_data, on=['date', 'state'])
    hospital_cases_data = hospital_cases_data.dropna(
        subset=['cases_new', 'admitted_total'])

    X = hospital_cases_data[['cases_new']]
    y = hospital_cases_data['admitted_total']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # Set up the pipeline with scaling and regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    # Set up the parameter grid for GridSearchCV
    param_grid = {
        'regressor__fit_intercept': [True, False]
    }

    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_test['cases_new'], y=y_test, mode='markers', name='Actual'))
    fig.add_trace(go.Scatter(x=X_test['cases_new'], y=y_pred, mode='lines', name='Predicted', line=dict(dash='dash')))
    
    fig.update_layout(xaxis_title='New COVID-19 Cases',
                      yaxis_title='Total Hospital Admissions')

    # Display the plot in Streamlit
    st.plotly_chart(fig)
    st.write("This scatter plot shows the relationship between the number of new COVID-19 cases and the number of hospital admissions.")

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

    st.title("🏥 Healthcare Facility Analysis in Malaysia")

    # Load the data
    hospital_data, icu_data, cases_data = load_data()

    # Sidebar for user input
    st.sidebar.title("Controls")
    start_year = st.sidebar.slider(
        "Select Start Year", min_value=2020, max_value=2024, value=2020)
    end_year = st.sidebar.slider(
        "Select End Year", min_value=2020, max_value=2024, value=2024)
    state = st.sidebar.multiselect("Select State(s)", options=hospital_data['state'].unique(
    ), default=hospital_data['state'].unique())

    # Preprocess the data
    hospital_data, icu_data, cases_data = preprocess_data(
        hospital_data, icu_data, cases_data, start_year, end_year)

    # Filter by state
    if state:
        hospital_data = hospital_data[hospital_data['state'].isin(state)]
        icu_data = icu_data[icu_data['state'].isin(state)]
        cases_data = cases_data[cases_data['state'].isin(state)]

   # Layout the graphs in a vertical manner
    st.header("Graphs")
    st.write("These are the graphs that relevant to the healthcare facility such as the usage of hospital beds, ICU and ventilators during pandemic Covid-19 in Malaysia from March 24th, 2020 until June 1st, 2024.")
    
    st.subheader("COVID-19 Beds Over Time")
    line_graph(hospital_data) 
    
    st.subheader("COVID-19 Beds vs Admitted Patients")
    bubble_chart(hospital_data) 
    
    st.subheader("ICU Bed and Ventilator Availability Over Time")
    icu_availability_chart(icu_data)

    st.subheader("Monthly ICU Bed and Ventilator Usage")
    monthly_icu_usage_chart(icu_data)

    st.subheader("Heatmap of ICU and Ventilator Metrics Correlation")
    heatmap_plot(hospital_data, icu_data)
    
    st.subheader("Regression Analysis: New Cases vs. Hospital Admissions")
    scatter_plot(hospital_data, cases_data)
    
    # Load the hospitalization data
    hospitalization_data = pd.read_csv('filtered_datasets/hospital.csv')

    # Ensure the date column is in datetime format
    hospitalization_data['date'] = pd.to_datetime(hospitalization_data['date'])

    # Extract the year from the date
    hospitalization_data['year'] = hospitalization_data['date'].dt.year

    # Line Chart to show admitted vs. discharged patients over time
    st.subheader('Admitted vs. Discharged Patients Over Time')

    # Filter the data based on the selected year from the sidebar slider
    filtered_data = hospitalization_data[(hospitalization_data['year'] >= start_year) & (
        hospitalization_data['year'] <= end_year)]

    fig_line = px.line(filtered_data, x='date', y=['admitted_total', 'discharged_total'],
                       labels={'value': 'Number of Patients', 'date': 'Date'},
                       title='Admitted vs. Discharged Patients Over Time')
    fig_line.update_layout(legend_title_text='Patient Status')

    fig_line.for_each_trace(lambda t: t.update(name={
        'admitted_total': 'Admitted Patients',
        'discharged_total': 'Discharged Patients'
    }[t.name]))

    st.plotly_chart(fig_line)

    # Machine Learning prediction to see covid-19 cases vs utilized beds
    st.subheader('Predicting COVID-19 Cases vs. Utilized Beds')
    


if __name__ == "__main__":
    main()
