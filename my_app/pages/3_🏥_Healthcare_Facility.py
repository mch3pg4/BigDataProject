import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
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


def pie_chart_patients(hospital_data):
    fig = px.pie(hospital_data, values='hosp_covid', names='state')
    st.plotly_chart(fig)
    st.write("This pie chart shows the distribution of hospitalized COVID-19 patients by state over time in the dataset.")


def pie_chart_beds(hospital_data):
    fig = px.pie(hospital_data, values='beds_covid', names='state')
    st.plotly_chart(fig)
    st.write(
        "This pie chart shows the distribution of COVID-19 beds by state over time in the dataset.")


def bubble_chart(hospital_data):
    fig = px.scatter(hospital_data, x='beds_covid', y='admitted_covid', size='hosp_covid',
                     color='state', labels={'hosp_covid': 'Hospitalized COVID-19'})
    st.plotly_chart(fig)
    st.write("This bubble chart shows the comparison between COVID-19 beds with admitted patients, colored by state and sized by hospitalized COVID-19 patients.")


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
    fig.add_trace(go.Scatter(
        x=X_test['cases_new'], y=y_test, mode='markers', name='Actual'))
    fig.add_trace(go.Scatter(x=X_test['cases_new'], y=y_pred,
                  mode='lines', name='Predicted', line=dict(dash='dash')))

    fig.update_layout(xaxis_title='New COVID-19 Cases',
                      yaxis_title='Total Hospital Admissions')

    # Display the plot in Streamlit
    st.plotly_chart(fig)
    st.write("This scatter plot shows the relationship between the number of new COVID-19 cases and the number of hospital admissions.")


def icu_availability_chart(icu_data):
    fig = px.line(icu_data, x='date', y=['beds_icu_covid', 'vent'],
                  labels={'value': 'Number Available', 'date': 'Date'})
    fig.update_layout(legend_title_text='Metrics')
    st.plotly_chart(fig)
    st.write(
        "This line chart shows the availability of ICU beds and ventilators over time.")


def monthly_icu_usage_chart(icu_data):
    # Convert the 'date' column to string format for compatibility with Plotly
    icu_data['month'] = icu_data['date'].dt.to_period('M').astype(str)

    # Aggregate data by month
    monthly_usage = icu_data.groupby('month').agg({
        'vent_covid': 'sum',
        'icu_covid': 'sum'
    }).reset_index()

    fig = px.bar(monthly_usage, x='month', y=['icu_covid', 'vent_covid'],
                 labels={'value': 'Number Used', 'month': 'Month'})
    fig.update_layout(barmode='stack')

    st.plotly_chart(fig)
    st.write(
        "This bar chart shows the total monthly usage of ICU beds and ventilators.")


def heatmap_plot(hospital_data, icu_data):
    # Merge the hospital and ICU data on 'date' and 'state'
    combined_data = pd.merge(hospital_data, icu_data, on=['date', 'state'])

    # Ensure only numeric columns are included
    numeric_columns = combined_data[[
        'beds_icu', 'vent', 'vent_used', 'icu_covid', 'icu_pui', 'icu_noncovid']].dropna()

    # Compute the correlation matrix
    correlation_matrix = numeric_columns.corr()

    # Create the heatmap
    fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto",
                    color_continuous_scale='RdBu_r')
    st.plotly_chart(fig)
    st.write("This heatmap shows the correlation between ICU metrics (beds, ventilators, usage) and patient categories (COVID, PUI, non-COVID).")


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

    # Sidebar for user control
    st.sidebar.title("Filters")
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

    # Show the map of no. of admitted patients by state
    st.subheader("Map of Hospitalized COVID-19 Patients by State")

    # Read hospitalization data
    hospital_data = pd.read_csv('filtered_datasets/hospital.csv')
    hospital_data['date'] = pd.to_datetime(hospital_data['date'])

    # Convert data from daily to monthly
    hospital_data['month'] = hospital_data['date'].dt.to_period('M')
    hosp_cases_monthly = hospital_data.groupby(['month', 'state'])[
        'admitted_total'].sum().reset_index()
    hosp_cases_monthly['month'] = hosp_cases_monthly['month'].dt.to_timestamp()

    # Read GeoJSON data for Malaysian states
    with open('my_app/map_json/malaysia.geojson') as response:
        malaysia_geojson = json.load(response)

    # Function to format date for display
    hosp_cases_monthly['month'] = hosp_cases_monthly['month'].dt.strftime(
        '%Y-%m')

    # Create a choropleth map
    fig_hosp_map = px.choropleth_mapbox(
        hosp_cases_monthly,
        geojson=malaysia_geojson,
        locations='state',
        color='admitted_total',
        color_continuous_scale="reds",
        featureidkey="properties.name",
        animation_frame='month',
        range_color=[0, hosp_cases_monthly['admitted_total'].quantile(0.98)],
        mapbox_style="carto-positron",
        zoom=4.38,
        center={"lat": 4, "lon": 109.5},
        opacity=0.75,
        labels={'admitted_total': 'Monthly Admissions'},
    )

    fig_hosp_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    # Display the map
    st.plotly_chart(fig_hosp_map)

    # show pie charts side by side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribution of COVID-19 Beds by State")
        pie_chart_beds(hospital_data)
    with col2:
        st.subheader("Distribution of Hospitalized COVID-19 Patients by State")
        pie_chart_patients(hospital_data)

    st.subheader("COVID-19 Beds vs Admitted Patients")
    bubble_chart(hospital_data)

    st.subheader("Regression Analysis: New Cases vs. Hospital Admissions")
    scatter_plot(hospital_data, cases_data)

    st.subheader("Heatmap of ICU and Ventilator Metrics Correlation")
    heatmap_plot(hospital_data, icu_data)

    st.subheader("ICU Bed and Ventilator Availability Over Time")
    icu_availability_chart(icu_data)

    st.subheader("Monthly ICU Bed and Ventilator Usage for COVID-19 Patients")
    monthly_icu_usage_chart(icu_data)

    # Show admitted vs discharged patients over time
    # Extract the year from the date
    hospital_data['year'] = hospital_data['date'].dt.year

    # Line Chart
    st.subheader('Admitted vs. Discharged Patients Over Time')

    # Filter the data based on the selected year from the sidebar slider
    filtered_data = hospital_data[(hospital_data['year'] >= start_year) & (
        hospital_data['year'] <= end_year)]

    fig_line = px.line(filtered_data, x='date', y=['admitted_total', 'discharged_total'],
                       labels={'value': 'Number of Patients', 'date': 'Date'},
                       title='Admitted vs. Discharged Patients Over Time')
    fig_line.update_layout(legend_title_text='Patient Status')

    fig_line.for_each_trace(lambda t: t.update(name={
        'admitted_total': 'Admitted Patients',
        'discharged_total': 'Discharged Patients'
    }[t.name]))

    st.plotly_chart(fig_line)

    # Machine Learning prediction to see COVID-19 cases vs admitted patients

    covid_data = pd.read_csv(
        'filtered_datasets/cases_state.csv', parse_dates=['date'])
    hospital_data = pd.read_csv(
        'filtered_datasets/hospital.csv', parse_dates=['date'])

    # Merge the data on the date column
    data = pd.merge(covid_data, hospital_data, on=['date', 'state'])

    # Sort the data by date
    data = data.sort_values(by='date')

    # List of states
    states = data['state'].unique()

    st.subheader('Predicting Admitted Patients Based on COVID-19 Cases')

    # Select a state
    state_to_plot = st.selectbox("Select State to Plot", states)
    state_data = data[data['state'] == state_to_plot]

    # Select the target variable
    target = 'admitted_covid'
    y = state_data.set_index('date')[target]

    # Split the data into training and testing sets
    train_end = '2022-06-30'
    test_start = '2022-07-01'
    test_end = '2022-07-31'
    y_train = y[:train_end]
    y_test = y[test_start:test_end]

    # Fit the ARIMA model
    model = ARIMA(y_train, order=(15, 2, 1))
    model_fit = model.fit()

    # Make predictions for the entire test period
    predictions = model_fit.predict(
        start=test_start, end=test_end, typ='levels')

    # Ensure y_test and predictions have the same index
    y_test = y_test.loc[predictions.index]

    # Combine the actual and predicted values into a single DataFrame for easier plotting
    plot_data = pd.DataFrame({
        'date': y.index,
        'actual': y,
        'predicted': np.nan
    })
    plot_data.loc[predictions.index, 'predicted'] = predictions

    # Plot the results
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_data[plot_data['date'] <= '2022-07-31']['date'], y=plot_data[plot_data['date']
                                                                                                 <= '2022-07-31']['actual'], mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=plot_data['date'], y=plot_data['predicted'],
                             mode='lines', name='Predicted', line=dict(color='red', dash='dash')))
    fig.update_layout(xaxis_title='Date', yaxis_title='Admitted COVID Patients',
                      title=f'Actual vs Predicted Admitted COVID Patients in {state_to_plot}')

    st.plotly_chart(fig)

    # Display the prediction results
    st.write('Mean Absolute Error:', np.mean(
        np.abs(predictions - y_test)))
    st.write('Mean Squared Error:', np.mean((predictions - y_test)**2))
    st.write('Root Mean Squared Error:', np.sqrt(
        np.mean((predictions - y_test)**2)))


if __name__ == "__main__":
    main()
