import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

def read_data():
    # Load the data
    vaccination_data = pd.read_csv('filtered_datasets/vax_state.csv')
    death_data = pd.read_csv('filtered_datasets/death_state.csv')
    cases_data = pd.read_csv('filtered_datasets/cases_state.csv')

    # Convert the 'date' columns to datetime
    vaccination_data['date'] = pd.to_datetime(vaccination_data['date'])
    death_data['date'] = pd.to_datetime(death_data['date'])
    cases_data['date'] = pd.to_datetime(cases_data['date'])

    return vaccination_data, death_data, cases_data

def preprocess_data(vaccination_data, death_data, cases_data, start_year, end_year, selected_states):
    # Filter the data for the selected date range and states
    start_date = pd.to_datetime(f'{start_year}-01-01')
    end_date = pd.to_datetime(f'{end_year}-12-31')

    vaccination_data = vaccination_data[(vaccination_data['date'] >= start_date) & (vaccination_data['date'] <= end_date) & (vaccination_data['state'].isin(selected_states))]
    death_data = death_data[(death_data['date'] >= start_date) & (death_data['date'] <= end_date) & (death_data['state'].isin(selected_states))]
    cases_data = cases_data[(cases_data['date'] >= start_date) & (cases_data['date'] <= end_date) & (cases_data['state'].isin(selected_states))]

    return vaccination_data, death_data, cases_data


def create_vaccination_and_death_bar_chart(vaccination_data, death_data):
    # Summarize the vaccination data for the bar chart
    vax_summary = vaccination_data[['daily_partial', 'daily_full', 'daily_booster']].sum().reset_index()
    vax_summary.columns = ['Type', 'Count']
    
    # Rename the vaccination types for better readability
    vax_type_mapping = {
        'daily_partial': 'Partial Vaccination',
        'daily_full': 'Full Vaccination',
        'daily_booster': 'Booster Vaccination'
    }
    vax_summary['Type'] = vax_summary['Type'].map(vax_type_mapping)

    # Summarize the death data for the bar chart
    death_summary = death_data[['deaths_pvax', 'deaths_fvax', 'deaths_boost']].sum().reset_index()
    death_summary.columns = ['Type', 'Count']
    
    # Rename the death types to match the vaccination types
    death_type_mapping = {
        'deaths_pvax': 'Partial Vaccination',
        'deaths_fvax': 'Full Vaccination',
        'deaths_boost': 'Booster Vaccination'
    }
    death_summary['Type'] = death_summary['Type'].map(death_type_mapping)

    # Create the bar chart with grouped bars
    fig = go.Figure()

    # Add the vaccination data
    fig.add_trace(go.Bar(
        x=vax_summary['Type'],
        y=vax_summary['Count'],
        name='Vaccination Counts',
        marker_color='blue',
        offsetgroup=1
    ))

    # Add the death data
    fig.add_trace(go.Bar(
        x=death_summary['Type'],
        y=death_summary['Count'],
        name='Death Counts',
        marker_color='red',
        offsetgroup=2,
        yaxis='y2'  # Use secondary y-axis for death counts
    ))

    # Update layout
    fig.update_layout(
        xaxis_title='Type',
        yaxis_title='Count',
        yaxis2=dict(
            title='Death Counts',
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=False
        ),
        legend=dict(x=0.7, y=1.1, orientation='h'),
        barmode='group'
    )

    return fig

def create_vaccination_line_chart(vaccination_data, cases_data):
    # Summarize the vaccination data for the line chart
    vax_summary = vaccination_data[['daily_partial', 'daily_full', 'daily_booster']].sum().reset_index()
    vax_summary.columns = ['Type', 'Vaccination Count']
    
    # Summarize the cases data for the line chart
    cases_summary = cases_data[['cases_pvax', 'cases_fvax', 'cases_boost']].sum().reset_index()
    cases_summary.columns = ['Type', 'Cases Count']
    
    # Rename the vaccination types for better readability
    vax_type_mapping = {
        'daily_partial': 'Partial Vaccination',
        'daily_full': 'Full Vaccination',
        'daily_booster': 'Booster Vaccination'
    }
    vax_summary['Type'] = vax_summary['Type'].map(vax_type_mapping)
    cases_summary['Type'] = cases_summary['Type'].map({
        'cases_pvax': 'Partial Vaccination',
        'cases_fvax': 'Full Vaccination',
        'cases_boost': 'Booster Vaccination'
    })

    # Create the line chart
    fig = go.Figure()

    # Add vaccination counts to the primary y-axis
    fig.add_trace(go.Scatter(
        x=vax_summary['Type'],
        y=vax_summary['Vaccination Count'],
        mode='lines+markers',
        name='Vaccination Counts',
        line=dict(color='blue'),
        yaxis='y1'  # Use primary y-axis for vaccination counts
    ))

    # Add cases counts to the secondary y-axis
    fig.add_trace(go.Scatter(
        x=cases_summary['Type'],
        y=cases_summary['Cases Count'],
        mode='lines+markers',
        name='Cases Counts',
        line=dict(color='red'),
        yaxis='y2'  # Use secondary y-axis for cases counts
    ))

    # Update layout
    fig.update_layout(
        xaxis_title='Type',
        legend=dict(x=0.7, y=1.1, orientation='h'),
        yaxis=dict(
            title='Vaccination Count',
            showgrid=True,
            zeroline=False,
            showline=False,
        ),
        yaxis2=dict(
            title='Cases Count',
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=False,
            showline=False,
        ),
    )

    return fig

def create_cases_and_death_stacked_area_chart(cases_data, death_data):
    # Summarize the cases data for the stacked area chart
    cases_summary = cases_data[['cases_unvax', 'cases_pvax', 'cases_fvax', 'cases_boost']].sum().reset_index()
    cases_summary.columns = ['Vaccination Type', 'Cases Count']

    # Summarize the death data for the stacked area chart
    death_summary = death_data[['deaths_unvax', 'deaths_pvax', 'deaths_fvax', 'deaths_boost']].sum().reset_index()
    death_summary.columns = ['Vaccination Type', 'Death Count']

    # Rename the vaccination types for better readability
    cases_summary['Vaccination Type'] = cases_summary['Vaccination Type'].map({
        'cases_unvax': 'Unvaccinated',
        'cases_pvax': 'Partial Vaccination',
        'cases_fvax': 'Full Vaccination',
        'cases_boost': 'Booster Vaccination'
    })

    death_summary['Vaccination Type'] = death_summary['Vaccination Type'].map({
        'deaths_unvax': 'Unvaccinated',
        'deaths_pvax': 'Partial Vaccination',
        'deaths_fvax': 'Full Vaccination',
        'deaths_boost': 'Booster Vaccination'
    })

    # Merge cases and death summaries
    summary = pd.merge(cases_summary, death_summary, on='Vaccination Type')

    # Create the stacked area chart with dual axes
    fig = go.Figure()

    # Add cases stacked area
    fig.add_trace(go.Scatter(
        x=summary['Vaccination Type'],
        y=summary['Cases Count'],
        mode='lines',
        stackgroup='cases',
        name='Cases Count',
        fill='tonexty',
        line=dict(color='blue')
    ))

    # Add deaths stacked area on secondary y-axis
    fig.add_trace(go.Scatter(
        x=summary['Vaccination Type'],
        y=summary['Death Count'],
        mode='lines',
        stackgroup='deaths',
        name='Death Count',
        fill='tonexty',
        line=dict(color='red'),
        yaxis='y2'
    ))

    # Update layout
    fig.update_layout(
        xaxis_title='Vaccination Status',
        yaxis_title='Cases Count',
        yaxis2=dict(
            title='Death Count',
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=False,
            showline=False,
        ),
        legend=dict(x=0.7, y=1.1, orientation='h'),
    )

    return fig

def create_vaccination_pie_chart(vaccination_data):
    # Calculate total vaccinations
    total_vaccinations = vaccination_data[['daily_partial', 'daily_full', 'daily_booster']].sum().sum()

    # Calculate percentages for each type
    vax_summary = vaccination_data[['daily_partial', 'daily_full', 'daily_booster']].sum()
    percentages = (vax_summary / total_vaccinations) * 100

    # Define labels and values for the pie chart
    labels = ['Partial Vaccination', 'Full Vaccination', 'Booster Vaccination']
    values = percentages.values.tolist()

    # Create the pie chart
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])

    # Update layout
    fig.update_layout(
        title='Vaccination Rates (%)',
        legend=dict(x=0.7, y=1.1, orientation='h'),
    )

    return fig

def vaccination_effectiveness_scatter_plot(vaccination_data, cases_data, target='cases_pvax'):
    # Merge vaccination and cases data
    data = pd.merge(vaccination_data, cases_data, on=['date', 'state'])

    # Prepare X (vaccination stages) and y (target variable)
    X = data[['daily_partial', 'daily_full', 'daily_booster']]
    y = data[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize the RandomForestRegressor
    model = RandomForestRegressor(random_state=42)

    # Set up a parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Perform GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Create a scatter plot of actual vs predicted values
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Actual'))
    fig.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()], mode='lines', name='Prediction', line=dict(dash='dash')))

    fig.update_layout(
        title=f"Actual vs Predicted {target.replace('_', ' ').title()}",
        xaxis_title=f"Actual {target.replace('_', ' ').title()}",
        yaxis_title=f"Predicted {target.replace('_', ' ').title()}",
    )

    st.plotly_chart(fig)
    st.write(f"This scatter plot shows the relationship between actual and predicted {target.replace('_', ' ').title()} based on vaccination stages.")


def main():
    # Add CSS to scale sidebar logo to be larger
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
    # Add a logo to the sidebar
    st.logo('images/logo_full.png', icon_image='images/logo.png')

    # Load the data
    vaccination_data, death_data, cases_data = read_data()

    # Sidebar for user input
    st.sidebar.title("Filters")
    years = list(range(2020, 2025))
    start_year = st.sidebar.selectbox("Start Year", years, index=0)
    end_year = st.sidebar.selectbox("End Year", years, index=len(years)-1)
    states = st.sidebar.multiselect("Select State(s)", options=vaccination_data['state'].unique(), default=vaccination_data['state'].unique())

    # Preprocess the data
    vaccination_data, death_data, cases_data = preprocess_data(vaccination_data, death_data, cases_data, start_year, end_year, states)

    st.title('💉Vaccination Page (2020-2024)')
    st.header("Graphs")
    st.write("The graphs shown below is relevant to the relationship between vaccination, Covid-19 cases and death cases during pandemic Covid-19 in Malaysia from January of 2020 until June of 2024.")

    # Create and display the pie chart with vaccination rates
    st.subheader("Distribution of Vaccination")
    pie_chart = create_vaccination_pie_chart(vaccination_data)
    st.plotly_chart(pie_chart)
    st.write("Explanation: The pie chart offers a snapshot of the proportions of different vaccination types, making it easy to see the distribution and coverage of partial, full, and booster vaccinations.")

    # Create and display the bar chart
    st.subheader("Vaccination vs Deaths ")
    bar_chart = create_vaccination_and_death_bar_chart(vaccination_data, death_data)
    st.plotly_chart(bar_chart)
    st.write("Explanation: The bar chart helps visualize the relationship between the number of vaccinations and the number of deaths over the given period. This can help identify trends and assess the impact of vaccination programs.")

    # Create and display the line chart
    st.subheader("Vaccination vs Covid-19 Cases")
    line_chart = create_vaccination_line_chart(vaccination_data, cases_data)
    st.plotly_chart(line_chart)
    st.write("Explanation: The line chart illustrates the trend in vaccinations and how they correlate with the number of reported cases. It is useful for analyzing the temporal effects of vaccination drives.")

    # Create and display the stacked area chart
    st.subheader("Covid-19 Cases vs Deaths")
    stacked_area_chart = create_cases_and_death_stacked_area_chart(cases_data, death_data)
    st.plotly_chart(stacked_area_chart)
    st.write("Explanation: The stacked area chart provides a clear view of the cumulative impact of the pandemic in terms of cases and deaths, helping to understand the overall trend and peaks over time.")

    # Assuming other sections of your main() function remain unchanged
    st.subheader("Vaccination Effectiveness Prediction")
    st.write("Select the target variable to predict effectiveness:")
    target_variable = st.selectbox("Select Target Variable", ['cases_pvax', 'cases_fvax', 'cases_boost'])

    vaccination_effectiveness_scatter_plot(vaccination_data, cases_data, target=target_variable)

if __name__ == "__main__":
    main()