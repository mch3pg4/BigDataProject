import json
import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_image_select import image_select

# List of states
states = ["Johor", "Kedah", "Kelantan", "Melaka", "Negeri Sembilan", "Pahang",
          "Perak", "Perlis", "Pulau Pinang", "Sabah", "Sarawak", "Selangor",
          "Terengganu", "W.P. Kuala Lumpur", "W.P. Labuan", "W.P. Putrajaya"]

# function to load and prepare data


def load_and_prepare_data(file_path, date_column):
    data = pd.read_csv(file_path)
    data[date_column] = pd.to_datetime(data[date_column])
    return data


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

    # add page navigation to the sidebar
    with st.sidebar:
        st.write('What\'s on this page?')
        st.markdown('''
        - [Welcome](#ffe148da)
        - [COVID-19 at a Glance](#covid-19-at-a-glance)
        - [Vaccination Progress](#vaccination-progress)
        - [Healthcare Facilities](#healthcare-facilities)
        - [Deaths](#deaths)
        - [Select a topic](#select-a-topic)
        ''')

    # dashboard title and description
    st.title('Welcome to MED ANALYTICS. 📊')
    st.write('''The MED ANALYTICS dashboard empowers users to navigate the challenges of 
             COVID-19 through data-driven insights. This platform offers data visualizations 
             on healthcare facilities, mental health, and vaccination efforts, providing a 
             comprehensive picture of the pandemic\'s impact. By leveraging machine learning,
              MED ANALYTICS aims to predict future trends and resource needs, supporting proactive decision-making.''')

    # COVID-19 at a Glance
    st.header('COVID-19 at a Glance')
    st.caption('(as of 1st June 2024)')

    # load and prepare all datasets
    cases_data = load_and_prepare_data(
        'filtered_datasets/cases_state.csv', 'date')
    death_data = load_and_prepare_data(
        'filtered_datasets/death_state.csv', 'date')
    hospital_data = load_and_prepare_data(
        'filtered_datasets/hospital.csv', 'date')
    icu_data = load_and_prepare_data('filtered_datasets/icu.csv', 'date')
    vax_data = load_and_prepare_data('filtered_datasets/vax_state.csv', 'date')

    # Three key metrics: Total cases, total deaths, and total recoveries
    # Calculate total cases, deaths, recoveries and vaccinations
    total_cases = cases_data['cases_new'].sum()
    total_deaths = death_data[[
        'deaths_unvax', 'deaths_pvax', 'deaths_fvax']].sum().sum()
    total_recoveries = hospital_data['discharged_total'].sum()
    total_vaccinations = vax_data['daily_full'].sum()

    # Display the metrics in a row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label='Total Cases', value=f"{total_cases:,}")
    with col2:
        st.metric(label='Total Deaths', value=f"{total_deaths:,}")
    with col3:
        st.metric(label='Total Recoveries', value=f"{total_recoveries:,}")
    with col4:
        st.metric(label='Total Vaccinations (2 doses)',
                  value=f"{total_vaccinations:,}")

    st.subheader('COVID-19 Cases in Malaysia ')
    st.caption('(as of 1st June 2024)')

    st.write('''The following visualizations provide insights into the COVID-19 situation in Malaysia,
                including the distribution of cases over time and a map showing the monthly distribution of cases by state.
                Additionally, there are pie charts to illustrate the distribution of cases by age group and state.''')

    # line graph of covid cases
    # Group by date, sum cases, reset index
    total_cases_by_date = cases_data.groupby(
        'date')['cases_new'].sum().reset_index()

    # line chart for total cases
    fig_total_cases = px.line(total_cases_by_date, x='date', y='cases_new',
                              title='COVID-19 Cases Over Time in Malaysia')
    fig_total_cases.update_layout(
        xaxis_title='Date', yaxis_title='New Cases')

    st.plotly_chart(fig_total_cases)

    # show the map of malaysia with covid cases
    # Convert data from daily to monthly
    cases_data['month'] = cases_data['date'].dt.to_period('M')
    covid_cases_monthly = cases_data.groupby(['month', 'state'])[
        'cases_new'].sum().reset_index()
    covid_cases_monthly['month'] = covid_cases_monthly['month'].dt.to_timestamp()

    # Read GeoJSON data for Malaysian states
    with open('my_app/map_json/malaysia.geojson') as response:
        malaysia_geojson = json.load(response)

    # Function to format date for display
    covid_cases_monthly['month'] = covid_cases_monthly['month'].dt.strftime(
        '%Y-%m')

    # Create a choropleth map
    fig_covid_map = px.choropleth_mapbox(
        covid_cases_monthly,
        geojson=malaysia_geojson,
        locations='state',
        color='cases_new',
        color_continuous_scale="reds",
        featureidkey="properties.name",
        animation_frame='month',
        range_color=[0, covid_cases_monthly['cases_new'].quantile(0.98)],
        mapbox_style="carto-positron",
        zoom=4.38,
        center={"lat": 4, "lon": 109.5},
        opacity=0.75,
        labels={'cases_new': 'New Cases'},
    )

    fig_covid_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    # Display the map
    st.plotly_chart(fig_covid_map)

    # Pie chart of covid cases by age group and state
    # Load the COVID-19 cases data
    # Calculate the total cases for each age group
    total_cases = {
        'Child': cases_data['cases_child'].sum(),
        'Adolescent': cases_data['cases_adolescent'].sum(),
        'Adult': cases_data['cases_adult'].sum(),
        'Elderly': cases_data['cases_elderly'].sum()
    }

    # Create a DataFrame for the pie chart
    cases_df = pd.DataFrame(list(total_cases.items()), columns=[
                            'Age Group', 'Total Cases'])

    # Create the pie chart for age group
    fig_pie_age_group = px.pie(cases_df, values='Total Cases', names='Age Group',
                               title='Overall COVID-19 Cases by Age Group',
                               color_discrete_sequence=px.colors.qualitative.Set3)
    fig_pie_age_group.update_layout(legend_title_text='Age Group')

    # Filter the data to include only the specified states
    filtered_cases_data = cases_data[cases_data['state'].isin(states)]

    # Calculate the total cases for each state
    total_cases_by_state = filtered_cases_data.groupby(
        'state')['cases_new'].sum().reset_index()

    # Create the pie chart for states
    fig_pie_state = px.pie(total_cases_by_state, values='cases_new', names='state',
                           title='Overall COVID-19 Cases by State',
                           color_discrete_sequence=px.colors.qualitative.Set3,
                           labels={'state': 'State', 'cases_new': 'Total Cases'})
    fig_pie_state.update_layout(legend_title_text='State')

    # Create columns to place the charts side by side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Distribution of COVID-19 Cases by Age Group')
        st.caption('(as of 1st June 2024)')
        st.plotly_chart(fig_pie_age_group, use_container_width=True)

    with col2:
        st.subheader('Distribution of COVID-19 Cases by State')
        st.caption('(as of 1st June 2024)')
        st.plotly_chart(fig_pie_state, use_container_width=True)

    # Vaccination Progress
    st.header('Vaccination Progress')
    st.subheader('Vaccination Progress in Malaysia')
    st.caption('(as of 1st June 2024)')

    st.write('''The following visualizations provide insights into the COVID-19 vaccination progress 
                in Malaysia throughout the pandemic.
                Shown below are a line chart illustrating the vaccination progress over time,
                and a stacked bar chart showing the cumulative vaccinations by state.''')

    # Line chart of vaccination progress over time
    fig_line_vax = px.line(vax_data, x='date', y=['daily_partial', 'daily_full', 'daily_booster'],
                           labels={'value': 'Number of Doses', 'date': 'Date'},
                           title='COVID-19 Vaccination Progress Over Time')

    # Update the legend and layout
    fig_line_vax.update_layout(legend_title_text='Dose Type')
    fig_line_vax.for_each_trace(lambda t: t.update(name={
        'daily_partial': '1st Dose',
        'daily_full': '2nd Dose',
        'daily_booster': 'Booster Dose'
    }[t.name]))

    # Bar chart of vaccination progress by state
    # Filter the data to include only states
    filtered_vaccination_data = vax_data[vax_data['state'].isin(
        states)]

    # Group by state and get the maximum value for the cumulative vaccination columns
    state_vaccination_totals = filtered_vaccination_data.groupby(
        'state')[['cumul_partial', 'cumul_full', 'cumul_booster', 'cumul_booster2']].max().reset_index()

    # Melt the DataFrame for Plotly
    vaccination_melted = state_vaccination_totals.melt(id_vars='state', value_vars=['cumul_partial', 'cumul_full', 'cumul_booster', 'cumul_booster2'],
                                                       var_name='Vaccination Type', value_name='Total Vaccinations')

    # Create the stacked bar chart
    fig_bar_vax = px.bar(vaccination_melted, x='state', y='Total Vaccinations', color='Vaccination Type', barmode='stack',
                         title='Cumulative Vaccinations by State',
                         labels={'state': 'State', 'Total Vaccinations': 'Total Vaccinations'})

    fig_bar_vax.for_each_trace(lambda t: t.update(name={
        'cumul_partial': '1st Dose',
        'cumul_full': '2nd Dose',
        'cumul_booster': 'Booster Dose',
        'cumul_booster2': '2nd Booster Dose'
    }[t.name]))

    # Display the charts side by side
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig_line_vax, use_container_width=False)

    with col2:
        st.plotly_chart(fig_bar_vax, use_container_width=False)

    # Healthcare Facilities
    st.header('Healthcare Facilities')
    st.subheader('Healthcare Facilities in Malaysia')
    st.caption('(as of 1st June 2024)')

    st.write('''The following visualizations provide insights into the healthcare facilities in Malaysia,
                including hospitalization trends over time, which include the metrics of admitted and discharged patients,
                and the distribution of ICU beds and ventilators used by state.''')

    # Filter the data to include only the specified states
    filtered_hospitalization_data = hospital_data[hospital_data['state'].isin(
        states)]

    # line chart to show hospitalization trends over time
    fig_line_hosp = px.line(filtered_hospitalization_data, x='date', y=['admitted_covid', 'admitted_pui',
                                                                        'admitted_total', 'discharged_covid',
                                                                        'discharged_pui', 'discharged_total'],
                            labels={'value': 'Number of Patients',
                                    'date': 'Date'},
                            title='Hospitalization Trends Over Time',
                            )

    fig_line_hosp.update_layout(legend_title_text='Patient Type')
    fig_line_hosp.for_each_trace(lambda t: t.update(name={
        'admitted_covid': 'Admitted COVID-19 Patients',
        'admitted_pui': 'Admitted PUI Patients',
        'admitted_total': 'Total Admitted Patients',
        'discharged_covid': 'Discharged COVID-19 Patients',
        'discharged_pui': 'Discharged PUI Patients',
        'discharged_total': 'Total Discharged Patients'
    }[t.name]))

    # stacked bar chart to show distribution of beds by state
    state_beds_totals = filtered_hospitalization_data.groupby(
        'state')[['beds', 'beds_covid', 'beds_noncrit']].max().reset_index()
    beds_melted = state_beds_totals.melt(id_vars='state', value_vars=['beds', 'beds_covid', 'beds_noncrit'],
                                         var_name='Bed Type', value_name='Total Beds')
    fig_bar_hosp = px.bar(beds_melted, x='state', y='Total Beds', color='Bed Type', barmode='stack',
                          title='Distribution of Beds by State',
                          labels={'state': 'State', 'Total Beds': 'Total Beds'})

    fig_bar_hosp.for_each_trace(lambda t: t.update(name={
        'beds': 'Hosp. Beds',
        'beds_covid': 'Beds for Covid-19 Patients',
        'beds_noncrit': 'Beds for Non-critical Patients'
    }[t.name]))

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_line_hosp)
    with col2:
        st.plotly_chart(fig_bar_hosp)

    # ICU Facilities in Malaysia
    st.subheader('ICU Facilities in Malaysia')
    st.caption('(as of 1st June 2024)')

    st.write('''The following visualizations provide insights into the ICU facilities in Malaysia,
                including the distribution of ICU beds and ventilators used by state,
                and a pie chart illustrating the proportion of ICU beds and ventilators used over time.''')

    # stacked bar chart to show distribution of ICU beds and ventilators by state
    st.subheader('Distribution of ICU Beds and Ventilators Used')
    icu_data['state'] = icu_data['state'].astype(str)

    # Group by state and sum the columns
    icu_data_grouped = icu_data.groupby('state')[
        ['icu_covid', 'icu_noncovid', 'vent_covid', 'vent_noncovid']].sum().reset_index()

    fig_stacked_bar_icu = px.bar(icu_data_grouped, x='state', y=['icu_covid', 'icu_noncovid', 'vent_covid', 'vent_noncovid'],
                                 labels={'value': 'Count', 'state': 'State'},
                                 title='Distribution of ICU Beds and Ventilators Used by State',
                                 barmode='stack')

    fig_stacked_bar_icu.update_layout(legend_title_text='Resource Type')
    fig_stacked_bar_icu.for_each_trace(lambda t: t.update(name={
        'icu_covid': 'ICU for COVID-19',
        'icu_noncovid': 'ICU for Non-COVID-19',
        'vent_covid': 'Ventilators for COVID-19',
        'vent_noncovid': 'Ventilators for Non-COVID-19'
    }[t.name]))

    # Melt the data for ICU and ventilator usage
    icu_melted = icu_data.melt(id_vars='date', value_vars=['icu_covid', 'icu_noncovid', 'vent_covid', 'vent_noncovid'],
                               var_name='Resource Type', value_name='Count')

    # Function to categorize resource types
    def categorize_resource(resource):
        if resource == 'icu_covid':
            return 'ICU for COVID-19'
        elif resource == 'icu_noncovid':
            return 'ICU for Non-COVID-19'
        elif resource == 'vent_covid':
            return 'Ventilators for COVID-19'
        elif resource == 'vent_noncovid':
            return 'Ventilators for Non-COVID-19'

    icu_melted['Resource Type'] = icu_melted['Resource Type'].apply(
        categorize_resource)

    # Create pie chart
    fig_pie_icu = px.pie(icu_melted, values='Count', names='Resource Type',
                         title='Proportion of ICU Beds and Ventilators Used Over Time',
                         color_discrete_sequence=px.colors.qualitative.Set3,
                         labels={'Count': 'Number of Resources'})
    fig_pie_icu.update_layout(legend_title_text='Resource Type')

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_stacked_bar_icu)
    with col2:
        st.plotly_chart(fig_pie_icu)

    # Deaths
    st.header('Deaths')
    st.subheader('Deaths in Malaysia during COVID-19')
    st.caption('(as of 1st June 2024)')

    st.write('''The following visualizations provide insights into the COVID-19 deaths in Malaysia,
                including the distribution of deaths by vaccination status and state.''')

    # pie chart to show distribution of deaths by vaccination status
    # Calculate the total deaths for each vaccination status
    total_deaths = {
        'Unvaccinated': death_data['deaths_unvax'].sum(),
        'Partially Vaccinated': death_data['deaths_pvax'].sum(),
        'Fully Vaccinated': death_data['deaths_fvax'].sum()
    }

    # Create a DataFrame for the pie chart
    deaths_df = pd.DataFrame(list(total_deaths.items()), columns=[
        'Vaccination Status', 'Total Deaths'])

    # Create the pie chart for deaths by vaccination status
    fig_pie_deaths = px.pie(deaths_df, values='Total Deaths', names='Vaccination Status',
                            title='COVID-19 Deaths by Vaccination Status',
                            color_discrete_sequence=px.colors.sequential.RdBu)
    fig_pie_deaths.update_layout(legend_title_text='Vaccination Status')

    # covid deaths by state
    # Filter the data to include only the specified states
    filtered_death_data = death_data[death_data['state'].isin(states)]

    # Calculate the total deaths for each state
    filtered_death_data['deaths_total'] = filtered_death_data[[
        'deaths_unvax', 'deaths_pvax', 'deaths_fvax', 'deaths_boost']].sum(axis=1)
    total_deaths_by_state = filtered_death_data.groupby(
        'state')['deaths_total'].sum().reset_index()

    # Create the pie chart for states
    fig_pie_deaths_state = px.pie(total_deaths_by_state, values='deaths_total', names='state',
                                  title='COVID-19 Deaths by State',
                                  color_discrete_sequence=px.colors.qualitative.Set3,
                                  labels={'state': 'State', 'deaths_total': 'Total Deaths'})
    fig_pie_deaths_state.update_layout(legend_title_text='State')

    # display pie charts side by side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Distribution of COVID-19 Deaths by Vaccination Status')
        st.plotly_chart(fig_pie_deaths, use_container_width=True)

    with col2:
        st.subheader('Distribution of COVID-19 Deaths by State')
        st.plotly_chart(fig_pie_deaths_state, use_container_width=True)

    # Select topic buttons
    st.header('Select a topic')
    img = image_select(
        label="Selecting a topic brings you to the topic's respective page to view more of the data and insights.",
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
