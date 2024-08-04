#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 02:45:45 2024

@author: sanjaimariappan
"""
import streamlit as st
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import plotly.express as px
import plotly.graph_objects as go

# Define your functions
def get_buy_year(vehicle_id):
    return int(vehicle_id.split('_')[-1])

def calculate_costs(row, sell_year=2038):
    buy_year = get_buy_year(row['ID'])
    years_active = row['Year'] - buy_year + 1
    buy_cost = bcost[row['ID']]
    
    if row['Type'] == 'Buy':
        return row['Num_Vehicles'] * buy_cost, 0, 0
    elif row['Type'] == 'Sell' and row['Year'] != sell_year:
        sell_cost = row['Num_Vehicles'] * buy_cost * 0.01 * DP[years_active]
        return 0, sell_cost, 0
    elif row['Type'] == 'Use':
        operational_cost = row['Num_Vehicles'] * buy_cost * 0.01 * (I[years_active] + M[years_active])
        fuel_cost = (
            row['Num_Vehicles'] * 
            row['Distance_per_vehicle(km)'] * 
            consume[(row['ID'], row['Fuel'])] * 
            Cuff[(row['Year'], row['Fuel'])]
        )
        return 0, 0, operational_cost + fuel_cost
    return 0, 0, 0

def format_int_columns(df):
    int_columns = df.select_dtypes(include=['int64']).columns
    for col in int_columns:
        df[col] = df[col].apply(lambda x: f"{x:,}")
    return df

# File upload widgets
st.sidebar.header('Upload your CSV files')

demand_file = st.sidebar.file_uploader("Upload demand.csv", type=['csv'])
carbon_emissions_file = st.sidebar.file_uploader("Upload carbon_emissions.csv", type=['csv'])
cost_profiles_file = st.sidebar.file_uploader("Upload cost_profiles.csv", type=['csv'])
fuels_file = st.sidebar.file_uploader("Upload fuels.csv", type=['csv'])
vehicles_fuels_file = st.sidebar.file_uploader("Upload vehicles_fuels.csv", type=['csv'])
vehicles_file = st.sidebar.file_uploader("Upload vehicles.csv", type=['csv'])

# Load data, using default files if no file is uploaded
if demand_file:
    demand = pd.read_csv(demand_file)
else:
    demand = pd.read_csv('demand.csv')

if carbon_emissions_file:
    carbon_emissions = pd.read_csv(carbon_emissions_file)
else:
    carbon_emissions = pd.read_csv('carbon_emissions.csv')

if cost_profiles_file:
    cost_profiles = pd.read_csv(cost_profiles_file)
else:
    cost_profiles = pd.read_csv('cost_profiles.csv')

if fuels_file:
    fuels = pd.read_csv(fuels_file)
else:
    fuels = pd.read_csv('fuels.csv')

if vehicles_fuels_file:
    vehicles_fuels = pd.read_csv(vehicles_fuels_file)
else:
    vehicles_fuels = pd.read_csv('vehicles_fuels.csv')

if vehicles_file:
    vehicles = pd.read_csv(vehicles_file)
else:
    vehicles = pd.read_csv('vehicles.csv')

# Prepare data
distance = ["D1", "D2", "D3", "D4"]
size = ["S1", "S2", "S3", "S4"]

v = vehicles["ID"].tolist()
yr = list(range(2023, 2039))
fuel = fuels["Fuel"].unique().tolist()

bcost = vehicles.set_index('ID')['Cost ($)'].to_dict()
yrange = vehicles.set_index('ID')['Yearly range (km)'].to_dict()
dem = demand.groupby(['Year', 'Size', 'Distance'])['Demand (km)'].sum().to_dict()

consume = {(row['ID'], row['Fuel']): row['Consumption (unit_fuel/km)'] for index, row in vehicles_fuels.iterrows()}
Cuff = {(row['Year'], row['Fuel']): row['Cost ($/unit_fuel)'] for index, row in fuels.iterrows()}
Cuff_var = {(row['Year'], row['Fuel']): row['Cost Uncertainty (±%)'] for index, row in fuels.iterrows()}
I = cost_profiles.set_index('End of Year')['Insurance Cost %'].to_dict()
M = cost_profiles.set_index('End of Year')['Maintenance Cost %'].to_dict()
DP = cost_profiles.set_index('End of Year')['Resale Value %'].to_dict()
VF = vehicles_fuels.groupby('ID')['Fuel'].apply(list).to_dict()
VD = vehicles.set_index('ID')['Distance'].to_dict()
emission = fuels.groupby('Fuel')['Emissions (CO2/unit_fuel)'].mean().to_dict()
emission_goal = carbon_emissions.set_index('Year')['Carbon emission CO2/kg'].to_dict()

# Original F dictionary
F = {
    (item, int(item[-4:]), sell_year, vehicles.loc[vehicles["ID"] == item, "Size"].values[0], dis, f): (
        bcost[item] - bcost[item] * 0.01 * DP[sell_year + 1 - int(item[-4:])] + bcost[item] * 0.01 * sum(I[i] + M[i] for i in range(1, sell_year + 2 - int(item[-4:]))), 
        yrange[item]
    )
    for item in v
    for sell_year in yr
    for f in VF[item]
    for dis in distance
    if dis == VD[item]
    if sell_year >= int(item[-4:]) and sell_year <= int(item[-4:]) + 9
    if not (item.startswith("D") and int(item[-4:]) <= 2030)
}

# Add dummy nodes for sell year = 2038 with sell year as 2039 and modified costs
for item in v:
    for f in VF[item]:
        for dis in distance:
            if dis == VD[item]:
                original_sell_year = 2038
                dummy_sell_year = 2039
                if original_sell_year >= int(item[-4:]) and original_sell_year <= int(item[-4:]) + 9:
                    cost_without_sell_cost = (
                        bcost[item] - bcost[item] * 0.01 * DP[original_sell_year + 1 - int(item[-4:])] + bcost[item] * 0.01 * sum(I[i] + M[i] for i in range(1, original_sell_year + 2 - int(item[-4:])))
                    )
                    F[(item, int(item[-4:]), dummy_sell_year, vehicles.loc[vehicles["ID"] == item, "Size"].values[0], dis, f)] = (
                        cost_without_sell_cost,
                        yrange[item]
                    )

# Create Streamlit app
st.title('Fleet Optimization')

# Sidebar filters
st.sidebar.header('Filter Options')

buy_year_filter = st.sidebar.slider('Buy Year', min_value=2023, max_value=2038, value=(2023, 2038))
sell_year_filter = st.sidebar.slider('Sell Year', min_value=2023, max_value=2039, value=(2023, 2039))
size_filter = st.sidebar.multiselect('Size', options=size, default=size)
distance_filter = st.sidebar.multiselect('Distance', options=distance, default=distance)
fuel_filter = st.sidebar.multiselect('Fuel', options=fuel, default=fuel)

# Apply filters
F_df = pd.DataFrame([(key[0], key[1], key[2], key[3], key[4], key[5], value[0], value[1]) for key, value in F.items()],
                    columns=['ID', 'Buy Year', 'Sell Year', 'Size', 'Distance', 'Fuel', 'Cost ($)', 'Yearly Range (km)'])
F_df_filtered = F_df[
    (F_df['Buy Year'] >= buy_year_filter[0]) & 
    (F_df['Buy Year'] <= buy_year_filter[1]) & 
    (F_df['Sell Year'] >= sell_year_filter[0]) & 
    (F_df['Sell Year'] <= sell_year_filter[1]) & 
    (F_df['Size'].isin(size_filter)) & 
    (F_df['Distance'].isin(distance_filter)) & 
    (F_df['Fuel'].isin(fuel_filter))
]

demand_filtered = demand[
    (demand['Year'] >= buy_year_filter[0]) & 
    (demand['Year'] <= buy_year_filter[1]) & 
    (demand['Size'].isin(size_filter)) & 
    (demand['Distance'].isin(distance_filter))
]

# Format integer columns in filtered DataFrames
F_df_filtered = format_int_columns(F_df_filtered)
demand_filtered = format_int_columns(demand_filtered)



#Dynamically change the nodes
F = {
    key: value for key, value in F.items()
    if (key[1] >= buy_year_filter[0] and key[1] <= buy_year_filter[1]) and
       (key[2] >= sell_year_filter[0] and key[2] <= sell_year_filter[1]) and
       (key[3] in size_filter) and
       (key[4] in distance_filter) and
       (key[5] in fuel_filter)
}

# Apply filters to dem dictionary
dem = {
    key: value for key, value in dem.items()
    if (key[0] >= buy_year_filter[0] and key[0] <= buy_year_filter[1]) and
       (key[1] in size_filter) and
       (key[2] in distance_filter)
}



# Display filtered dataframes with formatted numbers
st.subheader('Supply Nodes')
st.write(F_df_filtered)

st.subheader('Demand Nodes')
st.write(demand_filtered)

# Display total demand over the years and carbon emissions side by side
demand_yearly = demand.groupby('Year')['Demand (km)'].sum().reset_index()
fig_demand = px.line(demand_yearly, x='Year', y='Demand (km)', title='Total Demand Over the Years')

fig_emissions = px.line(carbon_emissions, x='Year', y='Carbon emission CO2/kg', title='Carbon Emissions Over the Years')

# Display total demand over the years and carbon emissions one after the other
st.subheader('Demand and Carbon Emissions Over the Years')

st.plotly_chart(fig_demand)
st.plotly_chart(fig_emissions)

# Display fuel analysis side by side
avg_consumption = vehicles_fuels.groupby('Fuel').agg({'Consumption (unit_fuel/km)': 'mean'}).reset_index()
avg_consumption = avg_consumption.sort_values(by='Consumption (unit_fuel/km)', ascending=False)
fig_avg_consumption = px.bar(avg_consumption, x='Fuel', y='Consumption (unit_fuel/km)', title='Average Consumption per Fuel')

fuel_cost_emission = fuels.groupby('Fuel').agg({'Cost ($/unit_fuel)': 'mean', 'Emissions (CO2/unit_fuel)': 'mean'}).reset_index()
fig_cost_emission = px.scatter(fuel_cost_emission, x='Cost ($/unit_fuel)', y='Emissions (CO2/unit_fuel)', color='Fuel', title='Mean Cost vs Mean Emission')


# Display fuel analysis one after the other
st.subheader('Fuel Analysis')

st.plotly_chart(fig_avg_consumption)
st.plotly_chart(fig_cost_emission)

st.sidebar.header('Solver Parameters')

# Add time limit slider
time_limit = st.sidebar.slider('Solver Time Limit (seconds)', min_value=10, max_value=9999, value=10)

# Create Gurobi model
model = gp.Model("Fleet_optimization")

model.setParam('MIPFocus', 1)  # Emphasize feasibility
model.setParam(GRB.Param.TimeLimit, time_limit) #Set Time Limit

# Create variables
N = model.addVars(F.keys(), name="Fleet", vtype=GRB.CONTINUOUS, lb=0)
X = model.addVars([(f, d) for f in F.keys() for d in dem.keys() if f[1] <= d[0] <= f[2] and f[3] == d[1] and d[2] <= f[4]], name="arcs", vtype=GRB.INTEGER, lb=0, ub=20)
D = model.addVars([(f, d) for f in F.keys() for d in dem.keys() if f[1] <= d[0] <= f[2] and f[3] == d[1] and d[2] <= f[4]], name="Distance_travelled", vtype=GRB.CONTINUOUS, lb=0)

# Objective function
model.setObjective(
    gp.quicksum(Cuff[(d[0], f[5])] * D[f, d] * consume[(f[0], f[5])] for f, d in D.keys()) +  # Fuel cost
    gp.quicksum(N[f] * F[f][0] for f in F.keys()),  # Total cost (N * first index in value of F)
    GRB.MINIMIZE
)

# Constraint: Sum of fuel used divided by consumption rate should meet demand
for d in dem.keys():
    model.addConstr(
        gp.quicksum(D[f, d] for f in F.keys() if (f, d) in D) >= dem[d] + 2,
        name=f"FuelConsumption_{d}"
    )

# Constraint: Fuel used divided by consumption rate should be less than or equal to N times the range
for f in F.keys():
    for d in dem.keys():
        if (f, d) in D:
            model.addConstr(
                D[f,d] <= X[f, d] * F[f][1],
                name=f"FuelCapacity_{f}_{d}"
            )

# Constraint: Sum of all emissions (X * fuel_used * emission) should be less than or equal to emission goals
for year, emission_limit in emission_goal.items():
    model.addConstr(
        gp.quicksum(D[f, d] * consume[(f[0], f[5])] * emission[f[5]] for f, d in D.keys() if d[0] == year) <= emission_limit - 2,
        name=f"EmissionGoal_{year}"
    )

# Constraint: Total number of vehicles sold in a given year should be less than 20% of the active vehicles in that year
for j in range(2023, 2039):
    sell_year_sum = gp.quicksum(N[f] for f in F.keys() if f[2] == j)
    active_year_sum = gp.quicksum(N[f] for f in F.keys() if f[1] <= j <= f[2])
    model.addConstr(
        sell_year_sum <= 0.2 * active_year_sum,
        name=f"ActiveFleet_{j}"
    )

# New constraint: Total number of vehicles in each node should equal the arcs connected to demand nodes across all active years
for j in yr:
    for f in F.keys():
        if f[1] <= j <= f[2]:
            model.addConstr(
                gp.quicksum(X[f, d] for d in dem.keys() if (f, d) in X and d[0] == j) == N[f],
                name=f"NodeArcSum_{f}_{j}"
            )

# Optimize model
model.optimize()

# Extract solution values of N, X, and D
solution_N = model.getAttr('X', N)
solution_X = model.getAttr('X', X)
solution_D = model.getAttr('X', D)

# Create a list to store rows for the final dataframe
solution_data = []

# Iterate over each key in F to determine Buy, Use, and Sell actions
for f in F.keys():
    buy_year = f[1]
    sell_year = f[2]
    size = f[3]
    distance = f[4]
    fuel = f[5]
    id_ = f[0]

    # Add Buy action if the buy year is <= 2038
    if solution_N[f] > 0.5 and buy_year <= 2038:
        solution_data.append([buy_year, id_, int(round(solution_N[f])), 'Buy', "", "", 0.0])

    # Add Sell action if the sell year is <= 2038
    if solution_N[f] > 0.5 and sell_year <= 2038:
        solution_data.append([sell_year, id_, int(round(solution_N[f])), 'Sell', "", "", 0.0])

# Create DataFrame from the solution data
solution_df = pd.DataFrame(solution_data, columns=[
    'Year', 'ID', 'Num_Vehicles', 'Type', 'Fuel', 'Distance_bucket', 'Distance_per_vehicle(km)'
])

# Filter out entries beyond the year 2038
solution_df = solution_df[solution_df['Year'] <= 2038]

# Aggregate "Buy" and "Sell" actions by ID and Year
solution_df = solution_df.groupby(['Year', 'ID', 'Type']).sum().reset_index()

# Add "Use" actions for active years up to 2038
use_data = []
for f in F.keys():
    buy_year = f[1]
    sell_year = f[2]
    id_ = f[0]
    
    for year in range(buy_year, min(sell_year + 1, 2039)):
        for d in dem.keys():
            if (f, d) in solution_X and d[0] == year and solution_X[(f, d)] > 0.5:
                use_data.append([
                    year, id_, int(round(solution_X[(f, d)])), 'Use', f[5], d[2], solution_D[(f,d)] / solution_X[(f, d)]
                ])

# Create DataFrame for "Use" actions
use_df = pd.DataFrame(use_data, columns=[
    'Year', 'ID', 'Num_Vehicles', 'Type', 'Fuel', 'Distance_bucket', 'Distance_per_vehicle(km)'
])

# Round Distance_per_vehicle(km) to 2 decimal places
use_df['Distance_per_vehicle(km)'] = use_df['Distance_per_vehicle(km)'].round(2)

# Concatenate "Buy", "Sell", and "Use" DataFrames
solution_df = pd.concat([solution_df, use_df], ignore_index=True)

# Sort the DataFrame by Year and ID
solution_df = solution_df.sort_values(by=['Year', 'ID']).reset_index(drop=True)

# Round Num_Vehicles to integers
solution_df['Num_Vehicles'] = solution_df['Num_Vehicles'].astype(int)

# Sidebar filters for optimization solution
st.sidebar.header('Optimization Solution Filter Options')


year_filter = st.sidebar.slider('Year', min_value=2023, max_value=2038, value=(2023, 2038))
type_filter = st.sidebar.multiselect('Type', options=solution_df['Type'].unique().tolist(), default=solution_df['Type'].unique().tolist())
fuel_filter_solution = st.sidebar.multiselect('Fuel', options=solution_df['Fuel'].unique().tolist(), default=solution_df['Fuel'].unique().tolist())
distance_bucket_filter = st.sidebar.multiselect('Distance Bucket', options=solution_df['Distance_bucket'].unique().tolist(), default=solution_df['Distance_bucket'].unique().tolist())
size_filter = st.sidebar.multiselect('Size Bucket', options=["S1", "S2", "S3", "S4"], default=["S1", "S2", "S3", "S4"])
# Apply filters to solution_df
solution_df_filtered = solution_df[
    (solution_df['Year'] >= year_filter[0]) & 
    (solution_df['Year'] <= year_filter[1]) & 
    (solution_df['Type'].isin(type_filter)) & 
    (solution_df['Fuel'].isin(fuel_filter_solution)) & 
    (solution_df['Distance_bucket'].isin(distance_bucket_filter)) &
    (solution_df['ID'].apply(lambda x: x[-7:-5]).isin(size_filter))
]

# Display filtered optimization solution DataFrame
st.subheader('Optimization Solution')
st.write(solution_df_filtered)

# Download button to download the solution as CSV
st.download_button(
    label="Download Solution as CSV",
    data=solution_df_filtered.to_csv(index=False).encode('utf-8'),
    file_name='fleet_optimization_solution_2039.csv',
    mime='text/csv'
)

solution_df_size = solution_df[(solution_df['ID'].apply(lambda x: x[-7:-5]).isin(size_filter))]

# Calculate costs
solution_df_size[['Buying Cost', 'Selling Cost', 'Operational Cost']] = solution_df_size.apply(
    lambda row: pd.Series(calculate_costs(row)),
    axis=1
)

# Calculate total costs
total_buying_cost = solution_df_size['Buying Cost'].sum()
total_selling_cost = solution_df_size['Selling Cost'].sum()
total_operational_cost = solution_df_size['Operational Cost'].sum()

# Handling the special case for year 2038
sell_year = 2038
vehicles_to_sell = solution_df_size[(solution_df_size['Year'] == sell_year) & (solution_df_size['Type'] == 'Use')]
for index, row in vehicles_to_sell.iterrows():
    buy_year = get_buy_year(row['ID'])
    years_active = sell_year - buy_year + 1
    buy_cost = bcost[row['ID']]
    sell_cost = row['Num_Vehicles'] * buy_cost * 0.01 * DP[years_active]
    total_selling_cost += sell_cost

# Calculate total cost
total_cost = total_buying_cost - total_selling_cost + total_operational_cost

# Print the costs
st.subheader('Cost Summary')
st.write(f"Total Buying Cost: ${int(total_buying_cost)}")
st.write(f"Total Selling Cost: ${int(total_selling_cost)}")
st.write(f"Total Operational Cost: ${int(total_operational_cost)}")
st.write(f"Total Cost: ${int(total_cost)}")



# Display a line plot by summing up costs for every year
yearly_cost = solution_df_size.groupby('Year')[['Buying Cost', 'Selling Cost', 'Operational Cost']].sum().reset_index()
fig_yearly_cost = go.Figure()
fig_yearly_cost.add_trace(go.Scatter(x=yearly_cost['Year'], y=yearly_cost['Buying Cost'], mode='lines', name='Buying Cost'))
fig_yearly_cost.add_trace(go.Scatter(x=yearly_cost['Year'], y=yearly_cost['Selling Cost'], mode='lines', name='Selling Cost'))
fig_yearly_cost.add_trace(go.Scatter(x=yearly_cost['Year'], y=yearly_cost['Operational Cost'], mode='lines', name='Operational Cost'))
fig_yearly_cost.update_layout(title='Total Costs Over the Years', xaxis_title='Year', yaxis_title='Cost ($)')
st.plotly_chart(fig_yearly_cost)



# Group by Year and Fuel, and aggregate fuel types
fuel_aggregation = {
    'LNG': 'LNG',
    'BioLNG': 'LNG',
    'HVO': 'Diesel',
    'B20': 'Diesel',
    'Electricity': 'Electricity'
}

solution_df_size['Fuel_Aggregated'] = solution_df_size['Fuel'].map(fuel_aggregation)
solution_df_size['Fuel_Aggregated'].fillna(solution_df['Fuel'], inplace=True)

# Group by Year and aggregated Fuel, summing the number of vehicles used
yearly_use_aggregated = solution_df_size[solution_df_size['Type'] == 'Use'].groupby(['Year', 'Fuel_Aggregated'])['Num_Vehicles'].sum().reset_index()

# Create a stacked bar chart
fig_yearly_use_aggregated = px.bar(yearly_use_aggregated, x='Year', y='Num_Vehicles', color='Fuel_Aggregated',
                                   title='Number of Cars Used Each Year by Aggregated Fuel Type', barmode='stack')

# Calculate total number of vehicles used each year
yearly_totals = yearly_use_aggregated.groupby('Year')['Num_Vehicles'].sum().reset_index()

# Add annotations for total number of vehicles at the top of each bar
for year in yearly_totals['Year']:
    total_vehicles = yearly_totals[yearly_totals['Year'] == year]['Num_Vehicles'].values[0]
    fig_yearly_use_aggregated.add_annotation(
        x=year, y=total_vehicles, text=str(total_vehicles), showarrow=False,
        font=dict(color="white", size=12), yshift=10
    )

# Display the chart
st.plotly_chart(fig_yearly_use_aggregated)



