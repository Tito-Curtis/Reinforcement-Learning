import pandas as pd

def rule_based_ems(df, batt_capacity_kwh=100, sc_capacity_kwh=5):
    # Initialize SOCs
    batt_soc = 0.5 * batt_capacity_kwh  # start 50%
    sc_soc = 0.5 * sc_capacity_kwh      # start 50%

    results = {
        "Battery_SOC_kWh": [],
        "SC_SOC_kWh": [],
        "Grid_Import_kW": [],
        "Unmet_Load_kW": [],
        "Cost_USD": []
    }

    for i in range(len(df)):
        demand = df["Load_Demand_kW"].iloc[i]
        renewable = df["Combined_Renewable_kW"].iloc[i]
        price = df["Electricity_Price_USDperkWh"].iloc[i]
        
        grid_used = 0
        unmet = 0

        # Case 1: renewable >= demand
        if renewable >= demand:
            surplus = renewable - demand
            # Charge battery (70%) and SC (30%)
            batt_soc = min(batt_capacity_kwh, batt_soc + surplus * 0.7)
            sc_soc = min(sc_capacity_kwh, sc_soc + surplus * 0.3)
        
        # Case 2: demand > renewable
        else:
            deficit = demand - renewable

            # Use Battery first
            if batt_soc >= deficit:
                batt_soc -= deficit
                deficit = 0
            else:
                deficit -= batt_soc
                batt_soc = 0

            # Use SC next
            if sc_soc >= deficit:
                sc_soc -= deficit
                deficit = 0
            else:
                deficit -= sc_soc
                sc_soc = 0

            # If still unmet → grid
            if deficit > 0:
                grid_used = deficit
                unmet = 0  # assume grid meets remainder fully

        # Log results
        results["Battery_SOC_kWh"].append(batt_soc)
        results["SC_SOC_kWh"].append(sc_soc)
        results["Grid_Import_kW"].append(grid_used)
        results["Unmet_Load_kW"].append(unmet)
        results["Cost_USD"].append(grid_used * price)
    results = pd.DataFrame(results)    
    
    return (results)
