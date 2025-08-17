from discretize_data import DataProcessor
import pandas as pd

file_path = "tu_project_sensors_aug.xlsx"
df = pd.read_excel(file_path)
numeric_cols = ['Vibration', 'Pressure', 'Flow_Meter']

# Convert columns with comma decimals to floats
for col in numeric_cols:
    df[col] = df[col].astype(str).str.replace(',', '.').astype(float) 

thresholds_cat = {
    'Vibration': {'low_t': 0.1542, 'high_t': 0.1988},
    'Pressure': {'low_t': 97.9582, 'high_t': 104.3336},
    'Flow_Meter': {'low_t': 47.854, 'high_t': 52.0577}
}

cause_cols =["Normal", 'Roll_surface_damage','Hydraulic_leaks', 'Blocked_cooling_channels']

processor = DataProcessor(thresholds_cat, cause_cols)
# apply the processor to categorize the data
df = processor.discretize_vars(df)

#save the transformed data to this path
output_path = "tu_project_categorized.xlsx"

df.to_excel(output_path, index=False)