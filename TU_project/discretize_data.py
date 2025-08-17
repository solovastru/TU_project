import pandas as pd
import numpy as np


class DataProcessor:
    def __init__(self, thresholds: dict, cause_cols: list):
        self.thresholds = thresholds
        self.cause_cols = cause_cols  

# apply the thresholds and transform numerical data into categories
    def apply_thresholds(self, series: pd.Series, thresholds: dict) -> pd.Series:
        low_t = thresholds['low_t']
        high_t = thresholds['high_t']

        def classify(val):
            if pd.isna(val):
                return np.nan
            if isinstance(val, str):
            #transform into float
                val = float(val.replace(',', '.'))
            if val < low_t:
                return 'Low'
            elif val < high_t:
                return 'Normal'
            else:
                return 'High'
        
        return series.apply(classify)

    def discretize_vars(self, df: pd.DataFrame) -> pd.DataFrame:
        # Start with only cause columns
        result = df[self.cause_cols].copy()
        
        # Add categorized numeric columns
        for column in self.thresholds:
            if column in df.columns:
                result[column] = self.apply_thresholds(df[column], self.thresholds[column])
        
        return result