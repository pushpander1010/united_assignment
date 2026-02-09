import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import re

class HierarchicalImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.impute_maps = {}
        self.global_defaults = {}

    def fit(self, X, y=None):
        X_copy = X.copy()
        if 'mo' not in X_copy.columns and 'flt_departure_dt' in X_copy.columns:
            X_copy['mo'] = pd.to_datetime(X_copy['flt_departure_dt']).dt.month

        hierarchies = [
            (['flight_duration_mean', 'tz_mean'], [
                ['origin', 'destination', 'carrier', 'flt_num'], 
                ['origin', 'destination', 'carrier'], 
                ['origin', 'destination']
            ]),
            (['scaled_demand', 'scaled_share'], [
                ['origin', 'destination', 'carrier', 'mo'], 
                ['origin', 'destination', 'carrier'],
                ['origin', 'destination']
            ])
        ]

        for targets, levels in hierarchies:
            existing_targets = [t for t in targets if t in X_copy.columns]
            if not existing_targets: continue
            for level in levels:
                valid_level = [l for l in level if l in X_copy.columns]
                if not valid_level: continue
                self.impute_maps[tuple(level)] = X_copy.groupby(valid_level)[existing_targets].mean().reset_index()
        
        if 'carrier' in X_copy.columns and 'scaled_share' in X_copy.columns:
            self.impute_maps[('carrier',)] = X_copy.groupby(['carrier'])[['scaled_share']].mean().reset_index()

        for col in X_copy.columns:
            if X_copy[col].dtype.kind in 'biufc':
                val = X_copy[col].mean()
                self.global_defaults[col] = float(val) if pd.notnull(val) else 0.0
            else:
                mode_res = X_copy[col].mode()
                self.global_defaults[col] = mode_res.iloc[0] if not mode_res.empty else "missing"
        return self

    def transform(self, X):
        X = X.copy()
        for key, map_df in self.impute_maps.items():
            level = list(key)
            targets = [c for c in map_df.columns if c not in level]
            X = X.merge(map_df, on=level, how='left', suffixes=('', '_grp'))
            for col in targets:
                grp_col = f"{col}_grp"
                if grp_col in X.columns:
                    X[col] = X[col].fillna(X[grp_col])
                    X.drop(columns=[grp_col], inplace=True)
        relevant_defaults = {k: v for k, v in self.global_defaults.items() if k in X.columns}
        return X.fillna(value=relevant_defaults)


def clean_and_date_convert(df):
    df = df.copy()
    date_pattern = re.compile(r'dt|time|date', re.IGNORECASE)
    for col in [c for c in df.columns if date_pattern.search(c)]:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df.drop_duplicates()

def process_schedule_data(df):
    df = clean_and_date_convert(df)
    time_cols = ['flt_departure_local_time', 'flt_arrival_local_time', 'flt_departure_gmt']
    for col in time_cols: 
        if col in df.columns: df[col] = pd.to_datetime(df[col], errors='coerce')
    df = df.dropna(subset=[c for c in time_cols if c in df.columns])
    
    df['flight_duration'] = abs((df['flt_departure_local_time'] - df['flt_arrival_local_time']).dt.total_seconds() / 60)
    df['tz'] = (df['flt_departure_local_time'] - df['flt_departure_gmt']).dt.total_seconds() / 3600
    
    return df.groupby(['carrier', 'flt_num', 'origin', 'destination', 'flt_departure_dt']).agg(
        flight_duration_mean=('flight_duration', 'mean'),
        tz_mean=('tz', 'mean')
    ).reset_index()

def process_service_data(df):
    df = clean_and_date_convert(df)
    return df.groupby(['mo', 'origin', 'destination', 'carrier']).agg(
        scaled_demand=('scaled_demand', 'mean'), scaled_share=('scaled_share', 'mean')
    ).reset_index()

def add_features(df):
    dp_dt = pd.to_datetime(df['flt_departure_dt'])
    ob_dt= pd.to_datetime(df['observation_date'])
    #world_holidays = holidays.CountryHoliday('US',years=dp_dt.dt.year[0])
    #df['is_holiday'] = dp_dt.dt.date.isin(world_holidays)
    #df['day_of_week'] = dp_dt.dt.weekday            # 0=Mon, 6=Sun
    #df['week_of_year'] = dp_dt.dt.isocalendar().week.astype(int)
    #df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    #df['is_month_start'] = dp_dt.dt.is_month_start.astype(int)
    #df['is_month_end'] = dp_dt.dt.is_month_end.astype(int)
    df['route']=df['origin']+'_'+df['destination']
    df['carrier_freq']=df.groupby('carrier')['carrier'].transform('count')
    df['flights_per_route_day'] = (df.groupby(['route', 'flt_departure_dt'])['flt_num'].transform('count'))
    df['carrier_flights_route_day'] = (df.groupby(['carrier', 'route', 'flt_departure_dt'])['flt_num'].transform('count'))
    df['carrier_share_route_day'] = (df['carrier_flights_route_day'] /df['flights_per_route_day'].replace(0, np.nan))
    df['days_until_departure']= abs((dp_dt - ob_dt).dt.days)
    df['flight_duration_log'] = np.log1p(df['flight_duration_mean'])
    df['origin_freq'] = df.groupby('origin')['origin'].transform('count')
    df['destination_freq'] = df.groupby('destination')['destination'].transform('count')
    df['route_freq'] = df.groupby('route')['route'].transform('count')
    df['share_x_congestion'] = df['scaled_share'] * df['flights_per_route_day']
    df['share_by_demand'] = df['scaled_share'] / df['scaled_demand']
    df['count_flt_duration_mean']=df.groupby('flight_duration_mean')['carrier'].transform('count')
    return df


def prepare_data(df_fares, y_series, schedule_mapping, service_mapping, imputer=None):
    df = df_fares.copy()
    df['temp_y'] = y_series
    
    df = clean_and_date_convert(df)
    
    schedule_mapping['mo'] = schedule_mapping['flt_departure_dt'].dt.month
    master_map = schedule_mapping.merge(service_mapping, on=['mo', 'origin', 'destination', 'carrier'], how='left')
    master_map = master_map.drop_duplicates(subset=['carrier', 'flt_num', 'origin', 'destination', 'flt_departure_dt'])

    df = df.merge(master_map, on=['carrier', 'flt_num', 'origin', 'destination', 'flt_departure_dt'], how='left')

    # Impute & Features
    if imputer is None: imputer = HierarchicalImputer().fit(df)
    df = imputer.transform(df)
    df = add_features(df)

    # Re-split y and X
    y_synced = df['temp_y']
    X_final = df.drop(columns=['temp_y'])
    
    drop_cols = ['flt_num', 'flt_departure_dt', 'observation_date', 'origin_city', 'destination_city', 'route', 'origin', 'destination', 'mo']
    X_final = X_final.drop(columns=[c for c in drop_cols if c in X_final.columns])
    
    return X_final, y_synced, imputer