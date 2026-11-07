"""
Data Preparation and Feature Engineering Pipeline
AI Client Project 1 — Spare Parts Inventory Forecasting (NewX Services)
"""

import os
import re
import numpy as np
import pandas as pd

# Standard service centre mapping based on vehicle RTO registration / regional hubs
RTO_SC_MAP = {
    'KA53': 'SC-East (KR Puram Hub)',
    'KA03': 'SC-Central-East (Indiranagar Hub)',
    'KA01': 'SC-Central-South (Koramangala Hub)',
    'KA51': 'SC-South-East (Electronic City Hub)',
    'KA05': 'SC-South-West (Jayanagar Hub)',
    'KA04': 'SC-North (Yeshwanthpur Hub)',
    'KA02': 'SC-West (Rajajinagar Hub)',
}

# Subsystem taxonomy mapping for spare parts
SUBSYSTEM_RULES = {
    'Powertrain & Lubrication': [
        'ENGINE OIL', '3M OIL', 'OIL FILTER', 'FORK OIL', 'COOLANT', 'DISC OIL', 
        'CARBURETOR', 'PISTON', 'CYLINDER', 'GASKET', 'ENGINE'
    ],
    'Braking System': [
        'BRAKE SHOE', 'DISC PAD', 'DISC PUMP KIT', 'BRAKE CABLE', 'BRAKE LEVER',
        'DISC', 'CALIPER', 'BRAKE'
    ],
    'Transmission & Drivetrain': [
        'CHAIN SPROCKET', 'SPROCKET RUBBER', 'SPROCKET BEARING', 'CLUTCH CABLE',
        'CLUTCH ASSEMBLY', 'CLUTCH COVER GASKET', 'CLUTCH LEVER', 'CLUTCH PLATE',
        'TIMMING CHAIN', 'TIMMING CHAIN PAD', 'TENSIONER', 'DRUM BOLT', 'WHEEL RUBBER'
    ],
    'Electrical & Lighting': [
        'SPARK PLUG', 'BATTERY', 'HEAD LIGHT BULB', 'TAIL LAMP BULB', 'PARKING BULB',
        'INDICATOR', 'HORN', 'FUSE', 'SWITCH', 'WIRING', 'REGULATOR', 'STARTER'
    ],
    'Body, Controls & Accessories': [
        'TANK COVER', 'SEAT COVER', 'MIRROR', 'FOOT REST', 'HANDLE BAR', 'SIDE STAND',
        'MAIN STAND', 'LOCK', 'KEY', 'LEVER', 'GRIP', 'POLISH', 'CONSUMABLES'
    ]
}

def clean_part_name(text):
    if not isinstance(text, str):
        return 'UNKNOWN'
    text = text.upper().strip()
    text = re.sub(r'\s+', ' ', text)
    # Standardize known variations
    if 'ENGINE OIL' in text:
        return 'ENGINE OIL'
    if '3M OIL' in text:
        return '3M OIL'
    if 'AIR FILTER' in text and 'CHECK' not in text:
        return 'AIR FILTER'
    if 'BRAKE SHOE' in text:
        return 'BRAKE SHOE'
    if 'DISC PAD' in text:
        return 'DISC PAD'
    if 'DISC OIL' in text:
        return 'DISC OIL'
    if 'OIL FILTER' in text:
        return 'OIL FILTER'
    if 'SPARK PLUG' in text:
        return 'SPARK PLUG'
    if 'CHAIN SPROCKET' in text:
        return 'CHAIN SPROCKET'
    if 'CLUTCH CABLE' in text:
        return 'CLUTCH CABLE'
    if 'CLUTCH ASS' in text:
        return 'CLUTCH ASSEMBLY'
    if 'WHEEL RUBBER' in text:
        return 'WHEEL RUBBER'
    if 'SPROCKET RUBBER' in text:
        return 'SPROCKET RUBBER'
    if 'SPROCKET BEARING' in text:
        return 'SPROCKET BEARING'
    if 'COOLANT' in text:
        return 'COOLANT'
    if 'HEAD LIGHT BULB' in text or 'HEADLIGHT BULB' in text:
        return 'HEAD LIGHT BULB'
    if 'TAIL LAMP BULB' in text or 'TAILLIGHT BULB' in text:
        return 'TAIL LAMP BULB'
    return text

def classify_subsystem(part_name):
    for subsystem, keywords in SUBSYSTEM_RULES.items():
        for kw in keywords:
            if kw in part_name:
                return subsystem
    return 'Other Components'

def is_physical_part(part_name):
    # Filter out pure labor operations
    labor_keywords = [
        'LABOUR', 'LABOR', 'GENERAL SERVICE', 'CHAIN LUBRICATION', 'AIR FILTER CHECKUP',
        'CHAIN OVERHAUL', 'CLUTCH OVERHUAL', 'CLUTCH OVERHAUL', 'DISC OVERHUAL',
        'DISC OVERHAUL', 'WASHING', 'WATER WASH', 'CHECKUP', 'SERVICE CHARGE',
        'FITTING CHARGES', 'WORK DONE', 'MINOR REPAIR', 'TESTING'
    ]
    for lk in labor_keywords:
        if lk in part_name:
            return False
    return True

def prepare_raw_data(raw_csv_path='data/raw/service_data.csv'):
    df = pd.read_csv(raw_csv_path)
    
    # 1. Parse Dates
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], format='%d-%m-%y', errors='coerce')
    df['job_card_date'] = pd.to_datetime(df['job_card_date'], format='%d-%m-%y', errors='coerce')
    
    # Drop rows with invalid invoice date
    df = df.dropna(subset=['invoice_date']).copy()
    
    # 2. Extract Service Centre from Vehicle No RTO prefix
    df['rto_prefix'] = df['vehicle_no'].str.extract(r'^([A-Za-z]{2}\s*\d{1,2})')[0].str.replace(' ', '').str.upper()
    df['service_centre'] = df['rto_prefix'].map(RTO_SC_MAP).fillna('SC-Metro-Satellite Hub')
    
    # 3. Clean Part Names & classify
    df['invoice_line_text'] = df['invoice_line_text'].fillna('UNKNOWN')
    df['part_name'] = df['invoice_line_text'].apply(clean_part_name)
    df['is_part'] = df['part_name'].apply(is_physical_part)
    df['subsystem'] = df['part_name'].apply(classify_subsystem)
    
    # Filter to physical parts for spare parts inventory management
    parts_df = df[df['is_part'] & (df['part_name'] != 'UNKNOWN')].copy()
    
    # 4. Weekly timestamp (Monday of each week)
    parts_df['week_start'] = parts_df['invoice_date'].dt.to_period('W-SUN').dt.start_time
    
    # Ensure processed directory exists
    os.makedirs('data/processed', exist_ok=True)
    parts_df.to_csv('data/processed/cleaned_parts_transactions.csv', index=False)
    print(f"Cleaned transactions saved. Total parts service records: {len(parts_df):,}")
    return parts_df

def build_weekly_demand_panel(parts_df, top_n_parts=25):
    """
    Build a balanced panel at (week_start, service_centre, part_name) level.
    Zero-fills inactive periods so intermittency and true demand patterns are learned.
    """
    # Select top N most demanded spare parts
    top_parts = parts_df['part_name'].value_counts().head(top_n_parts).index.tolist()
    filtered_df = parts_df[parts_df['part_name'].isin(top_parts)].copy()
    
    # Aggregate actual weekly demand count (units consumed)
    weekly_agg = filtered_df.groupby(
        ['week_start', 'service_centre', 'part_name']
    ).agg(
        demand=('vehicle_no', 'count'),
        avg_km=('current_km_reading', 'mean'),
        unique_vehicles=('vehicle_no', 'nunique'),
        unique_models=('vehicle_model', 'nunique')
    ).reset_index()
    
    # Create complete Cartesian index: all weeks x all active service centres x top parts
    all_weeks = pd.date_range(
        start=parts_df['week_start'].min(),
        end=parts_df['week_start'].max(),
        freq='W-MON'
    )
    all_scs = parts_df['service_centre'].unique()
    
    full_idx = pd.MultiIndex.from_product(
        [all_weeks, all_scs, top_parts],
        names=['week_start', 'service_centre', 'part_name']
    ).to_frame().reset_index(drop=True)
    
    # Merge with actual demand
    panel_df = pd.merge(full_idx, weekly_agg, on=['week_start', 'service_centre', 'part_name'], how='left')
    panel_df['demand'] = panel_df['demand'].fillna(0).astype(int)
    panel_df['unique_vehicles'] = panel_df['unique_vehicles'].fillna(0).astype(int)
    panel_df['unique_models'] = panel_df['unique_models'].fillna(0).astype(int)
    panel_df['avg_km'] = panel_df.groupby(['service_centre', 'part_name'])['avg_km'].transform(
        lambda s: s.ffill().bfill().fillna(25000)
    )
    
    panel_df['subsystem'] = panel_df['part_name'].apply(classify_subsystem)
    
    # Sort chronologically by series
    panel_df = panel_df.sort_values(['service_centre', 'part_name', 'week_start']).reset_index(drop=True)
    
    panel_df.to_csv('data/processed/weekly_demand_panel.csv', index=False)
    print(f"Weekly demand panel generated: {len(panel_df):,} rows across {len(top_parts)} parts and {len(all_scs)} service centres.")
    return panel_df

if __name__ == '__main__':
    parts = prepare_raw_data()
    panel = build_weekly_demand_panel(parts, top_n_parts=30)
    print("Top 10 parts by total demand:")
    print(panel.groupby('part_name')['demand'].sum().sort_values(ascending=False).head(10))
