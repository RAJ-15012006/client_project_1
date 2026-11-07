import os
import pymysql
import pandas as pd
from sqlalchemy import create_engine
import urllib.parse

def fetch_service_data():
    host = "18.136.157.135"
    port = 3306
    user = "dm_usdata_sql"
    password = urllib.parse.quote_plus("37z<49REb&mKnl4AV!vJ")
    database = "project_service_data"
    table = "service_data"

    print(f"Connecting to MySQL database '{database}' on {host}:{port}...")
    connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(connection_string)

    query = f"SELECT * FROM {table}"
    print(f"Executing query: {query}")
    df = pd.read_sql(query, con=engine)
    print(f"Retrieved {len(df)} rows and {len(df.columns)} columns.")
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())

    os.makedirs("data/raw", exist_ok=True)
    csv_path = "data/raw/service_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Successfully saved raw dataset to {csv_path}")
    return df

if __name__ == "__main__":
    fetch_service_data()
