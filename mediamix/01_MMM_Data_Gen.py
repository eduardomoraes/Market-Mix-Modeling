import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType, DateType
from pyspark.sql.functions import col

# --- Configuration ---
CONFIG = {
    "date_start": "2020-01-01",
    "date_end": "2022-12-31",
    "baseline_sales": 1000,
    "noise_level": 200,
    "channels": {
        "adwords": {
            "spend_range": (50, 500),
            "decay_rate": 0.6,
            "saturation_L": 2000,
            "saturation_k": 0.005,
            "saturation_x0": 300
        },
        "facebook": {
            "spend_range": (30, 400),
            "decay_rate": 0.5,
            "saturation_L": 1500,
            "saturation_k": 0.006,
            "saturation_x0": 250
        },
        "linkedin": {
            "spend_range": (20, 300),
            "decay_rate": 0.4,
            "saturation_L": 1000,
            "saturation_k": 0.007,
            "saturation_x0": 200
        }
    },
    "delta_table_name": "mmm_gold_data"
}

# --- Helper Functions ---

def adstock_geometric(spend_series: pd.Series, decay_rate: float) -> pd.Series:
    """Applies geometric decay adstock effect."""
    adstocked_spend = np.zeros_like(spend_series, dtype=float)
    adstocked_spend[0] = spend_series[0]
    for i in range(1, len(spend_series)):
        adstocked_spend[i] = spend_series[i] + decay_rate * adstocked_spend[i-1]
    return adstocked_spend

def saturation_logistic(spend_series: pd.Series, L: float, k: float, x0: float) -> pd.Series:
    """Applies logistic saturation effect."""
    return L / (1 + np.exp(-k * (spend_series - x0)))

# --- Main Data Generation ---

def generate_mmm_data(config: dict) -> pd.DataFrame:
    """Generates synthetic MMM data."""
    dates = pd.date_range(start=config["date_start"], end=config["date_end"], freq='D')
    df = pd.DataFrame({'date': dates})

    total_saturated_adstocked_spend = pd.Series(np.zeros(len(df)), index=df.index)

    for channel_name, params in config["channels"].items():
        # Generate random spend
        spend = np.random.uniform(params["spend_range"][0], params["spend_range"][1], size=len(df))
        df[f"{channel_name}_spend"] = spend

        # Apply adstock
        adstocked_spend = adstock_geometric(df[f"{channel_name}_spend"], params["decay_rate"])
        df[f"{channel_name}_adstocked_spend"] = adstocked_spend

        # Apply saturation
        saturated_spend = saturation_logistic(adstocked_spend, params["saturation_L"], params["saturation_k"], params["saturation_x0"])
        df[f"{channel_name}_saturated_adstocked_spend"] = saturated_spend
        total_saturated_adstocked_spend += saturated_spend

    # Generate sales
    baseline_sales = config["baseline_sales"]
    noise = np.random.normal(0, config["noise_level"], size=len(df))
    df['sales'] = baseline_sales + total_saturated_adstocked_spend + noise
    df['sales'] = df['sales'].clip(lower=0) # Sales cannot be negative

    # Select final columns for the gold table
    final_columns = ['date'] + [f"{ch}_spend" for ch in config["channels"].keys()] + ['sales']
    return df[final_columns]

# --- Spark Operations ---

def save_as_delta_table(spark: SparkSession, df_pandas: pd.DataFrame, table_name: str):
    """Converts Pandas DataFrame to Spark DataFrame and saves as Delta table."""
    
    # Define schema for Spark DataFrame
    schema_fields = [StructField("date", DateType(), True)]
    for channel_name in CONFIG["channels"].keys():
        schema_fields.append(StructField(f"{channel_name}_spend", FloatType(), True))
    schema_fields.append(StructField("sales", FloatType(), True))
    spark_schema = StructType(schema_fields)

    # Convert Pandas DataFrame to Spark DataFrame
    # Ensure date column is datetime.date for Spark DateType compatibility
    df_pandas['date'] = df_pandas['date'].dt.date
    spark_df = spark.createDataFrame(df_pandas, schema=spark_schema)

    # Save as Delta table
    spark_df.write.format("delta").mode("overwrite").saveAsTable(table_name)
    print(f"DataFrame saved as Delta table: {table_name}")

# --- Execution ---
if __name__ == "__main__":
    # Generate data
    mmm_data_pandas = generate_mmm_data(CONFIG)
    print("Pandas DataFrame generated:")
    print(mmm_data_pandas.head())
    print("\nSchema of Pandas DataFrame:")
    mmm_data_pandas.info()


    # Initialize Spark Session (for local testing, in Databricks this is usually pre-configured)
    spark = SparkSession.builder \
        .appName("MMM_Data_Gen_Local") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .master("local[*]") \
        .getOrCreate()

    try:
        # Save the data
        save_as_delta_table(spark, mmm_data_pandas, CONFIG["delta_table_name"])
        
        # Verify by reading
        print(f"\nVerifying Delta table {CONFIG['delta_table_name']}:")
        delta_df = spark.read.format("delta").table(CONFIG["delta_table_name"])
        delta_df.show(5)
        print(f"Count of rows in Delta table: {delta_df.count()}")

    finally:
        spark.stop()
        print("Spark session stopped.")
