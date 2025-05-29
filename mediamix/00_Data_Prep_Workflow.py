import os
import shutil
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, regexp_replace, when, sum as sql_sum, lower
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType, IntegerType

# --- Configuration ---
TEMP_CSV_PATH = "/tmp/raw_mmm_data.csv"
BRONZE_TABLE_PATH = "/tmp/delta/bronze/mmm_bronze_raw_data"
BRONZE_TABLE_NAME = "mmm_bronze_raw_data_table_view" # For SQL view if needed, path is primary
SILVER_TABLE_PATH = "/tmp/delta/silver/mmm_silver_processed_data"
SILVER_TABLE_NAME = "mmm_silver_processed_data_table_view"
GOLD_TABLE_PATH = "/tmp/delta/gold/mmm_gold_final_data_from_workflow"
GOLD_TABLE_NAME = "mmm_gold_final_data_from_workflow" # Final table name for SQL access


def create_sample_raw_data(file_path):
    """Creates a sample raw CSV file for demonstration."""
    data = {
        'TransactionDate': ['2021-01-01', '2021-01-01', '2021-01-01', '2021-01-02', '2021-01-02', '2021-01-03', '01/03/2021', None, '2021-01-04'],
        'ChannelName': ['FB', 'google', 'linkedin ads', 'fb ads', 'Adwords', 'FACEBOOK', 'LinkedIn', 'UnknownChannel', 'Google'],
        'SpendAmount': ['$100.50', '$200.00', '$50.25', '€80.75', '$150', '120.00 USD', '$60', '$10', 'N/A'], # Mixed currency and text
        'SalesVolume': [500.0, 1200.0, 300.0, 450.0, 1100.0, 600.0, 350.0, 50.0, 900.0],
        'Region': ['US', 'US', 'EMEA', 'US', 'EMEA', 'US', 'APAC', 'US', 'US'],
        'OtherMetric': [1,2,3,4,5,6,7,8,9] # Will be dropped in Silver
    }
    df_pandas = pd.DataFrame(data)
    df_pandas.to_csv(file_path, index=False)
    print(f"Sample raw data saved to {file_path}")

def cleanup_temp_dirs_and_files():
    """Cleans up temporary directories and files created by the script."""
    if os.path.exists(TEMP_CSV_PATH):
        os.remove(TEMP_CSV_PATH)
        print(f"Removed temporary CSV: {TEMP_CSV_PATH}")
    
    # Remove delta table directories
    # Base path for delta tables is /tmp/delta
    delta_base_path = "/tmp/delta"
    if os.path.exists(delta_base_path):
        shutil.rmtree(delta_base_path)
        print(f"Removed temporary Delta table directory: {delta_base_path}")


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("MMM_Data_Prep_Workflow") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .master("local[*]") \
        .getOrCreate() # Ensure Delta Lake is configured

    try:
        # 0. Cleanup any previous run and Create sample data
        cleanup_temp_dirs_and_files() # Clean up before starting
        create_sample_raw_data(TEMP_CSV_PATH)

        # --- 1. Bronze Layer: Raw Ingestion ---
        print("\n--- Starting Bronze Layer Processing ---")
        raw_df = spark.read.format("csv").option("header", "true").option("inferSchema", "false").load(TEMP_CSV_PATH)
        
        print("Raw data schema:")
        raw_df.printSchema()
        print("Raw data sample:")
        raw_df.show(5, truncate=False)

        raw_df.write.format("delta").mode("overwrite").save(BRONZE_TABLE_PATH)
        # spark.sql(f"CREATE TABLE IF NOT EXISTS {BRONZE_TABLE_NAME} USING DELTA LOCATION '{BRONZE_TABLE_PATH}'") # Optional: for SQL querying by name
        print(f"Bronze data saved to Delta table: {BRONZE_TABLE_PATH}")

        # --- 2. Silver Layer: Cleaning and Transformation ---
        print("\n--- Starting Silver Layer Processing ---")
        bronze_df = spark.read.format("delta").load(BRONZE_TABLE_PATH)

        # Date parsing: Try multiple formats
        # For this example, let's assume TransactionDate can be yyyy-MM-dd or MM/dd/yyyy
        # PySpark's to_date tries to parse common formats, but coalesce with specific patterns is more robust.
        silver_df = bronze_df.withColumn(
            "parsed_date",
            to_date(col("TransactionDate"), "yyyy-MM-dd") # Primary format
        ).withColumn(
            "parsed_date_alt",
            to_date(col("TransactionDate"), "MM/dd/yyyy") # Alternative format
        ).withColumn(
            "TransactionDateClean",
            when(col("parsed_date").isNotNull(), col("parsed_date")).otherwise(col("parsed_date_alt"))
        ).drop("parsed_date", "parsed_date_alt", "TransactionDate", "OtherMetric") # Drop original and temp, and unused col

        # Clean SpendAmount: remove currency symbols and text, convert to double
        silver_df = silver_df.withColumn(
            "SpendAmountClean",
            regexp_replace(col("SpendAmount"), "[^\\d\\.]", "").cast(DoubleType()) # Remove non-digit/non-period
        )
        # Handle nulls from conversion or original data
        silver_df = silver_df.fillna(0, subset=['SpendAmountClean', 'SalesVolume'])
        silver_df = silver_df.dropna(subset=['TransactionDateClean']) # Drop rows where date couldn't be parsed

        # Standardize ChannelName
        silver_df = silver_df.withColumn("ChannelNameClean",
            when(lower(col("ChannelName")).like("%fb%") | lower(col("ChannelName")).like("%facebook%"), "Facebook")
            .when(lower(col("ChannelName")).like("%google%") | lower(col("ChannelName")).like("%adwords%"), "Adwords")
            .when(lower(col("ChannelName")).like("%linkedin%"), "LinkedIn")
            .otherwise("Other") # Categorize unknown/unmapped channels
        )
        # Filter out "Other" channels for MMM if they are not relevant
        silver_df = silver_df.filter(col("ChannelNameClean") != "Other")


        print("Silver data schema:")
        silver_df.printSchema()
        print("Silver data sample (cleaned):")
        silver_df.select("TransactionDateClean", "ChannelNameClean", "SpendAmountClean", "SalesVolume", "Region").show(10, truncate=False)
        
        silver_df.write.format("delta").mode("overwrite").save(SILVER_TABLE_PATH)
        # spark.sql(f"CREATE TABLE IF NOT EXISTS {SILVER_TABLE_NAME} USING DELTA LOCATION '{SILVER_TABLE_PATH}'")
        print(f"Silver data saved to Delta table: {SILVER_TABLE_PATH}")

        # --- 3. Gold Layer: Aggregation and Feature Engineering for MMM ---
        print("\n--- Starting Gold Layer Processing ---")
        silver_loaded_df = spark.read.format("delta").load(SILVER_TABLE_PATH)

        # Aggregate Sales per day
        daily_sales_df = silver_loaded_df.groupBy("TransactionDateClean") \
            .agg(sql_sum("SalesVolume").alias("total_sales")) \
            .withColumnRenamed("TransactionDateClean", "date")

        # Aggregate Spend per channel per day, then pivot
        daily_channel_spend_df = silver_loaded_df.groupBy("TransactionDateClean", "ChannelNameClean") \
            .agg(sql_sum("SpendAmountClean").alias("spend")) \
            .groupBy("TransactionDateClean") \
            .pivot("ChannelNameClean", ["Adwords", "Facebook", "LinkedIn"]) \
            .agg(sql_sum("spend")) \
            .withColumnRenamed("TransactionDateClean", "date")

        # Fill NaNs in pivoted spend columns with 0 (if a channel had no spend on a day)
        # Get list of channel spend columns after pivot
        spend_cols = [ch + "_spend" for ch in ["Adwords", "Facebook", "LinkedIn"]] # Expected final names
        actual_pivoted_cols = [c for c in daily_channel_spend_df.columns if c not in ["date"]] # e.g. Adwords, Facebook
        
        # Rename pivoted columns to match desired schema (e.g., "Adwords" to "adwords_spend")
        # and fill NaNs
        for channel_name in ["Adwords", "Facebook", "LinkedIn"]:
            if channel_name in daily_channel_spend_df.columns:
                daily_channel_spend_df = daily_channel_spend_df.withColumnRenamed(channel_name, f"{channel_name.lower()}_spend")
            else: # If a channel had no spend at all in the dataset, pivot won't create its column
                daily_channel_spend_df = daily_channel_spend_df.withColumn(f"{channel_name.lower()}_spend", col("date").isNotNull().cast("double") * 0.0) # Create empty col

        # Fill NaN with 0 for spend columns
        final_spend_cols = [f"{ch.lower()}_spend" for ch in ["Adwords", "Facebook", "LinkedIn"]]
        daily_channel_spend_df = daily_channel_spend_df.fillna(0, subset=final_spend_cols)


        # Join daily sales with pivoted channel spends
        gold_df = daily_sales_df.join(daily_channel_spend_df, "date", "outer") # Use outer join if some days have sales but no spend or vice-versa
        gold_df = gold_df.fillna(0, subset=final_spend_cols + ["total_sales"]) # Fill any remaining NaNs from outer join
        
        # Rename total_sales to sales
        gold_df = gold_df.withColumnRenamed("total_sales", "sales")
        
        # Select final columns in desired order
        final_gold_columns = ["date"] + [f"{ch.lower()}_spend" for ch in ["Adwords", "Facebook", "LinkedIn"]] + ["sales"]
        gold_df = gold_df.select(final_gold_columns).orderBy("date")


        print("Gold data schema:")
        gold_df.printSchema()
        print("Gold data sample (aggregated for MMM):")
        gold_df.show(10)

        gold_df.write.format("delta").mode("overwrite").save(GOLD_TABLE_PATH)
        # Create a SQL table view on top of the Delta path
        spark.sql(f"DROP TABLE IF EXISTS {GOLD_TABLE_NAME}") # Drop if exists from previous run
        spark.sql(f"CREATE TABLE {GOLD_TABLE_NAME} USING DELTA LOCATION '{GOLD_TABLE_PATH}'")
        print(f"Gold data saved to Delta table: {GOLD_TABLE_PATH} and available as SQL table {GOLD_TABLE_NAME}")
        
        # Show final result from SQL table
        print(f"\n--- Showing final data from SQL table {GOLD_TABLE_NAME} ---")
        spark.sql(f"SELECT * FROM {GOLD_TABLE_NAME} ORDER BY date").show()

    except Exception as e:
        print(f"An error occurred: {e}")
        raise
    finally:
        spark.stop()
        print("Spark session stopped.")
        # Clean up temporary files and directories at the very end
        # cleanup_temp_dirs_and_files() # Comment out if you want to inspect tables after script run
        print(f"Script finished. To inspect Delta tables, check paths under /tmp/delta/ or query the SQL table {GOLD_TABLE_NAME} if your Spark session is still active.")
        print(f"To clean up, manually run: cleanup_temp_dirs_and_files() or remove /tmp/delta and {TEMP_CSV_PATH}")

# To manually clean up after inspection if cleanup in finally is commented out:
# cleanup_temp_dirs_and_files()
