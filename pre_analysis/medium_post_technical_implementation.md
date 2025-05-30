# Technical Implementation: Building an MMM Solution with Databricks

Implementing Media Mix Modeling on Databricks combines data engineering, statistical modeling, and business intelligence. This section provides a practical guide to building an MMM solution using Databricks' open-source implementation as a foundation.

## Data Preparation and Integration

The first step in any MMM project is preparing the necessary data. The Databricks MMM solution uses a medallion architecture approach:

### Bronze Layer: Data Ingestion

In this layer, raw marketing and sales data is ingested from various sources with minimal transformation. The goal is to preserve the original data while making it available for processing.

```python
# Example: Ingesting marketing spend data from a CSV file
marketing_spend_df = spark.read.format("csv") \
  .option("header", "true") \
  .option("inferSchema", "true") \
  .load("/path/to/marketing_spend.csv")

# Write to bronze layer
marketing_spend_df.write.format("delta") \
  .mode("overwrite") \
  .save("/delta/bronze/marketing_spend")
```

### Silver Layer: Data Cleaning and Transformation

The silver layer contains cleaned, validated, and transformed data. This includes handling missing values, standardizing formats, and resolving inconsistencies.

```python
# Example: Cleaning and transforming marketing spend data
from pyspark.sql.functions import col, to_date

# Read from bronze layer
bronze_df = spark.read.format("delta").load("/delta/bronze/marketing_spend")

# Clean and transform
silver_df = bronze_df \
  .withColumn("date", to_date(col("date"), "yyyy-MM-dd")) \
  .withColumn("spend", col("spend").cast("double")) \
  .filter(col("spend").isNotNull()) \
  .dropDuplicates(["date", "channel"])

# Write to silver layer
silver_df.write.format("delta") \
  .mode("overwrite") \
  .save("/delta/silver/marketing_spend")
```

### Gold Layer: Feature Engineering

The gold layer contains business-level aggregates and features specifically prepared for modeling. This includes daily marketing spend by channel, sales metrics, and external factors.

```python
# Example: Creating features for MMM
from pyspark.sql.functions import sum as sql_sum

# Read from silver layer
silver_df = spark.read.format("delta").load("/delta/silver/marketing_spend")

# Aggregate to daily level by channel
gold_df = silver_df \
  .groupBy("date", "channel") \
  .agg(sql_sum("spend").alias("daily_spend"))

# Write to gold layer
gold_df.write.format("delta") \
  .mode("overwrite") \
  .partitionBy("channel") \
  .save("/delta/gold/marketing_spend_daily")
```

## Model Development Using PyMC

The Databricks MMM solution uses PyMC, a Python library for Bayesian statistical modeling. The implementation follows these key steps:

### 1. Data Preparation for Modeling

First, we convert the Spark DataFrame to a pandas DataFrame for modeling:

```python
# Load the prepared data
df = spark.table("gold_marketing_data").toPandas()

# Examine the data
print(f"Data shape: {df.shape}")
df.head()
```

### 2. Model Specification

The core of the MMM solution is the Bayesian model specification. The Databricks implementation uses PyMC to define a model that captures marketing effects:

```python
import pymc as pm
import numpy as np

def build_mmm_model(data):
    """Build a Bayesian Media Mix Model"""
    
    with pm.Model() as model:
        # Priors for baseline sales
        baseline = pm.Normal("baseline", mu=data["sales"].mean(), sigma=data["sales"].std())
        
        # Priors for marketing channel effects
        channel_effects = {}
        for channel in ["adwords", "facebook", "linkedin"]:
            # Prior for effect size
            effect = pm.HalfNormal(f"{channel}_effect", sigma=1)
            
            # Prior for adstock (decay)
            decay = pm.Beta(f"{channel}_decay", alpha=3, beta=3)
            
            # Prior for saturation
            saturation = pm.HalfNormal(f"{channel}_saturation", sigma=1)
            
            channel_effects[channel] = {
                "effect": effect,
                "decay": decay,
                "saturation": saturation
            }
        
        # Calculate expected sales
        expected_sales = baseline
        for channel in channel_effects:
            # Apply adstock transformation
            adstocked = apply_adstock(data[channel], channel_effects[channel]["decay"])
            
            # Apply saturation transformation
            saturated = apply_saturation(adstocked, channel_effects[channel]["saturation"])
            
            # Add channel contribution
            expected_sales += channel_effects[channel]["effect"] * saturated
        
        # Likelihood (observation model)
        sales = pm.Normal("sales", mu=expected_sales, sigma=pm.HalfNormal("sigma", sigma=1),
                         observed=data["sales"])
    
    return model
```

### 3. Model Training

Once the model is specified, we train it using Markov Chain Monte Carlo (MCMC) sampling:

```python
with mmm_model:
    # Use NUTS sampling for efficient exploration of parameter space
    trace = pm.sample(1000, tune=1000, chains=4, cores=4)
    
    # Examine the trace
    pm.summary(trace).round(2)
```

### 4. Model Evaluation

After training, we evaluate the model's performance:

```python
import arviz as az

# Calculate model fit metrics
with mmm_model:
    # Calculate LOO (Leave-One-Out) cross-validation
    loo = az.loo(trace, mmm_model)
    print(f"LOO: {loo}")
    
    # Calculate R-squared
    y_pred = pm.sample_posterior_predictive(trace, var_names=["sales"])
    y_pred_mean = y_pred["sales"].mean(axis=0)
    r2 = 1 - ((data["sales"] - y_pred_mean) ** 2).sum() / ((data["sales"] - data["sales"].mean()) ** 2).sum()
    print(f"R-squared: {r2:.4f}")
```

## Visualization and Interpretation of Results

The insights from MMM are only valuable if they can be clearly communicated to stakeholders. The Databricks solution includes visualization capabilities:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot channel contributions
def plot_channel_contributions(trace, data):
    # Calculate contributions for each channel
    contributions = {}
    for channel in ["adwords", "facebook", "linkedin"]:
        # Get posterior samples
        effect = trace[f"{channel}_effect"]
        decay = trace[f"{channel}_decay"]
        saturation = trace[f"{channel}_saturation"]
        
        # Calculate mean contribution
        adstocked = apply_adstock(data[channel], decay.mean())
        saturated = apply_saturation(adstocked, saturation.mean())
        contributions[channel] = effect.mean() * saturated
    
    # Plot contributions
    plt.figure(figsize=(12, 6))
    for channel, contribution in contributions.items():
        plt.plot(data.index, contribution, label=channel)
    
    plt.title("Channel Contributions to Sales")
    plt.xlabel("Date")
    plt.ylabel("Contribution")
    plt.legend()
    plt.show()

# Plot ROI by channel
def plot_channel_roi(trace, data):
    # Calculate ROI for each channel
    roi = {}
    for channel in ["adwords", "facebook", "linkedin"]:
        # Get posterior samples
        effect = trace[f"{channel}_effect"]
        decay = trace[f"{channel}_decay"]
        saturation = trace[f"{channel}_saturation"]
        
        # Calculate mean contribution
        adstocked = apply_adstock(data[channel], decay.mean())
        saturated = apply_saturation(adstocked, saturation.mean())
        contribution = effect.mean() * saturated
        
        # Calculate ROI
        roi[channel] = contribution.sum() / data[channel].sum()
    
    # Plot ROI
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(roi.keys()), y=list(roi.values()))
    plt.title("Return on Investment by Channel")
    plt.xlabel("Channel")
    plt.ylabel("ROI")
    plt.show()
```

## MLflow for Experiment Tracking and Model Management

Databricks integrates with MLflow, an open-source platform for managing the machine learning lifecycle. This integration is valuable for MMM:

```python
import mlflow
import mlflow.pyfunc

# Start an MLflow run
with mlflow.start_run(run_name="mmm_model"):
    # Log model parameters
    mlflow.log_param("model_type", "bayesian_mmm")
    mlflow.log_param("channels", ["adwords", "facebook", "linkedin"])
    
    # Log model metrics
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("loo", loo.loo)
    
    # Log channel ROIs
    for channel, value in roi.items():
        mlflow.log_metric(f"roi_{channel}", value)
    
    # Log the model
    mlflow.pyfunc.log_model("mmm_model", python_model=MMMWrapper(trace, data))
    
    # Log figures
    mlflow.log_figure(plt.gcf(), "channel_contributions.png")
    plot_channel_roi(trace, data)
    mlflow.log_figure(plt.gcf(), "channel_roi.png")
```

By tracking experiments with MLflow, teams can compare different model versions, share results, and deploy models to production.

The technical implementation described above provides a foundation for building MMM solutions on Databricks. While the specific details may vary based on an organization's data and requirements, this approach demonstrates how Databricks' unified platform can support the end-to-end MMM workflow, from data preparation to model deployment and visualization.
