import pymc3 as pm
import numpy as np
import pandas as pd
import arviz as az
from pyspark.sql import SparkSession
from mediamix.utils import geometric_adstock, logistic_saturation # NumPy versions
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.pyfunc # For model logging

# --- Configuration ---
CHANNELS = ["adwords", "facebook", "linkedin"]
DELTA_TABLE_NAME = "mmm_gold_data" # From 01_MMM_Data_Gen.py

def load_data(spark_session, table_name):
    """Loads data from Delta table and converts to Pandas DataFrame."""
    print(f"Loading data from Delta table: {table_name}")
    spark_df = spark_session.read.format("delta").table(table_name)
    # Ensure correct column order for consistency if needed later, though dict access is robust
    # col_order = ['date'] + [f"{ch}_spend" for ch in CHANNELS] + ['sales']
    # pdf = spark_df.select(col_order).toPandas()
    pdf = spark_df.toPandas()
    pdf['date'] = pd.to_datetime(pdf['date'])
    pdf = pdf.sort_values(by='date').set_index('date')
    print("Data loaded successfully:")
    print(pdf.head())
    return pdf

def build_mmm_model(sales_data: np.ndarray, spend_data_dict: dict, channels: list):
    """
    Builds the PyMC3 Media Mix Model.

    Parameters:
    ----------
    sales_data : np.ndarray
        1D array of sales figures.
    spend_data_dict : dict
        Dictionary where keys are channel names and values are np.ndarray of spend for that channel.
    channels : list
        List of channel names.

    Returns:
    -------
    pm.Model
        The compiled PyMC3 model.
    """
    print("Building PyMC Model...")
    n_obs = len(sales_data)
    coords = {"channel": channels, "obs_id": np.arange(n_obs)}

    with pm.Model(coords=coords) as mmm_model:
        # --- Priors ---
        baseline_sales = pm.HalfNormal("baseline_sales", sigma=np.std(sales_data), testval=np.mean(sales_data)/2)

        # Channel-specific parameters
        beta_channels = pm.HalfNormal("beta_channels", sigma=1.0, dims="channel")
        decay_rate_channels = pm.Beta("decay_rate_channels", alpha=2.0, beta=2.0, dims="channel", testval=0.5)
        
        # Saturation parameters (as numpy arrays for now, to be used in pm.Deterministic)
        # These sigmas might need tuning based on typical spend/sales magnitudes
        # Using HalfNormal to keep them non-negative.
        # Max value for L could be total sales or a multiple of average sales.
        # x0 could be median spend for that channel.
        # k is steepness, might be more sensitive.

        # To make L, k, x0 part of the model and sampled, they must be PyMC distributions
        # L: maximum effect (saturation point)
        # k: steepness of the curve
        # x0: mid-point (spend level at L/2 effect)

        # For L, let's assume max possible contribution from a channel could be related to total sales variation
        # or max observed sales.
        saturation_L_channels = pm.HalfNormal("saturation_L_channels", sigma=np.max(sales_data), dims="channel", testval=np.max(sales_data)/len(channels))
        
        # For k (steepness), this is harder to set a generic prior for without domain knowledge.
        # A smaller sigma means less steepness is preferred by the prior.
        saturation_k_channels = pm.HalfNormal("saturation_k_channels", sigma=0.5, dims="channel", testval=0.01) # Assuming spend values are in hundreds/thousands

        # For x0 (mid-point), it should be related to the spend levels for that channel.
        # We need to pass channel spend data to set more informed priors or testvals for x0.
        # For now, a generic prior, but this is a key area for improvement.
        # Using a list to store x0 for each channel to use in pm.Deterministic later
        saturation_x0_list = []
        for i, channel in enumerate(channels):
            channel_spend = spend_data_dict[channel]
            median_spend = np.median(channel_spend[channel_spend > 0]) # Avoid median of zero if channel has no spend
            if median_spend <= 0: median_spend = 1.0 # Fallback if no positive spend
            sat_x0_ch = pm.HalfNormal(f"saturation_x0_{channel}", sigma=median_spend * 2, testval=median_spend, dims=None) # No dims needed if accessed by index
            saturation_x0_list.append(sat_x0_ch)


        # --- Transformations ---
        # This is where we apply adstock and saturation.
        # Using pm.Deterministic to wrap numpy functions if direct PyMC math is too complex.
        
        total_channel_contribution = pm.math.constant(np.zeros(n_obs))

        for i, channel in enumerate(channels):
            channel_spend_data = spend_data_dict[channel]

            # 1. Adstock
            # decay_rate_channels[i] is the PyMC variable for this channel's decay
            # geometric_adstock is the numpy function from utils.py
            # This will make PyMC treat this step as a "black box" for NUTS if decay_rate_channels[i] is used directly
            # It's better to use pm.math if possible, or ensure the function can handle pymc vars (needs pm.scan for loops)
            # For simplicity as requested by worker guidance, using pm.Deterministic with the numpy function:
            
            # Wrapper for geometric_adstock to be used in pm.Deterministic
            # It needs to take PyMC variables as inputs and return a NumPy array
            def adstock_wrapper(spend, decay):
                return geometric_adstock(spend, decay, L=None, normalize=False) # L can be configured
            
            adstocked_spend_channel = pm.Deterministic(
                f"adstocked_spend_{channel}",
                adstock_wrapper(channel_spend_data, decay_rate_channels[i])
            )

            # 2. Saturation
            # logistic_saturation is the numpy function from utils.py
            # We need to use pm.math for this to work with PyMC variables for L, k, x0
            # saturation_L_channels[i], saturation_k_channels[i], saturation_x0_list[i]
            
            # Using pm.math for saturation:
            # saturated_channel_effect = saturation_L_channels[i] / (1 + pm.math.exp(-saturation_k_channels[i] * (adstocked_spend_channel - saturation_x0_list[i])))
            
            # Wrapper for logistic_saturation for pm.Deterministic (if pm.math version is problematic)
            def saturation_wrapper(spend, L_sat, k_sat, x0_sat):
                 return logistic_saturation(spend, L_sat, k_sat, x0_sat)

            saturated_channel_effect = pm.Deterministic(
                f"saturated_effect_{channel}",
                saturation_wrapper(adstocked_spend_channel, saturation_L_channels[i], saturation_k_channels[i], saturation_x0_list[i])
            )
            
            # 3. Combine with Beta (Effect Size)
            total_channel_contribution += beta_channels[i] * saturated_channel_effect

        # --- Expected Sales ---
        expected_sales = pm.Deterministic("expected_sales", baseline_sales + total_channel_contribution)

        # --- Likelihood ---
        # Sigma for the likelihood
        sigma_likelihood = pm.HalfNormal("sigma_likelihood", sigma=np.std(sales_data) / 2, testval=np.std(sales_data)/10) # testval helps convergence
        
        sales_observed = pm.Normal(
            "sales_observed",
            mu=expected_sales,
            sigma=sigma_likelihood,
            observed=sales_data,
            dims="obs_id"
        )
        
        print("Model built successfully.")
        return mmm_model

def calculate_r_squared(y_true, y_pred_posterior_mean):
    """Calculates R-squared."""
    ss_res = np.sum((y_true - y_pred_posterior_mean)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

# --- Visualization Functions ---

def plot_channel_contributions(
    trace, sales_data, spend_data_dict, channels, 
    mean_beta_ch, mean_decay_ch, mean_sat_L_ch, mean_sat_k_ch, mean_sat_x0_ch, # Mean posterior params
    mean_baseline_sales, dates, model_obj # model_obj for context if needed, not strictly here
    ):
    """Plots actual sales, predicted sales, baseline, and individual channel contributions."""
    print("Generating channel contribution plot...")
    plt.figure(figsize=(15, 10))

    n_obs = len(sales_data)
    total_predicted_sales = np.full(n_obs, mean_baseline_sales)

    # Plot baseline sales contribution
    plt.plot(dates, np.full(n_obs, mean_baseline_sales), label='Baseline Sales', linestyle='--')

    # Calculate and plot each channel's contribution using mean posterior parameters
    for i, channel in enumerate(channels):
        channel_spend = spend_data_dict[channel]
        
        # Apply adstock (using mean decay rate for this channel)
        adstocked_spend = geometric_adstock(channel_spend, mean_decay_ch[i], normalize=False)
        
        # Apply saturation (using mean L, k, x0 for this channel)
        saturated_spend = logistic_saturation(adstocked_spend, mean_sat_L_ch[i], mean_sat_k_ch[i], mean_sat_x0_ch[i])
        
        # Contribution = beta * saturated_adstocked_spend
        channel_contribution = mean_beta_ch[i] * saturated_spend
        
        plt.plot(dates, channel_contribution, label=f'{channel.capitalize()} Contribution')
        total_predicted_sales += channel_contribution

    # Plot total predicted sales
    plt.plot(dates, total_predicted_sales, label='Total Predicted Sales', color='black', linewidth=2)

    # Plot actual sales
    plt.plot(dates, sales_data, label='Actual Sales', color='red', linestyle=':', alpha=0.7)

    plt.title('Sales Contributions Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    contributions_fig = plt.gcf()
    mlflow.log_figure(contributions_fig, "channel_contributions_plot.png")
    print("Channel contribution plot logged to MLflow.")
    plt.close(contributions_fig) # Close to free memory


def plot_channel_roi(
    trace, spend_data_dict, channels,
    mean_beta_ch, mean_decay_ch, mean_sat_L_ch, mean_sat_k_ch, mean_sat_x0_ch, # Mean posterior params
    model_obj # model_obj for context if needed
    ):
    """Calculates and plots ROI for each channel."""
    print("Generating channel ROI plot...")
    rois = {}
    
    for i, channel in enumerate(channels):
        channel_spend_total = np.sum(spend_data_dict[channel])
        if channel_spend_total == 0:
            rois[channel] = 0
            continue

        # Calculate total contribution for this channel using mean posterior parameters
        channel_spend_ts = spend_data_dict[channel]
        adstocked_spend = geometric_adstock(channel_spend_ts, mean_decay_ch[i], normalize=False)
        saturated_spend = logistic_saturation(adstocked_spend, mean_sat_L_ch[i], mean_sat_k_ch[i], mean_sat_x0_ch[i])
        total_channel_contribution = np.sum(mean_beta_ch[i] * saturated_spend)
        
        rois[channel] = (total_channel_contribution / channel_spend_total) if channel_spend_total > 0 else 0
        mlflow.log_metric(f"roi_{channel}", rois[channel])

    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(rois.keys()), y=list(rois.values()))
    plt.title('Channel Return on Investment (ROI)')
    plt.xlabel('Channel')
    plt.ylabel('ROI (Contribution / Spend)')
    
    roi_fig = plt.gcf()
    mlflow.log_figure(roi_fig, "channel_roi_plot.png")
    print("Channel ROI plot logged to MLflow.")
    plt.close(roi_fig)
    return rois


# --- MLflow PyFunc Model Wrapper ---
class MMMWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, trace, channels, mean_baseline, mean_betas, mean_decays, mean_sat_L, mean_sat_k, mean_sat_x0):
        self._model = model # The PyMC model object itself (optional, might not be fully serializable)
        self._trace = trace # The InferenceData object (contains posterior samples)
        self.channels = channels
        # Store mean parameters for prediction
        self.mean_baseline = mean_baseline
        self.mean_betas = dict(zip(channels, mean_betas))
        self.mean_decays = dict(zip(channels, mean_decays))
        self.mean_sat_L = dict(zip(channels, mean_sat_L))
        self.mean_sat_k = dict(zip(channels, mean_sat_k))
        self.mean_sat_x0 = dict(zip(channels, mean_sat_x0))

    def predict(self, context, model_input):
        """
        model_input: Pandas DataFrame with columns like '{channel}_spend' and a 'date' column.
        Returns predicted sales based on mean posterior parameters.
        """
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("Input must be a Pandas DataFrame.")
        
        # Ensure dates are handled if future predictions depend on time features (not in this simple model)
        # For now, we just need the spend columns.
        
        n_rows = len(model_input)
        predicted_sales = np.full(n_rows, self.mean_baseline)

        for channel in self.channels:
            spend_col = f"{channel}_spend"
            if spend_col not in model_input.columns:
                raise ValueError(f"Missing spend column for channel {channel}: {spend_col}")

            channel_spend_ts = model_input[spend_col].values
            
            adstocked_spend = geometric_adstock(channel_spend_ts, self.mean_decays[channel], normalize=False)
            saturated_spend = logistic_saturation(
                adstocked_spend, 
                self.mean_sat_L[channel], 
                self.mean_sat_k[channel], 
                self.mean_sat_x0[channel]
            )
            predicted_sales += self.mean_betas[channel] * saturated_spend
            
        return predicted_sales


# --- Main Execution ---
if __name__ == "__main__":
    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("MMM_PyMC_Model_v2") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .master("local[*]") \
        .getOrCreate()
    
    # MLflow Run
    with mlflow.start_run(run_name="MMM_PyMC_Run") as run:
        mlflow.log_param("channels", CHANNELS)
        num_draws, num_tune, num_chains = 1000, 1000, 2 # Example values
        mlflow.log_param("num_draws", num_draws)
        mlflow.log_param("num_tune", num_tune)
        mlflow.log_param("num_chains", num_chains)

        try:
            # 1. Load Data
            data_pdf = load_data(spark, DELTA_TABLE_NAME)
            sales_data = data_pdf['sales'].values
            dates = data_pdf.index # For plotting
            
            spend_data_dict = {}
            for channel in CHANNELS:
                spend_data_dict[channel] = data_pdf[f"{channel}_spend"].values

            # 2. Build Model
            mmm_model = build_mmm_model(sales_data, spend_data_dict, CHANNELS)

            # 3. Train Model (Sample)
            print("Starting PyMC sampling...")
            with mmm_model:
                trace = pm.sample(
                    draws=num_draws,
                    tune=num_tune,
                    chains=num_chains,
                    cores=1,
                    return_inferencedata=True,
                    target_accept=0.9 
                )
            print("Sampling complete.")

            # 4. Model Evaluation
            print("\n--- Model Evaluation ---")
            summary_df = az.summary(trace, hdi_prob=0.95)
            print("PyMC Model Summary (trace):\n", summary_df)
            mlflow.log_text(summary_df.to_string(), "model_summary.txt")

            with mmm_model:
                ppc = pm.sample_posterior_predictive(trace, var_names=["sales_observed", "expected_sales"], samples=500)
            
            y_pred_mean_expected = ppc["expected_sales"].mean(axis=0)
            r_squared_expected = calculate_r_squared(sales_data, y_pred_mean_expected)
            print(f"R-squared (based on expected_sales posterior mean): {r_squared_expected:.4f}")
            mlflow.log_metric("r_squared_expected", r_squared_expected)

            # (LOO calculation commented out as before)

            # 5. Extract Mean Posterior Parameters for Visualizations & PyFunc Model
            print("\nExtracting mean posterior parameters...")
            mean_baseline = trace.posterior['baseline_sales'].mean().item()
            
            # Order of dimensions in trace.posterior might be (chain, draw, channel_dim)
            # Need to ensure we get a 1D array of means per channel
            mean_beta_ch = trace.posterior['beta_channels'].mean(dim=("chain", "draw")).values
            mean_decay_ch = trace.posterior['decay_rate_channels'].mean(dim=("chain", "draw")).values
            mean_sat_L_ch = trace.posterior['saturation_L_channels'].mean(dim=("chain", "draw")).values
            mean_sat_k_ch = trace.posterior['saturation_k_channels'].mean(dim=("chain", "draw")).values
            
            # saturation_x0 was defined per channel with unique names, so access them directly
            # and then collect. Ensure the order matches CHANNELS list.
            mean_sat_x0_ch_list = []
            for ch_idx, channel_name in enumerate(CHANNELS):
                # Correctly access saturation_x0 for each channel if they were named individually
                # In the model, they were created as saturation_x0_adwords, saturation_x0_facebook etc.
                # and stored in saturation_x0_list. The trace will have them as separate variables.
                var_name = f"saturation_x0_{channel_name}"
                if var_name in trace.posterior:
                     mean_val = trace.posterior[var_name].mean().item()
                else: # Fallback if naming in model was different or if it's an array
                    # This part needs to align with how saturation_x0 was defined in build_mmm_model
                    # If saturation_x0_list was a pm.Deterministic combining individual x0s,
                    # or if saturation_x0 was a single multi-dim variable.
                    # The current build_mmm_model creates separate HalfNormal for each x0.
                    print(f"Warning: Could not find {var_name} directly in posterior. Attempting fallback for x0. This might be incorrect.")
                    # This fallback is a placeholder and likely needs adjustment based on actual trace structure for x0s.
                    # For now, assuming they are separate vars like "saturation_x0_adwords".
                    # If they were defined as a single array variable "saturation_x0_channels", then it would be:
                    # mean_sat_x0_ch = trace.posterior['saturation_x0_channels'].mean(dim=("chain", "draw")).values
                    # Given current model structure:
                    mean_val = trace.posterior[f'saturation_x0_{CHANNELS[ch_idx]}'].mean().item()


                mean_sat_x0_ch_list.append(mean_val)
            mean_sat_x0_ch = np.array(mean_sat_x0_ch_list)


            # 6. Visualizations (and log ROIs from plot_channel_roi)
            plot_channel_contributions(
                trace, sales_data, spend_data_dict, CHANNELS,
                mean_beta_ch, mean_decay_ch, mean_sat_L_ch, mean_sat_k_ch, mean_sat_x0_ch,
                mean_baseline, dates, mmm_model
            )
            
            channel_rois = plot_channel_roi(
                trace, spend_data_dict, CHANNELS,
                mean_beta_ch, mean_decay_ch, mean_sat_L_ch, mean_sat_k_ch, mean_sat_x0_ch,
                mmm_model
            )
            print("Channel ROIs:", channel_rois)

            # 7. Log PyFunc Model
            print("\nLogging PyFunc model...")
            mmm_pyfunc_model = MMMWrapper(
                model=None, trace=None, # Avoid logging potentially large model/trace objects directly in pyfunc
                channels=CHANNELS,
                mean_baseline=mean_baseline,
                mean_betas=mean_beta_ch,
                mean_decays=mean_decay_ch,
                mean_sat_L=mean_sat_L_ch,
                mean_sat_k=mean_sat_k_ch,
                mean_sat_x0=mean_sat_x0_ch
            )
            # Example input for signature
            # Create a dummy input df for signature
            input_example_data = {}
            for ch in CHANNELS:
                input_example_data[f"{ch}_spend"] = np.array([100, 150, 200], dtype=float)
            input_example_df = pd.DataFrame(input_example_data)
            
            # Infer signature
            # from mlflow.models.signature import infer_signature
            # signature = infer_signature(input_example_df, mmm_pyfunc_model.predict(None, input_example_df))

            # mlflow.pyfunc.log_model(
            #     artifact_path="mmm_model_pyfunc", 
            #     python_model=mmm_pyfunc_model,
            #     # signature=signature # Optional: define input/output schema
            #     # input_example=input_example_df # Optional: log an example input
            # )
            # Skipping log_model for now due to potential complexities with non-serializable objects if model/trace were included
            # A more robust way would be to save trace/model separately and load in wrapper if needed.
            # For this iteration, the wrapper uses mean parameters passed during init.
            # The current MMMWrapper is designed to be independent of the live model/trace object for serialization.
            mlflow.pyfunc.log_model(artifact_path="mmm_model_pyfunc", python_model=mmm_pyfunc_model)
            print("PyFunc model logged.")


            print("\n--- End of Script ---")

        except Exception as e:
            print(f"An error occurred: {e}")
            mlflow.log_param("run_status", "failed")
            mlflow.log_param("error_message", str(e))
            raise # Re-raise the exception after logging

        finally:
            spark.stop()
            print("Spark session stopped.")
