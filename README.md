# Media Mix Modeling (MMM) with Databricks

Welcome to the Media Mix Modeling (MMM) with Databricks repository! This repository is dedicated to providing a comprehensive understanding of Media Mix Modeling, its implementation, operationalization, and its significant business value when leveraged with Databricks.

## Purpose

The primary purpose of this repository is to serve as a central hub for information related to Media Mix Modeling. We aim to equip users with the knowledge and resources necessary to:

*   Understand the core concepts and benefits of MMM.
*   Implement MMM solutions effectively using Databricks and PyMC.
*   Operationalize MMM to drive continuous marketing optimization.
*   Showcase the tangible business value derived from MMM insights.

## Key Components & Content

This repository hosts a variety of documents to cater to different informational needs:

*   **Business Value Research:** Explore the strategic advantages and ROI of implementing MMM.
*   **Technical Implementation Details:** Dive deep into the technical aspects of building MMM models, specifically utilizing Databricks and the PyMC library. Find detailed guides and code examples.
*   **Operationalization Guides:** Learn how to integrate MMM into your ongoing marketing processes and decision-making workflows.
*   **Industry Case Studies:** Discover real-world examples of how MMM has been successfully applied in various industries.
*   **Drafts for Medium Posts:** Access pre-written and in-progress articles intended for broader dissemination on platforms like Medium, covering various facets of MMM.

## How to Navigate This Repository

We recommend the following starting points based on your role and interests:

*   **For Business Executives and Managers:**
    *   Begin with "Operationalizing Media Mix Modeling with Databricks: A Guide for Business Executives" to understand the strategic implications and how to derive business value.
    *   Explore the "Business Value Research" documents to further understand the ROI and benefits.
*   **For Data Scientists, Analysts, and Technical Users:**
    *   "Technical Analysis of Databricks Media Mix Modeling Solution.md" provides an in-depth look at the architecture and methodologies.
    *   Refer to the "Technical Implementation Details" for practical guidance and code.
*   **For Marketing Professionals:**
    *   The "Industry Case Studies" can provide inspiration and practical examples of MMM in action.
    *   "Drafts for Medium Posts" offer accessible insights into various MMM topics.

## Our Technical Solution

The documents and guides within this repository often refer to our powerful and flexible Media Mix Modeling solution built on the **Databricks** platform, leveraging the capabilities of the **PyMC** library for Bayesian modeling. This combination allows for scalable, robust, and insightful MMM implementations.

## Setup Instructions

### Prerequisites
*   The solution is primarily designed for a **Databricks environment**. However, it can also be run in a local environment with Apache Spark, Python, and Delta Lake correctly configured.
*   Python 3.x (tested with 3.8+).
*   Access to an active Spark session.
*   Delta Lake library installed and configured for your Spark session.

### Cloning the Repository
1.  Clone the repository to your local machine or Databricks environment:
    ```bash
    git clone <repository_url> 
    ```
    (Replace `<repository_url>` with the actual URL of this repository)
2.  Navigate to the repository directory:
    ```bash
    cd <repository_directory_name>
    ```

### Environment Setup (Recommended for Local Development)
1.  It's highly recommended to create and activate a Python virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Solution

The repository contains Python scripts to demonstrate data preparation and Media Mix Modeling. Here's how to run them:

### 1. `mediamix/00_Data_Prep_Workflow.py` (Optional Full ETL Demonstration)
*   **Purpose**: This script demonstrates an end-to-end data preparation pipeline, starting from simulated raw/messy data (CSV) and processing it through Bronze, Silver, and Gold layers using PySpark and Delta Lake.
*   **How to run**:
    ```bash
    python mediamix/00_Data_Prep_Workflow.py
    ```
    (If running on a Spark cluster and the script is packaged, you might use `spark-submit mediamix/00_Data_Prep_Workflow.py`)
*   **Expected Output**:
    *   Creates Delta tables: `mmm_bronze_raw_data`, `mmm_silver_processed_data`, and `mmm_gold_final_data_from_workflow` in the `/tmp/delta/` directory (or paths configured within the script).
    *   Prints the schema and sample data at each stage.
    *   Shows the content of the final Gold table (`mmm_gold_final_data_from_workflow`).

### 2. `mediamix/01_MMM_Data_Gen.py` (Direct Gold Table Generation)
*   **Purpose**: This script generates synthetic marketing data, including adstock and saturation effects, and saves it directly as a Gold Delta table named `mmm_gold_data`. This is a quicker alternative to running the full ETL workflow if you just want to proceed to modeling.
*   **How to run**:
    ```bash
    python mediamix/01_MMM_Data_Gen.py
    ```
    (Or `spark-submit mediamix/01_MMM_Data_Gen.py`)
*   **Expected Output**:
    *   Creates or overwrites the `mmm_gold_data` Delta table (typically in the `spark-warehouse` directory if no explicit path is given, or as defined in the script).
    *   Prints status messages and a sample of the generated data.

### 3. `mediamix/02_MMM_PyMC_Model.py` (Core MMM Model Training & Evaluation)
*   **Purpose**: This script loads the `mmm_gold_data` (generated by `01_MMM_Data_Gen.py`), trains a Bayesian Media Mix Model using PyMC3, evaluates the model (R-squared, summary statistics), generates visualizations (channel contributions, ROI), and logs the entire experiment to MLflow.
*   **Prerequisite**: Ensure the `mmm_gold_data` Delta table exists. Run `mediamix/01_MMM_Data_Gen.py` first.
*   **How to run**:
    ```bash
    python mediamix/02_MMM_PyMC_Model.py
    ```
    (Or `spark-submit mediamix/02_MMM_PyMC_Model.py`)
*   **Expected Output**:
    *   Console output showing model summary statistics and R-squared value.
    *   An MLflow run will be created. This run will contain:
        *   Logged parameters (e.g., channel names, sampling parameters).
        *   Logged metrics (e.g., R-squared, channel-specific ROIs).
        *   Logged artifacts:
            *   `channel_contributions_plot.png`: Visualization of sales contributions over time.
            *   `channel_roi_plot.png`: Bar plot of ROI per channel.
            *   `model_summary.txt`: Text file of the PyMC model summary.
            *   `mmm_model_pyfunc`: A logged MLflow PyFunc model for inference.

### Note on Databricks Environment
*   If you are working directly within Databricks notebooks, you would typically import functions from these Python scripts as modules or copy the relevant code sections into notebook cells to execute them.
*   The `python your_script.py` or `spark-submit your_script.py` commands are generally used when running scripts from a command-line interface that has access to a configured Spark context.
*   Databricks Jobs can also be configured to run these Python scripts directly from the repository.

We hope you find this repository valuable. Please feel free to explore the various documents and resources!
