import sys
import json
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline


# --- Configuration ---
LABEL_COLUMN = "TMP_C"
FEATURE_COLUMNS = ["HOUR", "MONTH", "DAY_OF_YEAR", "LATITUDE", "LONGITUDE", "ELEVATION",
                   "WIND_DIR", "WIND_SPEED", "DEW_C", "SLP_HPA", "PRECIP_MM", "RH_PCT", "VIS_M"]
STRING_TO_DOUBLE_COLS = ["LATITUDE", "LONGITUDE", "ELEVATION"]
MODEL_NAME = "DecisionTree_Tuned"

# --- Argument Parsing ---
if len(sys.argv) != 4:
    print("Usage: spark-submit train_dt.py <train_data_gcs_path> <test_data_gcs_path> <output_gcs_prefix>")
    sys.exit(-1)

TRAIN_DATA_PATH = sys.argv[1]
TEST_DATA_PATH = sys.argv[2]
OUTPUT_PREFIX = sys.argv[3].rstrip('/') # Base output path for model and artifacts

# --- Derived Paths ---
MODEL_OUTPUT_PATH = f"{OUTPUT_PREFIX}/{MODEL_NAME}/model"
METRICS_OUTPUT_PATH = f"{OUTPUT_PREFIX}/{MODEL_NAME}/metrics.json"
PLOT_OUTPUT_PATH = f"{OUTPUT_PREFIX}/{MODEL_NAME}/plot_data_csv"

# Artifact Paths added to match the standard set by RF and GBT
CV_OUTPUT_PATH = f"{OUTPUT_PREFIX}/{MODEL_NAME}/cv_results_csv"
LC_OUTPUT_PATH = f"{OUTPUT_PREFIX}/{MODEL_NAME}/learning_curve_csv"
FI_OUTPUT_PATH = f"{OUTPUT_PREFIX}/{MODEL_NAME}/feature_importances_csv"


# ------------------------------------------------------------
# 0. Utility Functions (Self-contained implementation)
# ------------------------------------------------------------

def read_data(spark, path, cols):
    """Reads Parquet data, inferring schema."""
    return spark.read.parquet(path).select(*cols)

def process_data(df, string_to_double_cols):
    """Converts specified columns to DoubleType for MLlib compatibility and handles nulls."""
    for col_name in string_to_double_cols:
        # Cast to DoubleType, which will result in NaN for non-numeric strings
        df = df.withColumn(col_name, col(col_name).cast(DoubleType()))
    # Replace NaN values (from casting) or standard nulls (None) with 0.0 in all relevant columns
    fill_cols = [c for c in df.columns if c != LABEL_COLUMN]
    df = df.fillna(0.0, subset=fill_cols)
    return df

def extract_best_params(model):
    """Extracts parameters from the best DecisionTreeRegressor model (last stage of the pipeline)."""
    if isinstance(model, DecisionTreeRegressor):
        return {
            "maxDepth": model.getMaxDepth(),
            "maxBins": model.getMaxBins(),
            "minInstancesPerNode": model.getMinInstancesPerNode()
        }
    return {} # Return empty dict if model is not the expected type

def save_metrics_to_gcs(spark, results_dict, output_path, model_name):
    """Saves metrics dictionary to GCS as a single JSON file."""
    # Ensure nested dictionary for hyperparameters is JSON serialized for storage
    if 'best_hyperparameters' in results_dict and isinstance(results_dict['best_hyperparameters'], dict):
        results_dict['best_hyperparameters'] = json.dumps(results_dict['best_hyperparameters'])

    metrics_df = spark.createDataFrame([Row(**results_dict)])
    metrics_df.coalesce(1).write.mode("overwrite").json(output_path)
    print(f"[{model_name}] Saved metrics to {output_path}")


# --- UTILITY FUNCTIONS FOR ARTIFACT GENERATION ---

def write_cv_table(spark, cv_model, param_grid, output_path, param_names):
    """
    Writes cross-validation metrics and parameters to a CSV.
    FIXED: Uses the robust parameter lookup logic from the RF script
           and maintains explicit type casting to prevent schema errors.
    """
    avg_metrics = cv_model.avgMetrics
    param_maps = cv_model.getEstimatorParamMaps() # param_maps is a list of ParamMap objects

    rows = []
    # Loop over the ParamMap object (pm) and the average metric (avg) for each run
    for pm, avg in zip(param_maps, avg_metrics):
        clean_row = {} # Using a dictionary for robustness

        for name in param_names:
            val = None

            # 1. Look up value by iterating over ParamMap (key is the Param object)
            for k, v in pm.items():
                if hasattr(k, "name") and k.name == name:
                    val = v
                    break

            # 2. Fallback: Look up value by string key (for newer Spark versions)
            if val is None and name in pm:
                val = pm[name]

            # --- Type Handling ---
            if val is None:
                clean_row[name] = None
            elif name in ["maxDepth", "maxBins"]:
                # Force casting to standard Python int to resolve schema ambiguity
                clean_row[name] = int(val)
            elif isinstance(val, (int, float, str)):
                clean_row[name] = val
            else:
                clean_row[name] = str(val)

        # Explicitly cast metric to Python's built-in float type
        clean_row['avg_rmse'] = float(avg)

        rows.append(clean_row) # Append the dictionary

    # Create DataFrame from a list of dictionaries (most robust schema inference)
    cv_df = spark.createDataFrame(rows)
    cv_df.coalesce(1).write.mode("overwrite").csv(output_path, header=True)
    print(f"[ARTIFACT] Saved Cross-Validation table to {output_path}")

def write_feature_importance_csv(spark, feature_columns, feature_importances, output_path):
    """
    Writes feature names and importance scores to a CSV.
    FIX 2 APPLIED: Switched from list of Row objects to list of dictionaries for robustness.
    """
    # Convert Vector to list
    if hasattr(feature_importances, "toArray"):
        importances_list = feature_importances.toArray().tolist()
    else:
        importances_list = feature_importances # Assume it's already a list/array

    # Create rows using dictionaries for robustness
    # Explicitly cast the importance value to a standard Python float
    rows = [{"feature": name, "importance": float(value)}
            for name, value in zip(feature_columns, importances_list)]

    fi_df = spark.createDataFrame(rows).orderBy(col("importance").desc())
    fi_df.coalesce(1).write.mode("overwrite").csv(output_path, header=True)
    print(f"[ARTIFACT] Saved Feature Importance to {output_path}")


def write_learning_curve(spark, base_stages, model_factory, best_params, train_df, test_df, evaluator, output_path):
    """
    Trains models on progressively larger subsets of the training data and
    records performance on both training and test sets.
    FIX 3 APPLIED: Simplified evaluator call and explicitly cast all Row values to standard types.
    """

    # Define training sizes (as fractions of the full training data)
    training_fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    lc_rows = []

    # Initialize the model with best hyperparameters
    model_instance = model_factory(**best_params)
    pipeline_stages = base_stages + [model_instance]

    for fraction in training_fractions:
        # Sample training data
        subset_df = train_df.sample(False, fraction, seed=42)
        subset_count = subset_df.count()

        if subset_count == 0:
            continue

        # 1. Fit the pipeline on the subset
        pipeline = Pipeline(stages=pipeline_stages)
        model = pipeline.fit(subset_df)

        # 2. Evaluate on the Training Subset
        train_predictions = model.transform(subset_df)
        train_rmse = evaluator.evaluate(train_predictions)

        # 3. Evaluate on the Full Test Set 
        test_predictions = model.transform(test_df)
        test_rmse = evaluator.evaluate(test_predictions)

        lc_rows.append(Row(
            fraction=float(fraction), # Explicit float cast
            train_count=int(subset_count),       # Explicit int cast
            train_rmse=float(train_rmse),        # Explicit float cast
            test_rmse=float(test_rmse)           # Explicit float cast
        ))

    lc_df = spark.createDataFrame(lc_rows)
    lc_df.coalesce(1).write.mode("overwrite").csv(output_path, header=True)
    print(f"[ARTIFACT] Saved Learning Curve data to {output_path}")

# ------------------------------------------------------------

# 1. Setup Spark Session
spark = SparkSession.builder.appName(f"MLlib Training: {MODEL_NAME}").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# 2. Data Loading and Preprocessing
train = read_data(spark, TRAIN_DATA_PATH, FEATURE_COLUMNS + [LABEL_COLUMN])
test = read_data(spark, TEST_DATA_PATH, FEATURE_COLUMNS + [LABEL_COLUMN])

train = process_data(train, STRING_TO_DOUBLE_COLS).withColumnRenamed(LABEL_COLUMN, "label")
test = process_data(test, STRING_TO_DOUBLE_COLS).withColumnRenamed(LABEL_COLUMN, "label")

# --- Pipeline Stages ---
assembler = VectorAssembler(inputCols=FEATURE_COLUMNS, outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features",
                        withStd=True, withMean=False)
dt = DecisionTreeRegressor(featuresCol='features', labelCol='label', seed=42)
pipeline = Pipeline(stages=[assembler, scaler, dt])

# --- Hyperparameter Grid ---
paramGrid = (ParamGridBuilder()
    .addGrid(dt.maxDepth, [4,6]) # Max depth of the tree
    .addGrid(dt.maxBins, [32,64])     # Max number of bins for discretizing continuous features
    .build())

# --- Cross-Validation ---
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
cv = CrossValidator(estimator=pipeline,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator,
                    numFolds=3,
                    seed=42)

print(f"[{MODEL_NAME}] Starting Cross-Validation...")
dt_cv_model = cv.fit(train)
print(f"[{MODEL_NAME}] Cross-Validation finished.")

# 2. Evaluate
predictions = dt_cv_model.transform(test)
# Simplified evaluation calls (metricName is already set)
rmse = evaluator.evaluate(predictions)
r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2").evaluate(predictions)
best_params = extract_best_params(dt_cv_model.bestModel.stages[-1])


# ----------------------------------------------------------------------
# --- ARTIFACTS SAVING ---
# ----------------------------------------------------------------------
print(f"[{MODEL_NAME}] Saving cross-validation, learning curve, and feature importance data...")

# 1. Cross-Validation Results
write_cv_table(spark, dt_cv_model, paramGrid, CV_OUTPUT_PATH, ["maxDepth", "maxBins"])

# 2. Learning Curve (Using the DT regressor factory function)
base_stages = [assembler, scaler]
write_learning_curve(spark, base_stages, lambda **kw: DecisionTreeRegressor(featuresCol='features', labelCol='label', seed=42, **kw),
                     best_params, train, test, evaluator, LC_OUTPUT_PATH)

# 3. Feature Importance
best_dt_model = dt_cv_model.bestModel.stages[-1]
write_feature_importance_csv(spark, FEATURE_COLUMNS, best_dt_model.featureImportances, FI_OUTPUT_PATH)

# ----------------------------------------------------------------------

# 3. Save Metrics and Model
results = {
    "model_name": MODEL_NAME,
    "test_rmse": float(rmse),
    "test_r2": float(r2),
    "best_hyperparameters": best_params
}

print(f"\n--- {MODEL_NAME} Metrics ---")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test R^2: {r2:.4f}")

# --- Save metrics using DataFrame write (reliable GCS method) ---
save_metrics_to_gcs(spark, results, METRICS_OUTPUT_PATH, MODEL_NAME)
# ----------------------------------------------------------------------

# Save Best Model
dt_cv_model.write().overwrite().save(MODEL_OUTPUT_PATH)
print(f"[{MODEL_NAME}] Saved best model to {MODEL_OUTPUT_PATH}")

# Save Plot Data (Sampled Predictions vs. Actuals)
print(f"[{MODEL_NAME}] Saving predicted vs actual plot data...")
plot_sample = predictions.select(col("prediction").alias("Predicted_Temperature_C"), col("label").alias("Actual_Temperature_C"))
# Sample 1% of the data (max 100k records) and coalesce to 1 file for easy plotting later
plot_sample.sample(withReplacement=False, fraction=0.01, seed=42).limit(100000).coalesce(1).write.mode("overwrite").csv(PLOT_OUTPUT_PATH, header=True)
print(f"[{MODEL_NAME}] Saved plot data to {PLOT_OUTPUT_PATH}")

spark.stop()
