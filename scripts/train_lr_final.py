# ============================================================
# train_lr.py  (Linear Regression with CV logs, learning curve, PLOT DATA)
# ============================================================
import sys, json
from pyspark.sql import SparkSession, Row
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
from pyspark.ml.linalg import DenseVector, SparseVector

LABEL_COLUMN = "TMP_C"
FEATURE_COLUMNS = ["HOUR","MONTH","DAY_OF_YEAR","LATITUDE","LONGITUDE","ELEVATION",
                   "WIND_DIR","WIND_SPEED","DEW_C","SLP_HPA","PRECIP_MM","RH_PCT","VIS_M"]
STRING_TO_DOUBLE_COLS = ["LATITUDE","LONGITUDE","ELEVATION"]
MODEL_NAME = "LinearRegression_Tuned"

if len(sys.argv) != 4:
    print("Usage: spark-submit train_lr.py <train.parquet> <test.parquet> <output_dir>")
    sys.exit(1)

TRAIN_PATH, TEST_PATH, OUTPUT_PREFIX = sys.argv[1:4]

spark = SparkSession.builder.appName(f"MLlib Training: {MODEL_NAME}").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

def vector_to_list(v):
    if isinstance(v, DenseVector):
        return list(v)
    if isinstance(v, SparseVector):
        dense = [0.0] * v.size
        for i, val in zip(v.indices, v.values):
            dense[i] = float(val)
        return dense
    return []

def write_lr_coefficients_csv(spark_session, feature_names, coeff_vec, intercept, out_csv_dir):
    rows = [Row(feature=f, coefficient=float(c)) for f, c in zip(feature_names, vector_to_list(coeff_vec))]
    df = spark_session.createDataFrame(rows)            .withColumn("abs_coeff", F.abs(F.col("coefficient")))            .orderBy(F.desc("abs_coeff"))            .withColumn("intercept", F.lit(float(intercept)))
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(out_csv_dir)
    print(f"[LR] Wrote coefficients to {out_csv_dir}")

def write_cv_table(spark_session, cv_model, param_maps, out_csv_dir, param_names):
    rows = []
    for pm, avg in zip(param_maps, cv_model.avgMetrics):
        clean_row = {}
        for name in param_names:
            val = None
            for k, v in pm.items():
                if hasattr(k, 'name') and k.name == name:
                    val = v
                    break
            if val is None and name in pm:
                val = pm[name]
            # force simple type
            if val is None:
                clean_row[name] = None
            elif isinstance(val, (int, float, str)):
                clean_row[name] = val
            else:
                clean_row[name] = str(val)
        clean_row["avg_rmse"] = float(avg)
        rows.append(clean_row)
    # Spark infers schema safely now
    df = spark_session.createDataFrame(rows)
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(out_csv_dir)
    print(f"[CV] Wrote results to {out_csv_dir}")

def write_learning_curve(spark_session, base_stages, mk_estimator_fn, best_params, train_df, test_df,
                         evaluator, out_csv_dir, fractions=(0.1,0.25,0.5,0.75,1.0)):
    rows = []
    for fr in fractions:
        sub = train_df.sample(False, fr, seed=42).cache()
        _ = sub.count()
        est = mk_estimator_fn(**best_params)
        pipe = Pipeline(stages=base_stages + [est])
        model = pipe.fit(sub)
        preds = model.transform(test_df)
        rmse = evaluator.evaluate(preds)
        r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2").evaluate(preds)
        rows.append(Row(fraction=float(fr), n_rows=sub.count(), rmse=float(rmse), r2=float(r2)))
        sub.unpersist()
    spark_session.createDataFrame(rows).coalesce(1).write.mode("overwrite").option("header", True).csv(out_csv_dir)
    print(f"[LC] Wrote learning curve to {out_csv_dir}")

def preprocess(df):
    for c in STRING_TO_DOUBLE_COLS:
        df = df.withColumn(c, F.col(c).cast("double"))
    df = df.withColumnRenamed(LABEL_COLUMN, "label")
    req = ["label"] + [c for c in FEATURE_COLUMNS]
    df = df.na.drop(subset=req)
    return df.cache()

# Load & clean
train_df = spark.read.parquet(TRAIN_PATH)
test_df  = spark.read.parquet(TEST_PATH)
train = preprocess(train_df)
test  = preprocess(test_df)

# Pipeline (LR needs scaling)
assembler = VectorAssembler(inputCols=FEATURE_COLUMNS, outputCol="raw_features")
scaler    = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=False)
base_stages = [assembler, scaler]

lr = LinearRegression(featuresCol="features", labelCol="label")
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")

lr_paramGrid = (ParamGridBuilder()
                .addGrid(lr.regParam, [0.0, 0.01, 0.1])
                .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
                .build())

cv = CrossValidator(estimator=lr, estimatorParamMaps=lr_paramGrid,
                    evaluator=evaluator, numFolds=3, parallelism=16, seed=42)

pipeline = Pipeline(stages=base_stages + [cv])

print(f"[{MODEL_NAME}] Fitting...")
cv_pipeline_model = pipeline.fit(train)
cv_model = cv_pipeline_model.stages[-1]
param_maps = cv_model.getEstimatorParamMaps()

print(f"[{MODEL_NAME}] Evaluating...")
predictions = cv_pipeline_model.transform(test)
rmse = evaluator.evaluate(predictions)
r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2").evaluate(predictions)
print(f"[{MODEL_NAME}] Test RMSE={rmse:.4f}  R2={r2:.4f}")

# Extract best params
best_idx = cv_model.avgMetrics.index(min(cv_model.avgMetrics))
pm = cv_model.getEstimatorParamMaps()[best_idx]
best = {"regParam": pm.get(lr.regParam), "elasticNetParam": pm.get(lr.elasticNetParam)}

# Outputs
cv_out = f"{OUTPUT_PREFIX}/{MODEL_NAME}/cv_results_csv"
lc_out = f"{OUTPUT_PREFIX}/{MODEL_NAME}/learning_curve_csv"
coef_out = f"{OUTPUT_PREFIX}/{MODEL_NAME}/coefficients_csv"
model_out = f"{OUTPUT_PREFIX}/{MODEL_NAME}/model"
metrics_out = f"{OUTPUT_PREFIX}/{MODEL_NAME}/metrics.json"
plot_out = f"{OUTPUT_PREFIX}/{MODEL_NAME}/plot_data_csv" # <-- ADDED

write_cv_table(spark, cv_model, param_maps, cv_out, ["regParam", "elasticNetParam"])
write_learning_curve(spark, base_stages, lambda **kw: LinearRegression(featuresCol='features', labelCol='label', **kw),
                     best, train, test, evaluator, lc_out)
write_lr_coefficients_csv(spark, FEATURE_COLUMNS, cv_model.bestModel.coefficients, cv_model.bestModel.intercept, coef_out)

# Save best model & metrics
cv_pipeline_model.write().overwrite().save(model_out)
metrics_df = spark.createDataFrame([Row(model_name=MODEL_NAME, test_rmse=float(rmse), test_r2=float(r2),
                                        best_hyperparameters=json.dumps(best))])
metrics_df.coalesce(1).write.mode("overwrite").json(metrics_out)
print(f"[{MODEL_NAME}] Saved model to {model_out}")
print(f"[{MODEL_NAME}] Saved metrics to {metrics_out}")

# --- ADDED: Save Plot Data (Sampled Predictions vs. Actuals) ---
print(f"[{MODEL_NAME}] Saving predicted vs actual plot data...")
plot_sample = predictions.select(F.col("prediction").alias("Predicted_Temperature_C"), F.col("label").alias("Actual_Temperature_C"))
plot_sample.sample(withReplacement=False, fraction=0.01, seed=42).limit(100000) \
    .coalesce(1).write.mode("overwrite").option("header", True).csv(plot_out)
print(f"[{MODEL_NAME}] Saved plot data CSV to {plot_out}")
# -------------------------------------------------------------

spark.stop()
