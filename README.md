# DSA5208 Project 2 – Temperature Prediction Using Spark MLlib

This guide outlines the steps to:
- Run preprocessing locally to generate training and test datasets
- Set up Google Cloud Dataproc Serverless
- Upload data and scripts to GCS
- Train four models using Dataproc (LR, DT, RF, GBT)
- Download model results
- Generate the final report

---

## 📁 Folder Structure

```
.
├── data/                     # Will store train.parquet and test.parquet
├── scripts/                 # Contains preprocessing and training scripts
│   ├── preprocess.py
│   ├── train_lr_final.py
│   ├── train_rf_final.py
│   ├── train_gbt_final.py
│   ├── train_dt_final.py
│   └── report_generation_final.py
```

---

## 💻 Step 1: Run Preprocessing Locally Using PySpark

Open **PowerShell**, navigate to the project folder containing `scripts/`, and run:

```bash
spark-submit scripts/preprocess.py
```

This will output:
```
data/train.parquet
data/test.parquet
```

> Make sure you have Apache Spark installed locally with PySpark configured in your PATH.

---

## ☁️ Step 2: Authenticate with Google Cloud

```bash
gcloud auth login
```

---

## 🌐 Step 3: Set Environment Variables

```bash
export PROJECT_ID="dsa5208-project2-474818" #change this according your project ID
export PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")
export SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
export REGION="asia-southeast1"
export GCS_STAGING_BUCKET="gs://weather-ml-2024" #change this according to the name of the bucket created for the project
export MODEL_OUTPUT_GCS="gs://weather-ml-2024/models/"
export TEST_DATA_GCS="gs://weather-ml-2024/data/test.parquet"
export TRAIN_DATA_GCS="gs://weather-ml-2024/data/train.parquet"
export SCRIPT_GCS_PATH_RF="gs://weather-ml-2024/scripts/train_rf_final.py"
export SCRIPT_GCS_PATH_GBT="gs://weather-ml-2024/scripts/train_gbt_final.py"
export SCRIPT_GCS_PATH_LR="gs://weather-ml-2024/scripts/train_lr_final.py"
export SCRIPT_GCS_PATH_DT="gs://weather-ml-2024/scripts/train_dt_final.py"
```

---

## ☁️ Step 4: Upload Data and Scripts to GCS

```bash
gsutil cp data/train.parquet ${TRAIN_DATA_GCS}
gsutil cp data/test.parquet ${TEST_DATA_GCS}
gsutil cp scripts/*.py gs://weather-ml-2024/scripts/
```

---

## 🚀 Step 5: Submit Model Training Jobs to Dataproc Serverless

### Linear Regression
```bash
gcloud dataproc batches submit pyspark "${SCRIPT_GCS_PATH_LR}" \
  --project="${PROJECT_ID}" --region="${REGION}" --batch="weather-ml-lr-v1" \
  --version="2.2" --deps-bucket="${GCS_STAGING_BUCKET}" \
  --properties spark.executor.cores=4,spark.executor.memory=8g,spark.executor.instances=7,spark.driver.memory=4g \
  -- "${TRAIN_DATA_GCS}" "${TEST_DATA_GCS}" "${MODEL_OUTPUT_GCS}"
```

### Decision Tree
```bash
gcloud dataproc batches submit pyspark "${SCRIPT_GCS_PATH_DT}" \
  --project="${PROJECT_ID}" --region="${REGION}" --batch="weather-ml-dt-v1" \
  --version="2.2" --deps-bucket="${GCS_STAGING_BUCKET}" \
  --properties spark.executor.cores=4,spark.executor.memory=8g,spark.executor.instances=7,spark.driver.memory=4g \
  -- "${TRAIN_DATA_GCS}" "${TEST_DATA_GCS}" "${MODEL_OUTPUT_GCS}"
```

### Gradient-Boosted Trees
```bash
gcloud dataproc batches submit pyspark "${SCRIPT_GCS_PATH_GBT}" \
  --project="${PROJECT_ID}" --region="${REGION}" --batch="weather-ml-gbt-v1" \
  --version="2.2" --deps-bucket="${GCS_STAGING_BUCKET}" \
  --properties spark.executor.cores=4,spark.executor.memory=8g,spark.executor.instances=7,spark.driver.memory=4g \
  -- "${TRAIN_DATA_GCS}" "${TEST_DATA_GCS}" "${MODEL_OUTPUT_GCS}"
```

### Random Forest
```bash
gcloud dataproc batches submit pyspark "${SCRIPT_GCS_PATH_RF}" \
  --project="${PROJECT_ID}" --region="${REGION}" --batch="weather-ml-rf-v1" \
  --version="2.2" --deps-bucket="${GCS_STAGING_BUCKET}" \
  --properties spark.executor.cores=4,spark.executor.memory=8g,spark.executor.instances=7,spark.driver.memory=4g \
  -- "${TRAIN_DATA_GCS}" "${TEST_DATA_GCS}" "${MODEL_OUTPUT_GCS}"
```

---

## 📦 Step 6: Download Model Output

```bash
gsutil -m cp -r "gs://weather-ml-2024/models" .
```

---

## 📝 Step 7: Generate Final Report Locally

Ensure the following are in the same local folder:
- The zipped version of the four model result folders (from GCS or `results/`)
- The `scripts/` folder containing `report_generation_final.py`

```bash
python scripts/report_generation_final.py .
```

This creates:
```
assignment_report.zip
```

---

## 📁 Additional Folder: `results/`

The GitHub repository includes a `results/` folder which contains:
- Previously trained model output folders
- Generated visualizations and metrics
- all result plots and report-ready figures
