# ==========================================================
# 🚀 PHASE 1: Initialize Spark Session
# ==========================================================
import logging
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import os
import re
from pyspark.sql.types import DoubleType

# ---------------- Setup logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

log.info("\n" + "=" * 70 + "\n🚀 PHASE 1: Initialize Spark Session\n" + "=" * 70)

# ---------------- Create Spark session ----------------
spark = (
    SparkSession.builder.appName("NOAA_Preprocessing")
    .config("spark.driver.memory", "6g")
    .config("spark.executor.memory", "6g")
    .getOrCreate()
)

# Silence Spark’s internal noise
spark.sparkContext.setLogLevel("ERROR")

log.info("✅ Phase 1 complete. Spark session started successfully.")

# ==========================================================
# 🚀 PHASE 2: Load and Inspect NOAA CSVs
# ==========================================================
log.info("\n" + "=" * 70 + "\n🚀 PHASE 2: Load and Inspect NOAA CSVs\n" + "=" * 70)

data_dir = "data/2024"

df = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .csv(os.path.join(data_dir, "*.csv"))
)

row_count = df.count()
col_count = len(df.columns)
log.info(f"✅ Loaded {row_count:,} rows × {col_count} columns.")
log.info(f"✅ Phase 2 complete. Columns: {df.columns}. Rows: {row_count:,}")

# ==========================================================
# 🚀 PHASE 3: Select Essential Columns
# ==========================================================
log.info("\n" + "=" * 70 + "\n🚀 PHASE 3: Select Essential Columns\n" + "=" * 70)

essential_cols = [
    "STATION", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION",
    "WND", "TMP", "DEW", "SLP", "VIS", "AA1"
]

existing_cols = [c for c in essential_cols if c in df.columns]
missing_cols = set(essential_cols) - set(existing_cols)

if missing_cols:
    log.warning(f"⚠️ Missing columns in input: {missing_cols}")
else:
    log.info("✅ All essential columns found.")

df = df.select(existing_cols)
df = df.filter(F.col("TMP").isNotNull() & F.col("DATE").isNotNull())
log.info(f"✅ Phase 3 complete. Kept {len(existing_cols)} columns; remaining rows: {df.count():,}")

# ======================================================================
# 🚀 PHASE 4: Parse & Clean Packed ISD Fields (Enhanced)
# ======================================================================
log.info("\n" + "=" * 70 + "\n🚀 PHASE 4: Parse & Clean Packed ISD Fields\n" + "=" * 70)
log.info("🔍 Starting field parsing with enhanced cleaning...")

# --- UDFs ---
@F.udf("struct<dir:double,speed:double>")
def parse_wnd_udf(value):
    if not value or value in ["", "NULL", "None"]:
        return (None, None)
    parts = str(value).split(",")
    if len(parts) < 5:
        return (None, None)
    try:
        direction = float(parts[0])
        if direction == 999:
            direction = None
    except:
        direction = None
    try:
        speed = float(parts[3]) / 10.0
        if speed >= 999.9:
            speed = None
    except:
        speed = None
    return (direction, speed)

@F.udf(DoubleType())
def parse_numeric_udf(value, scale):
    if not value or value in ["", "NULL", "None"]:
        return None
    try:
        s = str(value).split(",")[0].strip("+")
        if re.match(r"^9{4,}$", s):
            return None
        return float(s) / float(scale)
    except:
        return None

# --- Clean whitespace ---
df = df.select([F.regexp_replace(F.col(c), r"\s+", "").alias(c) for c in df.columns])

# --- Apply parsing ---
df = df.withColumn("WND_STRUCT", parse_wnd_udf(F.col("WND")))
df = df.withColumn("WIND_DIR", F.col("WND_STRUCT.dir"))
df = df.withColumn("WIND_SPEED", F.col("WND_STRUCT.speed")).drop("WND_STRUCT")

df = (
    df.withColumn("TMP_C", parse_numeric_udf(F.col("TMP"), F.lit(10.0)))
      .withColumn("DEW_C", parse_numeric_udf(F.col("DEW"), F.lit(10.0)))
      .withColumn("SLP_HPA", parse_numeric_udf(F.col("SLP"), F.lit(10.0)))
      .withColumn("PRECIP_MM", parse_numeric_udf(F.col("AA1"), F.lit(10.0)))
)

# --- Compute Relative Humidity (RH_PCT) ---
df = df.withColumn(
    "RH_PCT",
    F.when(
        F.col("TMP_C").isNotNull() & F.col("DEW_C").isNotNull(),
        100 * (
            F.exp((17.625 * F.col("DEW_C")) / (243.04 + F.col("DEW_C"))) /
            F.exp((17.625 * F.col("TMP_C")) / (243.04 + F.col("TMP_C")))
        )
    ).otherwise(None)
)

# Handle calm wind directions
df = df.withColumn(
    "WIND_DIR",
    F.when(F.col("WIND_DIR").isNull() & (F.col("WIND_SPEED") > 0), 0.0)
     .otherwise(F.col("WIND_DIR"))
)

# --- Parse VIS (Visibility) ---
# ISD format: VVVVV,quality -> e.g. "16000,1" means 16000 meters visibility
def parse_vis(v):
    try:
        main_part = v.split(",")[0]
        if main_part.strip().isdigit():
            val = float(main_part)
            # Convert unrealistic 99999 (no limit) to None
            return val if val < 99999 else None
        else:
            return None
    except Exception:
        return None

parse_vis_udf = F.udf(parse_vis, DoubleType())
df = df.withColumn("VIS_M", parse_vis_udf("VIS"))

df = df.cache()
log.info(f"✅ Phase 4 complete. Derived columns: TMP_C, DEW_C, SLP_HPA, PRECIP_MM, RH_PCT, VIS_M. Rows: {df.count():,}")


# ======================================================================
# 🚀 PHASE 5: Filter and Validate Features
# ======================================================================
log.info("\n" + "=" * 70 + "\n🚀 PHASE 5: Filter and Validate Features\n" + "=" * 70)

df_clean = df.filter(
    F.col("TMP_C").isNotNull()
    | F.col("WIND_SPEED").isNotNull()
    | F.col("DEW_C").isNotNull()
    | F.col("SLP_HPA").isNotNull()
    | F.col("PRECIP_MM").isNotNull()
    | F.col("RH_PCT").isNotNull()
)

df_clean = (
    df_clean
    .filter((F.col("TMP_C") >= -90) & (F.col("TMP_C") <= 65))
    .filter((F.col("DEW_C") >= -95) & (F.col("DEW_C") <= 40))
    .filter((F.col("SLP_HPA") >= 850) & (F.col("SLP_HPA") <= 1100))
    .filter((F.col("WIND_SPEED") >= 0) & (F.col("WIND_SPEED") <= 90))
    .filter((F.col("PRECIP_MM") >= 0) & (F.col("PRECIP_MM") <= 1000))
    .filter((F.col("RH_PCT") >= 0) & (F.col("RH_PCT") <= 100))
    .dropDuplicates(["STATION", "DATE"])
)

log.info(f"✅ Phase 5 complete. Validated dataset size: {df_clean.count():,} rows.")

# ======================================================================
# 🚀 PHASE 6: Prepare Cleaned Hourly Dataset for Modeling
# ======================================================================
log.info("\n" + "=" * 70 + "\n🚀 PHASE 6: Prepare Cleaned Hourly Dataset for Modeling\n" + "=" * 70)

df_clean = df_clean.withColumn("DATETIME", F.to_timestamp("DATE"))
df_clean = df_clean.filter(F.minute("DATETIME") == 0)
df_clean = df_clean.withColumn("DATE_HOUR", F.date_trunc("hour", F.col("DATETIME")))
df_clean = df_clean.withColumn("DATE_DAY", F.to_date(F.col("DATETIME")))

# --- Temporal feature engineering ---
df_clean = (
    df_clean.withColumn("HOUR", F.hour("DATE_HOUR"))
    .withColumn("MONTH", F.month("DATE_HOUR"))
    .withColumn("DAY_OF_YEAR", F.dayofyear("DATE_HOUR"))
)

log.info(f"✅ Phase 6 complete. Feature-engineered dataset size: {df_clean.count():,} rows.")

# ======================================================================
# 🚀 PHASE 7: Train/Test Split, Validation & Export
# ======================================================================
log.info("\n" + "=" * 70 + "\n🚀 PHASE 7: Train/Test Split, Validation & Export\n" + "=" * 70)

# --- Select modeling features ---
df_ml = df_clean.select(
    "STATION", "DATE_HOUR", "HOUR", "MONTH", "DAY_OF_YEAR",
    "LATITUDE", "LONGITUDE", "ELEVATION",
    "WIND_DIR", "WIND_SPEED", "DEW_C", "SLP_HPA",
    "PRECIP_MM", "RH_PCT", "VIS_M", "TMP_C"
)

# --- Drop rows with any missing values ---
before_drop = df_ml.count()
df_ml = df_ml.na.drop()
after_drop = df_ml.count()
log.info(f"🧹 Dropped {before_drop - after_drop:,} rows with nulls. Remaining: {after_drop:,}.")

# --- Summary statistics ---
numeric_cols = [
    "LATITUDE", "LONGITUDE", "ELEVATION", "WIND_DIR", "WIND_SPEED",
    "DEW_C", "SLP_HPA", "PRECIP_MM", "RH_PCT", "VIS_M", "TMP_C",
    "HOUR", "MONTH", "DAY_OF_YEAR"
]
summary_df = (
    df_ml.select(
        *[F.count(F.when(F.col(c).isNull(), c)).alias(f"{c}_nulls") for c in numeric_cols],
        *[F.min(c).alias(f"{c}_min") for c in numeric_cols],
        *[F.max(c).alias(f"{c}_max") for c in numeric_cols]
    )
)
summary = summary_df.collect()[0].asDict()
log.info("📊 Feature summary (nulls/min/max):")
for c in numeric_cols:
    log.info(
        f"   {c:<12} | nulls: {summary.get(f'{c}_nulls', 0):>5} | "
        f"min: {summary.get(f'{c}_min', 'N/A')} | max: {summary.get(f'{c}_max', 'N/A')}"
    )

# --- Train/Test Split ---
train_df, test_df = df_ml.randomSplit([0.7, 0.3], seed=42)
output_dir = "output"

# --- Write as single parquet files ---
train_df.coalesce(1).write.mode("overwrite").parquet(f"{output_dir}/train_tmp")
test_df.coalesce(1).write.mode("overwrite").parquet(f"{output_dir}/test_tmp")

import shutil, glob
train_part = glob.glob(f"{output_dir}/train_tmp/part-*.parquet")[0]
test_part = glob.glob(f"{output_dir}/test_tmp/part-*.parquet")[0]
shutil.move(train_part, f"{output_dir}/train.parquet")
shutil.move(test_part, f"{output_dir}/test.parquet")
shutil.rmtree(f"{output_dir}/train_tmp")
shutil.rmtree(f"{output_dir}/test_tmp")

log.info("✅ Preprocessing complete — single train/test parquet files saved in 'output/'")
log.info(f"   Train count: {train_df.count():,} | Test count: {test_df.count():,}")



