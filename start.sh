#!/bin/bash
# ── ICU Monitoring System — Full Startup Script ──────────────
# Run from project root: bash start.sh
# Logs written to logs/ for debugging

PROJECT="/Users/malcolmdivinec/Documents/icu-monitoring-system"
mkdir -p "$PROJECT/logs"

echo ""
echo "══════════════════════════════════════════════════"
echo "  ICU Real-Time Monitoring System — Starting Up"
echo "══════════════════════════════════════════════════"

# ── Step 1: Start Docker containers ──────────────────────────
echo ""
echo "[1/5] Starting Kafka + MongoDB containers..."
cd "$PROJECT"
docker compose up -d

echo "      Waiting 15s for services to be ready..."
sleep 15

# Quick health check
if ! docker ps | grep -q "kafka"; then
    echo "ERROR: Kafka container not running. Check: docker compose logs kafka"
    exit 1
fi
if ! docker ps | grep -q "mongodb"; then
    echo "ERROR: MongoDB container not running. Check: docker compose logs mongodb"
    exit 1
fi
echo "      Containers OK."

# ── Step 2: Start Spark Stream Processor ─────────────────────
echo ""
echo "[2/5] Starting Spark stream processor (background)..."
nohup spark-submit \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
    "$PROJECT/spark/stream_processor.py" \
    > "$PROJECT/logs/spark.log" 2>&1 &
SPARK_PID=$!
echo "      Spark PID: $SPARK_PID  |  tail -f logs/spark.log to monitor"
sleep 10

# ── Step 3: Start Kafka Producer ─────────────────────────────
echo ""
echo "[3/5] Starting Kafka producer (background)..."
nohup python3 "$PROJECT/simulator/producer.py" \
    > "$PROJECT/logs/producer.log" 2>&1 &
PRODUCER_PID=$!
echo "      Producer PID: $PRODUCER_PID  |  tail -f logs/producer.log to monitor"
sleep 3

# ── Step 4: Run Sepsis Detector (batch, once) ─────────────────
echo ""
echo "[4/5] Running batch enrichment scripts..."
python3 "$PROJECT/sepsis_detector.py" > "$PROJECT/logs/sepsis.log" 2>&1
[ $? -eq 0 ] && echo "      Sepsis scores written to MongoDB." \
             || echo "      Warning — check logs/sepsis.log"

python3 "$PROJECT/charlson_scorer.py" > "$PROJECT/logs/charlson.log" 2>&1
[ $? -eq 0 ] && echo "      CCI scores written to MongoDB." \
             || echo "      Warning — check logs/charlson.log"

python3 "$PROJECT/demographics_enricher.py" > "$PROJECT/logs/demographics.log" 2>&1
[ $? -eq 0 ] && echo "      Demographics written to MongoDB." \
             || echo "      Warning — check logs/demographics.log"

# ── Step 5: Launch Streamlit Dashboard ───────────────────────
echo ""
echo "[5/5] Launching Streamlit dashboard..."
echo "      Dashboard → http://localhost:8501"
echo ""
echo "══════════════════════════════════════════════════"
echo "  To stop everything: bash stop.sh"
echo "══════════════════════════════════════════════════"
echo ""

cd "$PROJECT/dashboard"
streamlit run app.py
