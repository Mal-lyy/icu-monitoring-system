#!/bin/bash
# ── ICU Monitoring System — Shutdown Script ───────────────────

PROJECT="/Users/malcolmdivinec/Documents/icu-monitoring-system"

echo ""
echo "[1/3] Killing Spark and producer processes..."
pkill -f "stream_processor.py" 2>/dev/null && echo "      Spark stopped." || echo "      Spark not running."
pkill -f "producer.py"         2>/dev/null && echo "      Producer stopped." || echo "      Producer not running."
pkill -f "streamlit"           2>/dev/null && echo "      Streamlit stopped." || echo "      Streamlit not running."

echo ""
echo "[2/3] Stopping Docker containers..."
cd "$PROJECT" && docker compose down
echo "      Containers stopped."

echo ""
echo "[3/3] Done. All services shut down."
echo ""
