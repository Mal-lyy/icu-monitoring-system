import pandas as pd
import json
import time
from kafka import KafkaProducer
from datetime import datetime

DATA_DIR   = "/Users/malcolmdivinec/Downloads/mimic-iii-clinical-database-1.4"
CHART_PATH = f"{DATA_DIR}/CHARTEVENTS.csv"
ICU_PATH   = f"{DATA_DIR}/ICUSTAYS.csv"

VITAL_ITEMS = {
    211:    'heart_rate',    220045: 'heart_rate',
    51:     'systolic_bp',   442:    'systolic_bp',
    455:    'systolic_bp',   6701:   'systolic_bp',
    220179: 'systolic_bp',   220050: 'systolic_bp',
    8368:   'diastolic_bp',  8440:   'diastolic_bp',
    8441:   'diastolic_bp',  220180: 'diastolic_bp',
    220051: 'diastolic_bp',
    646:    'spo2',          220277: 'spo2',
    678:    'temperature',   679:    'temperature',
    223761: 'temperature',   224642: 'temperature',
    618:    'respiratory_rate', 615: 'respiratory_rate',
    220210: 'respiratory_rate', 224690: 'respiratory_rate'
}

print("Loading ICU patient list...")
icustays = pd.read_csv(ICU_PATH)
icustays.columns = icustays.columns.str.upper()
valid_ids  = set(icustays['ICUSTAY_ID'].dropna().astype(int))
valid_sids = set(icustays['SUBJECT_ID'].dropna().astype(int))
print(f"ICU patients found: {len(valid_sids)}")
print(f"ICU stays found:    {len(valid_ids)}")

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    key_serializer=lambda k: str(k).encode('utf-8')
)

print("\nStreaming CHARTEVENTS in chunks (ICU patients only)...")
print("This streams real ICU vitals from 46,000+ patients\n")

CHUNK_SIZE = 500000
chunk_num  = 0
total_sent = 0

for chunk in pd.read_csv(CHART_PATH, low_memory=False, chunksize=CHUNK_SIZE):
    chunk_num += 1
    chunk.columns = chunk.columns.str.upper()

    chunk = chunk[
        chunk['SUBJECT_ID'].isin(valid_sids) &
        chunk['ITEMID'].isin(VITAL_ITEMS.keys())
    ].copy()

    if chunk.empty:
        continue

    chunk = chunk[['SUBJECT_ID', 'ICUSTAY_ID', 'CHARTTIME',
                   'ITEMID', 'VALUENUM']].dropna(subset=['VALUENUM'])
    chunk['vital_name'] = chunk['ITEMID'].map(VITAL_ITEMS)

    grouped = chunk.groupby(['SUBJECT_ID', 'ICUSTAY_ID', 'CHARTTIME'])

    for (subject_id, icustay_id, charttime), group in grouped:
        reading = {
            'patient_id':       f"P{int(subject_id)}",
            'icustay_id':       int(icustay_id) if not pd.isna(icustay_id) else None,
            'timestamp':        datetime.now().isoformat(),
            'charttime':        str(charttime),
            'heart_rate':       None,
            'systolic_bp':      None,
            'diastolic_bp':     None,
            'spo2':             None,
            'temperature':      None,
            'respiratory_rate': None
        }

        for _, row in group.iterrows():
            val = float(row['VALUENUM'])
            if row['vital_name'] == 'temperature' and val > 50:
                val = round((val - 32) * 5 / 9, 1)
            else:
                val = round(val, 1)
            reading[row['vital_name']] = val

        vitals_present = sum(
            1 for k in ['heart_rate', 'systolic_bp', 'spo2',
                        'temperature', 'respiratory_rate']
            if reading[k] is not None
        )

        if vitals_present >= 2:
            producer.send('icu-vitals',
                          key=str(subject_id),
                          value=reading)
            total_sent += 1

            print(f"[{reading['patient_id']}] "
                  f"HR:{reading['heart_rate']} | "
                  f"BP:{reading['systolic_bp']} | "
                  f"SpO2:{reading['spo2']} | "
                  f"Temp:{reading['temperature']} | "
                  f"RR:{reading['respiratory_rate']}")

            time.sleep(0.5)

    print(f"Chunk {chunk_num} done — Total sent: {total_sent}")

producer.flush()
producer.close()
print(f"\nAll records streamed. Total sent: {total_sent}")
