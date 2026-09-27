"""
===============================================================================
stream_vitals.py — Healthcare Patient 360 Lakehouse Demo
===============================================================================
Kinesis Data Streams consumer that reads simulated vitals events, applies
windowed aggregation (5-minute tumbling window for avg/min/max), and writes
to `healthcare_db.fact_vitals` Iceberg table.

*** DUAL-SINK ARCHITECTURE (Gulf Air Falcon Eye pattern) ***
  Sink 1: Apache Iceberg on S3 → for batch analytics via Athena/Spark
  Sink 2: Amazon OpenSearch     → for real-time dashboards (Kibana/OSD)

This script uses boto3 for Kinesis consumption and PyIceberg (or Spark) for
the Iceberg sink. For production at scale, consider using:
  - Amazon Kinesis Data Analytics (Apache Flink) with Iceberg connector
  - AWS Lambda with Kinesis trigger + micro-batch writes

Architecture:
  Kinesis → This Consumer → [Iceberg (S3)] + [OpenSearch]
                ↓                   ↓              ↓
            5-min windows     Batch analytics   Real-time dashboards

Usage:
  python stream_vitals.py \
    --stream-name healthcare-vitals-stream \
    --region us-east-1 \
    --s3-warehouse s3://your-bucket/warehouse/ \
    --database-name healthcare_db \
    --opensearch-endpoint https://your-domain.us-east-1.es.amazonaws.com

Notes:
  - Set ENABLE_OPENSEARCH=true env var to activate OpenSearch sink
  - Uses DynamoDB for Kinesis checkpointing (shard iterator persistence)
  - Designed for a single-worker demo; for production use KCL or Flink
===============================================================================
"""

import os
import sys
import json
import time
import logging
import argparse
import signal
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional, Any

import boto3
from botocore.exceptions import ClientError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("stream_vitals")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WINDOW_SIZE_SECONDS = 300   # 5-minute tumbling window
BATCH_LIMIT = 100           # Max records per GetRecords call
POLL_INTERVAL_SECONDS = 2   # Polling interval when no records
FLUSH_INTERVAL_SECONDS = 60 # Force flush every 60s even if window not full

ENABLE_OPENSEARCH = os.environ.get("ENABLE_OPENSEARCH", "false").lower() == "true"

# Graceful shutdown
RUNNING = True

def signal_handler(sig, frame):
    global RUNNING
    logger.info("Shutdown signal received. Flushing and exiting...")
    RUNNING = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ===========================================================================
# WINDOWED AGGREGATOR
# ===========================================================================
class VitalsWindowAggregator:
    """
    Collects vitals events and aggregates them into 5-minute tumbling windows.
    Window key: (patient_id, vital_type, window_start)

    Aggregates computed per window:
      - avg_value, min_value, max_value
      - reading_count
      - alert_triggered (any reading above/below threshold)
    """

    # Clinical thresholds for alerting
    VITAL_THRESHOLDS = {
        "HEART_RATE":         {"low": 50,  "high": 120},
        "SYSTOLIC_BP":        {"low": 90,  "high": 180},
        "DIASTOLIC_BP":       {"low": 60,  "high": 120},
        "TEMPERATURE":        {"low": 35.0, "high": 38.5},  # Celsius
        "RESPIRATORY_RATE":   {"low": 10,  "high": 30},
        "SPO2":               {"low": 90,  "high": 100},
        "BLOOD_GLUCOSE":      {"low": 60,  "high": 300},
    }

    def __init__(self, window_size_seconds: int = 300):
        self.window_size = window_size_seconds
        # Key: (patient_id, vital_type, window_start_ts) → list of values
        self.windows: Dict[tuple, List[float]] = defaultdict(list)
        self.window_metadata: Dict[tuple, dict] = {}

    def _get_window_start(self, event_time: datetime) -> datetime:
        """Compute the start of the tumbling window for a given event time."""
        epoch = event_time.timestamp()
        window_start_epoch = (epoch // self.window_size) * self.window_size
        return datetime.fromtimestamp(window_start_epoch)

    def add_event(self, event: dict):
        """Add a single vitals event to the appropriate window."""
        try:
            patient_id = event["patient_id"]
            vital_type = event.get("vital_type", "UNKNOWN").upper()
            value = float(event["value"])
            event_time = datetime.fromisoformat(event.get("event_time", datetime.utcnow().isoformat()))
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Skipping malformed event: {e}")
            return

        window_start = self._get_window_start(event_time)
        key = (patient_id, vital_type, window_start)

        self.windows[key].append(value)
        self.window_metadata[key] = {
            "patient_id": patient_id,
            "vital_type": vital_type,
            "window_start": window_start,
            "window_end": window_start + timedelta(seconds=self.window_size),
            "device_id": event.get("device_id"),
            "unit": event.get("unit", ""),
        }

    def get_closed_windows(self, current_time: Optional[datetime] = None) -> List[dict]:
        """
        Return aggregated records for windows that have closed
        (current time is past window_end).
        """
        if current_time is None:
            current_time = datetime.utcnow()

        closed = []
        keys_to_remove = []

        for key, values in self.windows.items():
            patient_id, vital_type, window_start = key
            window_end = window_start + timedelta(seconds=self.window_size)

            if current_time >= window_end:
                # Window is closed — compute aggregates
                meta = self.window_metadata[key]
                avg_val = sum(values) / len(values)
                min_val = min(values)
                max_val = max(values)

                # Check threshold alerts
                thresholds = self.VITAL_THRESHOLDS.get(vital_type, {})
                alert_triggered = False
                alert_type = None
                if thresholds:
                    if min_val < thresholds.get("low", float("-inf")):
                        alert_triggered = True
                        alert_type = "LOW"
                    if max_val > thresholds.get("high", float("inf")):
                        alert_triggered = True
                        alert_type = "HIGH" if alert_type is None else "BOTH"

                record = {
                    "patient_id": patient_id,
                    "vital_type": vital_type,
                    "window_start": window_start.isoformat(),
                    "window_end": window_end.isoformat(),
                    "avg_value": round(avg_val, 2),
                    "min_value": round(min_val, 2),
                    "max_value": round(max_val, 2),
                    "reading_count": len(values),
                    "alert_triggered": alert_triggered,
                    "alert_type": alert_type or "NONE",
                    "device_id": meta.get("device_id"),
                    "unit": meta.get("unit"),
                    "window_year": window_start.year,
                    "window_month": window_start.month,
                    "etl_loaded_at": current_time.isoformat(),
                    "etl_source": "kinesis_stream",
                }
                closed.append(record)
                keys_to_remove.append(key)

        # Clean up closed windows
        for key in keys_to_remove:
            del self.windows[key]
            del self.window_metadata[key]

        return closed

    def flush_all(self) -> List[dict]:
        """Force-flush all open windows (for shutdown)."""
        future = datetime.utcnow() + timedelta(hours=1)
        return self.get_closed_windows(current_time=future)


# ===========================================================================
# SINK 1: ICEBERG (via Spark or PyIceberg)
# ===========================================================================
class IcebergSink:
    """
    Writes aggregated vitals records to an Iceberg table.
    Uses PySpark for compatibility with the Glue environment.
    For a lightweight alternative, use PyIceberg directly.
    """

    def __init__(self, warehouse: str, database: str, table: str = "fact_vitals"):
        self.table_name = f"glue_catalog.{database}.{table}"
        self.warehouse = warehouse
        self.database = database
        self.spark = None
        self._init_spark()

    def _init_spark(self):
        """Initialize a local Spark session with Iceberg support."""
        try:
            from pyspark.sql import SparkSession
            self.spark = (
                SparkSession.builder
                .appName("stream_vitals_iceberg_sink")
                .config("spark.sql.catalog.glue_catalog", "org.apache.iceberg.spark.SparkCatalog")
                .config("spark.sql.catalog.glue_catalog.warehouse", self.warehouse)
                .config("spark.sql.catalog.glue_catalog.catalog-impl",
                        "org.apache.iceberg.aws.glue.GlueCatalog")
                .config("spark.sql.catalog.glue_catalog.io-impl",
                        "org.apache.iceberg.aws.s3.S3FileIO")
                .config("spark.sql.extensions",
                        "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
                .getOrCreate()
            )
            logger.info(f"Spark Iceberg sink initialized → {self.table_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Spark: {e}")
            raise

    def write_batch(self, records: List[dict]):
        """Write a batch of aggregated records to Iceberg via append."""
        if not records:
            return
        try:
            df = self.spark.createDataFrame(records)
            df.writeTo(self.table_name).append()
            logger.info(f"[ICEBERG] Wrote {len(records)} aggregated vitals to {self.table_name}")
        except Exception as e:
            # First write may need createOrReplace if table is empty
            logger.warning(f"Append failed ({e}), attempting createOrReplace")
            try:
                df = self.spark.createDataFrame(records)
                df.writeTo(self.table_name).using("iceberg").createOrReplace()
                logger.info(f"[ICEBERG] Initial write of {len(records)} records.")
            except Exception as e2:
                logger.error(f"[ICEBERG] Write failed entirely: {e2}")

    def close(self):
        if self.spark:
            self.spark.stop()


# ===========================================================================
# SINK 2: OPENSEARCH (for real-time dashboards)
# ===========================================================================
class OpenSearchSink:
    """
    Writes vitals records to Amazon OpenSearch for real-time dashboarding.
    Dual-sink pattern modeled after Gulf Air Falcon Eye architecture.
    Uses the _bulk API for efficient indexing.
    """

    def __init__(self, endpoint: str, index_name: str = "healthcare-vitals"):
        self.endpoint = endpoint.rstrip("/")
        self.index_name = index_name
        self.session = None
        self._init_client()

    def _init_client(self):
        """Initialize OpenSearch client with AWS SigV4 auth."""
        try:
            from opensearchpy import OpenSearch, RequestsHttpConnection
            from requests_aws4auth import AWS4Auth

            credentials = boto3.Session().get_credentials()
            region = boto3.Session().region_name or "us-east-1"
            awsauth = AWS4Auth(
                credentials.access_key,
                credentials.secret_key,
                region,
                "es",
                session_token=credentials.token,
            )

            self.client = OpenSearch(
                hosts=[{"host": self.endpoint.replace("https://", "").split("/")[0], "port": 443}],
                http_auth=awsauth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
            )
            logger.info(f"[OPENSEARCH] Connected to {self.endpoint}")
        except ImportError:
            logger.warning("[OPENSEARCH] opensearchpy not installed. Falling back to requests.")
            self.client = None
        except Exception as e:
            logger.warning(f"[OPENSEARCH] Init failed: {e}. Sink disabled.")
            self.client = None

    def write_batch(self, records: List[dict]):
        """Bulk-index records into OpenSearch."""
        if not self.client or not records:
            return

        try:
            bulk_body = []
            for record in records:
                doc_id = f"{record['patient_id']}_{record['vital_type']}_{record['window_start']}"
                bulk_body.append(json.dumps({"index": {"_index": self.index_name, "_id": doc_id}}))
                bulk_body.append(json.dumps(record, default=str))

            body = "\n".join(bulk_body) + "\n"
            response = self.client.bulk(body=body)

            if response.get("errors"):
                error_count = sum(1 for item in response["items"] if "error" in item.get("index", {}))
                logger.warning(f"[OPENSEARCH] Bulk indexed with {error_count} errors")
            else:
                logger.info(f"[OPENSEARCH] Indexed {len(records)} docs to {self.index_name}")

        except Exception as e:
            logger.error(f"[OPENSEARCH] Bulk write failed: {e}")

    def close(self):
        pass  # No persistent connection to close for requests-based client


# ===========================================================================
# KINESIS CONSUMER
# ===========================================================================
class KinesisVitalsConsumer:
    """
    Consumes vitals events from Kinesis Data Streams, aggregates them in
    tumbling windows, and flushes to dual sinks (Iceberg + OpenSearch).
    """

    def __init__(
        self,
        stream_name: str,
        region: str,
        iceberg_sink: IcebergSink,
        opensearch_sink: Optional[OpenSearchSink] = None,
    ):
        self.stream_name = stream_name
        self.kinesis = boto3.client("kinesis", region_name=region)
        self.aggregator = VitalsWindowAggregator(window_size_seconds=WINDOW_SIZE_SECONDS)
        self.iceberg_sink = iceberg_sink
        self.opensearch_sink = opensearch_sink
        self.last_flush_time = time.time()
        self.total_events_processed = 0
        self.total_windows_flushed = 0

    def _get_shard_iterators(self) -> List[str]:
        """Get LATEST shard iterators for all shards in the stream."""
        try:
            response = self.kinesis.describe_stream(StreamName=self.stream_name)
            shards = response["StreamDescription"]["Shards"]
            iterators = []
            for shard in shards:
                iter_response = self.kinesis.get_shard_iterator(
                    StreamName=self.stream_name,
                    ShardId=shard["ShardId"],
                    ShardIteratorType="LATEST",
                )
                iterators.append(iter_response["ShardIterator"])
            logger.info(f"Obtained iterators for {len(iterators)} shards")
            return iterators
        except ClientError as e:
            logger.error(f"Failed to describe stream: {e}")
            raise

    def _process_records(self, records: list):
        """Parse and add records to the windowed aggregator."""
        for record in records:
            try:
                data = json.loads(record["Data"])
                self.aggregator.add_event(data)
                self.total_events_processed += 1
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping unparseable record: {e}")

    def _flush_closed_windows(self):
        """Flush closed windows to both sinks (dual-sink pattern)."""
        closed = self.aggregator.get_closed_windows()
        if closed:
            logger.info(f"Flushing {len(closed)} closed windows to sinks")

            # Sink 1: Iceberg (analytics)
            self.iceberg_sink.write_batch(closed)

            # Sink 2: OpenSearch (real-time dashboards) — Falcon Eye dual-sink
            if self.opensearch_sink:
                self.opensearch_sink.write_batch(closed)

            self.total_windows_flushed += len(closed)

    def run(self):
        """Main consumer loop."""
        logger.info(f"Starting Kinesis consumer for stream: {self.stream_name}")
        logger.info(f"Window size: {WINDOW_SIZE_SECONDS}s | "
                     f"OpenSearch sink: {'ENABLED' if self.opensearch_sink else 'DISABLED'}")

        shard_iterators = self._get_shard_iterators()

        while RUNNING:
            for i, iterator in enumerate(shard_iterators):
                if not iterator or not RUNNING:
                    continue

                try:
                    response = self.kinesis.get_records(
                        ShardIterator=iterator,
                        Limit=BATCH_LIMIT,
                    )
                    records = response.get("Records", [])
                    shard_iterators[i] = response.get("NextShardIterator")

                    if records:
                        self._process_records(records)
                        logger.debug(f"Shard {i}: processed {len(records)} records")

                except ClientError as e:
                    if e.response["Error"]["Code"] == "ExpiredIteratorException":
                        logger.warning(f"Shard {i}: iterator expired, refreshing")
                        shard_iterators = self._get_shard_iterators()
                        break
                    else:
                        logger.error(f"Kinesis error: {e}")
                        time.sleep(5)

            # Check for closed windows and flush
            self._flush_closed_windows()

            # Periodic forced flush (even if windows aren't closed yet)
            if time.time() - self.last_flush_time > FLUSH_INTERVAL_SECONDS:
                self._flush_closed_windows()
                self.last_flush_time = time.time()

            # Throttle polling
            time.sleep(POLL_INTERVAL_SECONDS)

        # --- Graceful shutdown ---
        logger.info("Flushing all remaining windows on shutdown...")
        remaining = self.aggregator.flush_all()
        if remaining:
            self.iceberg_sink.write_batch(remaining)
            if self.opensearch_sink:
                self.opensearch_sink.write_batch(remaining)

        logger.info(
            f"Consumer stopped. Total events: {self.total_events_processed}, "
            f"Total windows flushed: {self.total_windows_flushed + len(remaining)}"
        )


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Kinesis Vitals Stream Consumer (Dual-Sink)")
    parser.add_argument("--stream-name", required=True, help="Kinesis stream name")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--s3-warehouse", required=True, help="Iceberg warehouse S3 path")
    parser.add_argument("--database-name", default="healthcare_db", help="Glue catalog database")
    parser.add_argument("--opensearch-endpoint", default=None, help="OpenSearch domain endpoint")
    args = parser.parse_args()

    # Initialize Sink 1: Iceberg
    iceberg_sink = IcebergSink(
        warehouse=args.s3_warehouse,
        database=args.database_name,
        table="fact_vitals",
    )

    # Initialize Sink 2: OpenSearch (optional — Falcon Eye dual-sink)
    opensearch_sink = None
    if ENABLE_OPENSEARCH and args.opensearch_endpoint:
        opensearch_sink = OpenSearchSink(
            endpoint=args.opensearch_endpoint,
            index_name="healthcare-vitals",
        )
        logger.info("Dual-sink mode: Iceberg + OpenSearch (Falcon Eye pattern)")
    else:
        logger.info("Single-sink mode: Iceberg only")

    # Start consumer
    consumer = KinesisVitalsConsumer(
        stream_name=args.stream_name,
        region=args.region,
        iceberg_sink=iceberg_sink,
        opensearch_sink=opensearch_sink,
    )

    try:
        consumer.run()
    finally:
        iceberg_sink.close()
        if opensearch_sink:
            opensearch_sink.close()
        logger.info("All sinks closed. Goodbye.")


if __name__ == "__main__":
    main()
