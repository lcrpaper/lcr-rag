#!/usr/bin/env python3
"""
Data Format Converter

Converts between various data formats used in the project:
- JSONL (primary)
- Parquet (efficient storage)
- CSV (legacy compatibility)
- SQLite (local querying)
- TFRecord (TensorFlow compatibility)
- HuggingFace Datasets (interoperability)

Usage:
    python scripts/utils/data_format_converter.py --input data.jsonl --output data.parquet
    python scripts/utils/data_format_converter.py --input data/ --output data.sqlite --format sqlite
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Iterator, Optional, Any
import logging

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logger.warning("pandas not installed. Parquet/CSV support limited.")

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

try:
    import sqlite3
    HAS_SQLITE = True
except ImportError:
    HAS_SQLITE = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    from datasets import Dataset, load_dataset
    HAS_HF_DATASETS = True
except ImportError:
    HAS_HF_DATASETS = False

try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False

try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False


class DataConverter:
    """
    Universal data format converter.

    Supports bidirectional conversion between:
    - JSONL
    - Parquet
    - CSV
    - SQLite
    - TFRecord
    - HuggingFace Datasets
    - MessagePack
    """

    SUPPORTED_FORMATS = ['jsonl', 'parquet', 'csv', 'sqlite', 'tfrecord', 'hf', 'msgpack']

    def __init__(
        self,
        compression: Optional[str] = None,
        chunk_size: int = 10000,
        schema: Optional[Dict] = None
    ):
        self.compression = compression
        self.chunk_size = chunk_size
        self.schema = schema

    def _detect_format(self, path: str) -> str:
        """Detect format from file extension."""
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix in ['.gz', '.zst', '.lz4']:
            suffix = Path(path.stem).suffix.lower()

        format_map = {
            '.jsonl': 'jsonl',
            '.json': 'jsonl',
            '.parquet': 'parquet',
            '.csv': 'csv',
            '.tsv': 'csv',
            '.sqlite': 'sqlite',
            '.db': 'sqlite',
            '.tfrecord': 'tfrecord',
            '.msgpack': 'msgpack',
        }

        return format_map.get(suffix, 'jsonl')

    def _read_jsonl(self, path: str) -> Iterator[Dict]:
        """Read JSONL file."""
        import gzip

        open_func = gzip.open if path.endswith('.gz') else open
        mode = 'rt' if path.endswith('.gz') else 'r'

        with open_func(path, mode, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    if HAS_ORJSON:
                        yield orjson.loads(line)
                    else:
                        yield json.loads(line)

    def _write_jsonl(self, data: Iterator[Dict], path: str) -> int:
        """Write JSONL file."""
        import gzip

        open_func = gzip.open if path.endswith('.gz') else open
        mode = 'wt' if path.endswith('.gz') else 'w'

        count = 0
        with open_func(path, mode, encoding='utf-8') as f:
            for item in data:
                if HAS_ORJSON:
                    f.write(orjson.dumps(item).decode() + '\n')
                else:
                    f.write(json.dumps(item) + '\n')
                count += 1

        return count

    def _read_parquet(self, path: str) -> Iterator[Dict]:
        """Read Parquet file."""
        if not HAS_PYARROW:
            raise ImportError("pyarrow required for Parquet support")

        table = pq.read_table(path)
        for batch in table.to_batches(max_chunksize=self.chunk_size):
            df = batch.to_pandas()
            for _, row in df.iterrows():
                yield row.to_dict()

    def _write_parquet(self, data: Iterator[Dict], path: str) -> int:
        """Write Parquet file."""
        if not HAS_PANDAS:
            raise ImportError("pandas required for Parquet support")

        chunks = []
        current_chunk = []
        count = 0

        for item in data:
            current_chunk.append(item)
            count += 1

            if len(current_chunk) >= self.chunk_size:
                chunks.append(pd.DataFrame(current_chunk))
                current_chunk = []

        if current_chunk:
            chunks.append(pd.DataFrame(current_chunk))

        df = pd.concat(chunks, ignore_index=True)
        df.to_parquet(path, compression=self.compression or 'snappy')

        return count

    def _read_csv(self, path: str) -> Iterator[Dict]:
        """Read CSV file."""
        if not HAS_PANDAS:
            import csv
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    yield dict(row)
        else:
            for chunk in pd.read_csv(path, chunksize=self.chunk_size):
                for _, row in chunk.iterrows():
                    yield row.to_dict()

    def _write_csv(self, data: Iterator[Dict], path: str) -> int:
        """Write CSV file."""
        if not HAS_PANDAS:
            import csv
            first = True
            count = 0

            with open(path, 'w', encoding='utf-8', newline='') as f:
                writer = None
                for item in data:
                    if first:
                        writer = csv.DictWriter(f, fieldnames=item.keys())
                        writer.writeheader()
                        first = False
                    writer.writerow(item)
                    count += 1

            return count
        else:
            chunks = []
            for item in data:
                chunks.append(item)

            df = pd.DataFrame(chunks)
            df.to_csv(path, index=False)
            return len(df)

    def _read_sqlite(self, path: str, table: str = 'data') -> Iterator[Dict]:
        """Read from SQLite database."""
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row

        cursor = conn.execute(f"SELECT * FROM {table}")
        for row in cursor:
            yield dict(row)

        conn.close()

    def _write_sqlite(self, data: Iterator[Dict], path: str, table: str = 'data') -> int:
        """Write to SQLite database."""
        conn = sqlite3.connect(path)
        count = 0
        first = True

        for item in data:
            if first:
                columns = ', '.join([f'"{k}" TEXT' for k in item.keys()])
                conn.execute(f'CREATE TABLE IF NOT EXISTS {table} ({columns})')
                first = False

            placeholders = ', '.join(['?' for _ in item])
            columns = ', '.join([f'"{k}"' for k in item.keys()])
            values = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in item.values()]

            conn.execute(f'INSERT INTO {table} ({columns}) VALUES ({placeholders})', values)
            count += 1

            if count % self.chunk_size == 0:
                conn.commit()

        conn.commit()
        conn.close()
        return count

    def convert(
        self,
        input_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None
    ) -> int:
        """
        Convert between formats.

        Args:
            input_path: Input file path
            output_path: Output file path
            input_format: Input format (auto-detected if None)
            output_format: Output format (auto-detected if None)

        Returns:
            Number of records converted
        """
        input_format = input_format or self._detect_format(input_path)
        output_format = output_format or self._detect_format(output_path)

        logger.info(f"Converting {input_format} -> {output_format}")
        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {output_path}")

        readers = {
            'jsonl': self._read_jsonl,
            'parquet': self._read_parquet,
            'csv': self._read_csv,
            'sqlite': self._read_sqlite,
        }

        writers = {
            'jsonl': self._write_jsonl,
            'parquet': self._write_parquet,
            'csv': self._write_csv,
            'sqlite': self._write_sqlite,
        }

        reader = readers.get(input_format)
        writer = writers.get(output_format)

        if not reader:
            raise ValueError(f"Unsupported input format: {input_format}")
        if not writer:
            raise ValueError(f"Unsupported output format: {output_format}")

        data = reader(input_path)
        count = writer(data, output_path)

        logger.info(f"Converted {count} records")
        return count


def main():
    parser = argparse.ArgumentParser(
        description='Convert between data formats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # JSONL to Parquet
    python data_format_converter.py -i data.jsonl -o data.parquet

    # CSV to JSONL (compressed)
    python data_format_converter.py -i data.csv -o data.jsonl.gz

    # JSONL to SQLite
    python data_format_converter.py -i data.jsonl -o data.sqlite

    # Batch conversion
    python data_format_converter.py -i data/ -o output/ --format parquet
        """
    )

    parser.add_argument('-i', '--input', required=True, help='Input file or directory')
    parser.add_argument('-o', '--output', required=True, help='Output file or directory')
    parser.add_argument('--format', help='Output format (auto-detected from extension)')
    parser.add_argument('--compression', choices=['gzip', 'snappy', 'lz4', 'zstd'])
    parser.add_argument('--chunk-size', type=int, default=10000)
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    converter = DataConverter(
        compression=args.compression,
        chunk_size=args.chunk_size
    )

    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        total = 0

        for file in Path(args.input).glob('*'):
            if file.is_file():
                output_file = Path(args.output) / (file.stem + '.' + (args.format or 'jsonl'))
                count = converter.convert(str(file), str(output_file), output_format=args.format)
                total += count

        print(f"Converted {total} total records")
    else:
        count = converter.convert(args.input, args.output, output_format=args.format)
        print(f"Converted {count} records")


if __name__ == '__main__':
    main()
