"""Dehydrate the sensor and random tweet datasets for publication.

Produces privacy- and license-safe parquet files. Three protections are
applied. First, all free text and profile fields are removed. Second,
all URL and domain columns are removed so the per-domain NewsGuard
score table cannot be reconstructed from the published rows. Third, all
platform identifiers (tweet id, user id, referenced tweet id,
conversation id) are replaced by salted SHA-256 hashes, which blocks
rehydration via the X API while preserving referential integrity, so
retweet chains and per-user groupings survive.

The salt is generated once and stored NEXT TO THE PRIVATE source data
(never published, never committed). With the salt the mapping stays
reproducible internally, without it the hashes are one-way.

Run: python dehydrate.py  (streams both parquets, ~58M rows total)
"""
import hashlib
import secrets
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SRC_DIR = Path("/home/mangermaier/cs2/twitter/user")
OUT_DIR = SRC_DIR / "dehydrated_for_publication"
SALT_FILE = SRC_DIR / "dehydration_salt_PRIVATE.txt"

SOURCES = {
    "sensor": SRC_DIR / "result_sensor_with_users_unraveled_v5.parquet",
    "random": SRC_DIR / "result_random_with_users_unraveled_v5.parquet",
}

HASH_COLS = ["TWEET_id", "USER_id", "referenced_tweet_id", "conversation_id"]

# Columns carried over unchanged. Everything not listed here or in
# HASH_COLS is dropped on purpose (text, profiles, mentions, URLs,
# domains, unused API metadata, unused model scores).
KEEP_COLS = [
    "referenced_tweet_type",
    "TWEET_created_at",
    "lang",
    "TWEET_like_count",
    "retweet_count",
    "quote_count",
    "bookmark_count",
    "reply_count",
    "impression_count",
    "newsguard_scores_expanded",
    "newsguard_orientation",
    "USER_followers_count",
    "USER_following_count",
    "USER_tweet_count",
    "USER_listed_count",
    "USER_verified",
    "USER_created_at",
]

RENAME = {"newsguard_scores_expanded": "newsguard_scores"}


def load_salt():
    if SALT_FILE.exists():
        return SALT_FILE.read_text().strip().encode()
    salt = secrets.token_hex(32)
    SALT_FILE.write_text(salt + "\n")
    SALT_FILE.chmod(0o600)
    print(f"generated new salt -> {SALT_FILE}")
    return salt.encode()


def hash_id(value, salt, cache):
    if value is None or value == "":
        return None
    h = cache.get(value)
    if h is None:
        h = hashlib.sha256(salt + value.encode()).hexdigest()[:16]
        cache[value] = h
    return h


def dehydrate(name, src, salt):
    out_path = OUT_DIR / f"{name}_dehydrated.parquet"
    pf = pq.ParquetFile(src)
    available = set(pf.schema_arrow.names)
    hash_cols = [c for c in HASH_COLS if c in available]
    keep_cols = [c for c in KEEP_COLS if c in available]
    missing = [c for c in HASH_COLS + KEEP_COLS if c not in available]
    if missing:
        print(f"[{name}] note: source lacks {missing}")

    writer = None
    cache = {}
    done = 0
    t0 = time.time()
    for batch in pf.iter_batches(batch_size=200_000,
                                 columns=hash_cols + keep_cols):
        d = batch.to_pydict()
        arrays, names = [], []
        for c in hash_cols:
            names.append(c + "_hashed")
            arrays.append(pa.array(
                [hash_id(v, salt, cache) for v in d[c]], type=pa.string()))
        for c in keep_cols:
            names.append(RENAME.get(c, c))
            arrays.append(batch.column(batch.schema.get_field_index(c)))
        table = pa.table(dict(zip(names, arrays)))
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema,
                                      compression="snappy")
        writer.write_table(table, row_group_size=1_000_000)
        done += len(batch)
        if done % 2_000_000 < 200_000:
            print(f"[{name}] {done:,} rows  ({time.time()-t0:.0f}s)",
                  flush=True)
    writer.close()
    print(f"[{name}] DONE {done:,} rows -> {out_path}  "
          f"({time.time()-t0:.0f}s, {out_path.stat().st_size/1e9:.2f} GB)",
          flush=True)


def main():
    salt = load_salt()
    for name, src in SOURCES.items():
        dehydrate(name, src, salt)


if __name__ == "__main__":
    main()
