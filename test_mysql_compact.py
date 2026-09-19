import pickle
from pathlib import Path

import numpy as np
import mysql.connector
from afis.pipeline import FingerprintTemplate, MindtctExtractor


GALLERY_PATH = Path(
    "experiments/afis_gallery/mindtct_gallery.pkl"
)


# --------------------------------------------------
# Load one existing template
# --------------------------------------------------

with GALLERY_PATH.open("rb") as f:
    gallery = pickle.load(f)

record = gallery["records"][0]

original = record["template"]


# --------------------------------------------------
# Create compact template
# --------------------------------------------------

compact = FingerprintTemplate(
    minutiae=np.array(original.minutiae, copy=True),
    header=original.header,
    report=original.report,
)

print("Identity:", record["identity"])
print("Image:", record["path"])
print("Minutiae:", compact.minutiae.shape)


# --------------------------------------------------
# Serialize compact template
# --------------------------------------------------

compact_blob = pickle.dumps(
    compact,
    protocol=pickle.HIGHEST_PROTOCOL
)

print("Compact template size:", len(compact_blob), "bytes")


# --------------------------------------------------
# Connect to MySQL
# --------------------------------------------------

connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password=input("Enter MySQL root password: "),
    database="fingerprint_authentication"
)

cursor = connection.cursor()


# --------------------------------------------------
# Insert temporary test person
# --------------------------------------------------

cursor.execute(
    """
    INSERT INTO persons
        (identity_code, family_id, member_type)
    VALUES
        (%s, %s, %s)
    """,
    (
        "COMPACT-TEST",
        "COMPACT-TEST-FAMILY",
        "CHILD"
    )
)

person_id = cursor.lastrowid


# --------------------------------------------------
# Insert compact template
# --------------------------------------------------

cursor.execute(
    """
    INSERT INTO fingerprints
        (
            person_id,
            impression_id,
            image_path,
            minutiae_template,
            minutiae_count,
            image_width,
            image_height
        )
    VALUES
        (%s, %s, %s, %s, %s, %s, %s)
    """,
    (
        person_id,
        1,
        record["path"],
        compact_blob,
        compact.minutiae.shape[0],
        record["width"],
        record["height"]
    )
)

connection.commit()

print("Compact template inserted.")
print("Person ID:", person_id)


# --------------------------------------------------
# Read template back from MySQL
# --------------------------------------------------

cursor.execute(
    """
    SELECT minutiae_template
    FROM fingerprints
    WHERE person_id = %s
    """,
    (person_id,)
)

row = cursor.fetchone()

if row is None:
    raise RuntimeError("Could not retrieve template from MySQL.")

database_blob = row[0]

print("Retrieved blob size:", len(database_blob), "bytes")


# --------------------------------------------------
# Reconstruct template
# --------------------------------------------------

database_template = pickle.loads(database_blob)

print(
    "Retrieved template type:",
    type(database_template)
)

print(
    "Retrieved minutiae:",
    database_template.minutiae.shape
)


# --------------------------------------------------
# Compare original vs database-restored template
# --------------------------------------------------

extractor = MindtctExtractor()

result_original = extractor.match(
    original,
    original
)

result_database = extractor.match(
    original,
    database_template
)

print()
print("Original → Original:", result_original)
print("Original → Database:", result_database)


# --------------------------------------------------
# Verify
# --------------------------------------------------

if result_original == result_database:
    print()
    print("PASS: MySQL round-trip preserved the Bozorth3 result.")
else:
    print()
    print("WARNING: Bozorth3 results differ.")


# --------------------------------------------------
# Cleanup temporary test record
# --------------------------------------------------

cursor.execute(
    "DELETE FROM persons WHERE person_id = %s",
    (person_id,)
)

connection.commit()

print("Temporary test record deleted.")


cursor.close()
connection.close()