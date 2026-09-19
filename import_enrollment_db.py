import pickle
from pathlib import Path

import numpy as np
import mysql.connector
from afis.pipeline import FingerprintTemplate


GALLERY_PATH = Path(
    "experiments/afis_gallery/mindtct_gallery.pkl"
)


# --------------------------------------------------
# Load gallery
# --------------------------------------------------

print("Loading gallery...")
with GALLERY_PATH.open("rb") as f:
    gallery = pickle.load(f)

records = gallery["records"]

print("Gallery records:", len(records))


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
# Import records
# --------------------------------------------------

persons_created = 0
templates_inserted = 0
skipped = 0


for index, record in enumerate(records, start=1):

    identity = record["identity"]
    image_path = record["path"]

    # Example:
    # FAMILY-1/CHILD
    parts = identity.split("/")

    if len(parts) != 2:
        print("Skipping invalid identity:", identity)
        skipped += 1
        continue

    family_id = parts[0]
    member_type = parts[1]

    # Example:
    # FM001_C1.png
    filename = Path(image_path).name
    stem = Path(filename).stem

    # Extract impression number from filename.
    # C1 -> 1, C2 -> 2, etc.
    try:
        impression_id = int(stem.split("_")[-1][1:])
    except (ValueError, IndexError):
        print("Skipping invalid filename:", image_path)
        skipped += 1
        continue

    identity_code = identity

    # --------------------------------------------------
    # Find or create person
    # --------------------------------------------------

    cursor.execute(
        """
        SELECT person_id
        FROM persons
        WHERE identity_code = %s
        """,
        (identity_code,)
    )

    row = cursor.fetchone()

    if row:
        person_id = row[0]
    else:
        cursor.execute(
            """
            INSERT INTO persons
                (identity_code, family_id, member_type)
            VALUES
                (%s, %s, %s)
            """,
            (
                identity_code,
                family_id,
                member_type
            )
        )

        person_id = cursor.lastrowid
        persons_created += 1

    # --------------------------------------------------
    # Check duplicate impression
    # --------------------------------------------------

    cursor.execute(
        """
        SELECT fingerprint_id
        FROM fingerprints
        WHERE person_id = %s
          AND impression_id = %s
        """,
        (
            person_id,
            impression_id
        )
    )

    if cursor.fetchone():
        skipped += 1
        continue

    # --------------------------------------------------
    # Create compact template
    # --------------------------------------------------

    original = record["template"]

    compact = FingerprintTemplate(
        minutiae=np.array(
            original.minutiae,
            copy=True
        ),
        header=original.header,
        report=original.report,
    )

    compact_blob = pickle.dumps(
        compact,
        protocol=pickle.HIGHEST_PROTOCOL
    )

    # --------------------------------------------------
    # Insert fingerprint
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
            impression_id,
            image_path,
            compact_blob,
            int(compact.minutiae.shape[0]),
            int(record["width"]),
            int(record["height"])
        )
    )

    templates_inserted += 1

    if index % 100 == 0:
        connection.commit()
        print(
            f"Processed {index}/{len(records)} "
            f"| persons: {persons_created} "
            f"| templates: {templates_inserted}"
        )


# --------------------------------------------------
# Final commit
# --------------------------------------------------

connection.commit()


# --------------------------------------------------
# Verify counts
# --------------------------------------------------

cursor.execute("SELECT COUNT(*) FROM persons")
person_count = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM fingerprints")
fingerprint_count = cursor.fetchone()[0]


print()
print("========================================")
print("IMPORT COMPLETE")
print("========================================")
print("Persons created:", persons_created)
print("Templates inserted:", templates_inserted)
print("Skipped:", skipped)
print("Persons in database:", person_count)
print("Fingerprints in database:", fingerprint_count)


cursor.close()
connection.close()