import pickle
from pathlib import Path

import mysql.connector


# ---------------------------------------------------------
# 1. Load ONE existing MINDTCT template from our .pkl
# ---------------------------------------------------------
gallery_path = Path(
    "experiments/afis_gallery/mindtct_gallery.pkl"
)

with gallery_path.open("rb") as f:
    gallery = pickle.load(f)

record = gallery["records"][0]

template = record["template"]

print("Identity:", record["identity"])
print("Image:", record["path"])
print("Template type:", type(template))
print("Minutiae:", template.minutiae.shape)


# ---------------------------------------------------------
# 2. Serialize the exact FingerprintTemplate object
# ---------------------------------------------------------
template_blob = pickle.dumps(
    template,
    protocol=pickle.HIGHEST_PROTOCOL
)

print("Serialized template size:",
      len(template_blob), "bytes")


# ---------------------------------------------------------
# 3. Connect to MySQL
# ---------------------------------------------------------
connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password=input("Enter MySQL root password: "),
    database="fingerprint_authentication"
)

cursor = connection.cursor()


# ---------------------------------------------------------
# 4. Insert ONE test person
# ---------------------------------------------------------
cursor.execute(
    """
    INSERT INTO persons
        (identity_code, family_id, member_type)
    VALUES
        (%s, %s, %s)
    """,
    (
        "TEST-FAMILY-1-CHILD",
        "TEST-FAMILY-1",
        "CHILD"
    )
)

person_id = cursor.lastrowid


# ---------------------------------------------------------
# 5. Insert ONE fingerprint template
# ---------------------------------------------------------
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
        template_blob,
        template.minutiae.shape[0],
        record["width"],
        record["height"]
    )
)

connection.commit()

print("Database insert successful.")
print("Person ID:", person_id)

cursor.close()
connection.close()