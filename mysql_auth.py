import pickle

import mysql.connector


def get_person_templates(identity_code, mysql_password):
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password=mysql_password,
        database="fingerprint_authentication",
    )

    cursor = connection.cursor(dictionary=True)

    cursor.execute(
        """
        SELECT
            p.person_id,
            p.identity_code,
            p.family_id,
            p.member_type,
            f.fingerprint_id,
            f.impression_id,
            f.image_path,
            f.minutiae_template,
            f.minutiae_count,
            f.image_width,
            f.image_height
        FROM persons p
        JOIN fingerprints f
            ON p.person_id = f.person_id
        WHERE p.identity_code = %s
        ORDER BY f.impression_id
        """,
        (identity_code,),
    )

    rows = cursor.fetchall()

    cursor.close()
    connection.close()

    if not rows:
        raise ValueError(
            f"No enrolled fingerprint found for: {identity_code}"
        )

    templates = []

    for row in rows:
        template = pickle.loads(
            row["minutiae_template"]
        )

        templates.append(
            {
                "fingerprint_id": row["fingerprint_id"],
                "impression_id": row["impression_id"],
                "image_path": row["image_path"],
                "template": template,
                "minutiae_count": row["minutiae_count"],
                "image_width": row["image_width"],
                "image_height": row["image_height"],
            }
        )

    return {
        "person_id": rows[0]["person_id"],
        "identity_code": rows[0]["identity_code"],
        "family_id": rows[0]["family_id"],
        "member_type": rows[0]["member_type"],
        "templates": templates,
    }