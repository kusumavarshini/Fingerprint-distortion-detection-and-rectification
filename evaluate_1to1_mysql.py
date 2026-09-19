import csv
import time
from pathlib import Path

import mysql.connector
import numpy as np
from afis.pipeline import MindtctExtractor
from sklearn.metrics import roc_auc_score, roc_curve

from mysql_auth import get_person_templates


# --------------------------------------------------
# Configuration
# --------------------------------------------------

OUTPUT_DIR = Path("experiments/1to1_mysql")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "scores.csv"

# Number of impostor comparisons to evaluate.
# Start with 3,000 for a fast baseline.
MAX_IMPOSTOR_PAIRS = 3000

MYSQL_PASSWORD = input("MySQL password: ")

extractor = MindtctExtractor()


# --------------------------------------------------
# Connect to MySQL
# --------------------------------------------------

connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password=MYSQL_PASSWORD,
    database="fingerprint_authentication",
)

cursor = connection.cursor()

cursor.execute(
    """
    SELECT identity_code
    FROM persons
    ORDER BY identity_code
    """
)

identities = [row[0] for row in cursor.fetchall()]

cursor.close()
connection.close()

print("Identities:", len(identities))


# --------------------------------------------------
# Load all templates once
# --------------------------------------------------

people = {}

print()
print("Loading templates...")

for i, identity in enumerate(identities, start=1):

    people[identity] = get_person_templates(
        identity,
        MYSQL_PASSWORD
    )

    if i % 25 == 0:
        print(
            f"Loaded {i}/{len(identities)} identities"
        )


# --------------------------------------------------
# Prepare CSV
# --------------------------------------------------

fieldnames = [
    "label",
    "identity_a",
    "identity_b",
    "impression_a",
    "impression_b",
    "score",
    "raw_score",
    "decision",
]

with CSV_PATH.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()


# --------------------------------------------------
# Helper: append result
# --------------------------------------------------

def save_result(result):

    with CSV_PATH.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writerow(result)


# --------------------------------------------------
# Genuine comparisons
# --------------------------------------------------

print()
print("========================================")
print("GENUINE MATCHING")
print("========================================")

start_time = time.time()

genuine_count = 0
genuine_scores = []

total_genuine = len(identities) * 10

for identity, person in people.items():

    templates = person["templates"]

    for i in range(len(templates)):

        for j in range(i + 1, len(templates)):

            result = extractor.match(
                templates[i]["template"],
                templates[j]["template"]
            )

            score = float(result.score)

            genuine_scores.append(score)

            save_result(
                {
                    "label": 1,
                    "identity_a": identity,
                    "identity_b": identity,
                    "impression_a": templates[i]["impression_id"],
                    "impression_b": templates[j]["impression_id"],
                    "score": score,
                    "raw_score": float(result.raw_score),
                    "decision": bool(result.decision),
                }
            )

            genuine_count += 1

            if genuine_count % 100 == 0:

                elapsed = time.time() - start_time
                rate = genuine_count / elapsed

                print(
                    f"Genuine: {genuine_count}/{total_genuine} "
                    f"| {rate:.2f} comparisons/sec "
                    f"| elapsed {elapsed/60:.1f} min"
                )


genuine_elapsed = time.time() - start_time

print()
print(
    f"Genuine complete: {genuine_count} comparisons"
)

print(
    f"Genuine time: {genuine_elapsed/60:.2f} minutes"
)


# --------------------------------------------------
# Impostor comparisons
# --------------------------------------------------

print()
print("========================================")
print("IMPOSTOR MATCHING")
print("========================================")

start_time = time.time()

impostor_scores = []
impostor_count = 0

# Compare impression 1 of each identity.
#
# We generate deterministic pairs by identity order.
# Stop after MAX_IMPOSTOR_PAIRS.

stop = False

for i in range(len(identities)):

    if stop:
        break

    identity_a = identities[i]

    template_a = people[
        identity_a
    ]["templates"][0]["template"]

    for j in range(i + 1, len(identities)):

        identity_b = identities[j]

        template_b = people[
            identity_b
        ]["templates"][0]["template"]

        result = extractor.match(
            template_a,
            template_b
        )

        score = float(result.score)

        impostor_scores.append(score)

        save_result(
            {
                "label": 0,
                "identity_a": identity_a,
                "identity_b": identity_b,
                "impression_a": 1,
                "impression_b": 1,
                "score": score,
                "raw_score": float(result.raw_score),
                "decision": bool(result.decision),
            }
        )

        impostor_count += 1

        if impostor_count % 100 == 0:

            elapsed = time.time() - start_time
            rate = impostor_count / elapsed
            remaining = (
                MAX_IMPOSTOR_PAIRS - impostor_count
            )

            eta = remaining / rate if rate > 0 else 0

            print(
                f"Impostor: {impostor_count}/"
                f"{MAX_IMPOSTOR_PAIRS} "
                f"| {rate:.2f} comparisons/sec "
                f"| ETA {eta/60:.1f} min"
            )

        if impostor_count >= MAX_IMPOSTOR_PAIRS:
            stop = True
            break


impostor_elapsed = time.time() - start_time

print()
print(
    f"Impostor complete: {impostor_count} comparisons"
)

print(
    f"Impostor time: {impostor_elapsed/60:.2f} minutes"
)


# --------------------------------------------------
# Metrics
# --------------------------------------------------

y_true = np.array(
    [1] * len(genuine_scores)
    + [0] * len(impostor_scores)
)

y_score = np.array(
    genuine_scores
    + impostor_scores
)

auc = roc_auc_score(
    y_true,
    y_score
)

fpr, tpr, thresholds = roc_curve(
    y_true,
    y_score
)

fnr = 1.0 - tpr

eer_index = np.argmin(
    np.abs(fpr - fnr)
)

eer = (
    fpr[eer_index]
    + fnr[eer_index]
) / 2

eer_threshold = thresholds[eer_index]


# --------------------------------------------------
# Final results
# --------------------------------------------------

print()
print("========================================")
print("1:1 MYSQL MATCHING RESULTS")
print("========================================")

print("Persons:", len(identities))

print(
    "Genuine comparisons:",
    len(genuine_scores)
)

print(
    "Impostor comparisons:",
    len(impostor_scores)
)

print()
print(
    "Genuine mean:",
    np.mean(genuine_scores)
)

print(
    "Genuine median:",
    np.median(genuine_scores)
)

print(
    "Genuine min:",
    np.min(genuine_scores)
)

print(
    "Genuine max:",
    np.max(genuine_scores)
)

print()
print(
    "Impostor mean:",
    np.mean(impostor_scores)
)

print(
    "Impostor median:",
    np.median(impostor_scores)
)

print(
    "Impostor min:",
    np.min(impostor_scores)
)

print(
    "Impostor max:",
    np.max(impostor_scores)
)

print()
print(
    "ROC-AUC:",
    auc
)

print()
print(
    "Approximate EER:",
    eer
)

print(
    "EER threshold:",
    eer_threshold
)

print()
print("CSV:", CSV_PATH)