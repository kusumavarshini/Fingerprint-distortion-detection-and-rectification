import csv
import time
from pathlib import Path

import mysql.connector
import numpy as np
from afis.pipeline import MindtctExtractor
from sklearn.metrics import roc_auc_score, roc_curve

from mysql_auth import get_person_templates


OUTPUT_DIR = Path("experiments/1to1_mysql")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "scores_safe.csv"

MYSQL_PASSWORD = input("MySQL password: ")

MAX_GENUINE = 2500
MAX_IMPOSTOR = 2500

extractor = MindtctExtractor()


# --------------------------------------------------
# Load identities
# --------------------------------------------------

connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password=MYSQL_PASSWORD,
    database="fingerprint_authentication",
)

cursor = connection.cursor()

cursor.execute(
    "SELECT identity_code FROM persons ORDER BY identity_code"
)

identities = [row[0] for row in cursor.fetchall()]

cursor.close()
connection.close()

print("Identities:", len(identities))


# --------------------------------------------------
# Load templates
# --------------------------------------------------

people = {}

for i, identity in enumerate(identities, 1):

    people[identity] = get_person_templates(
        identity,
        MYSQL_PASSWORD
    )

    if i % 25 == 0:
        print(f"Loaded {i}/{len(identities)} identities")


# --------------------------------------------------
# CSV
# --------------------------------------------------

fields = [
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
        fieldnames=fields
    )

    writer.writeheader()


def save(row):

    with CSV_PATH.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writerow(row)


# --------------------------------------------------
# Genuine
# --------------------------------------------------

print()
print("GENUINE MATCHING")

genuine_scores = []
count = 0

start = time.time()

for identity in identities:

    templates = people[identity]["templates"]

    for i in range(5):

        for j in range(i + 1, 5):

            if count >= MAX_GENUINE:
                break

            result = extractor.match(
                templates[i]["template"],
                templates[j]["template"]
            )

            score = float(result.score)

            genuine_scores.append(score)

            save({
                "label": 1,
                "identity_a": identity,
                "identity_b": identity,
                "impression_a": templates[i]["impression_id"],
                "impression_b": templates[j]["impression_id"],
                "score": score,
                "raw_score": float(result.raw_score),
                "decision": bool(result.decision),
            })

            count += 1

            if count % 100 == 0:

                elapsed = time.time() - start
                rate = count / elapsed

                print(
                    f"Genuine {count}/{MAX_GENUINE} "
                    f"| {rate:.2f}/sec"
                )

        if count >= MAX_GENUINE:
            break

    if count >= MAX_GENUINE:
        break


print("Genuine complete:", count)


# --------------------------------------------------
# Impostor
# --------------------------------------------------

print()
print("IMPOSTOR MATCHING")

impostor_scores = []
count = 0

start = time.time()

for i in range(len(identities)):

    identity_a = identities[i]

    template_a = people[
        identity_a
    ]["templates"][0]["template"]

    for j in range(i + 1, len(identities)):

        if count >= MAX_IMPOSTOR:
            break

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

        save({
            "label": 0,
            "identity_a": identity_a,
            "identity_b": identity_b,
            "impression_a": 1,
            "impression_b": 1,
            "score": score,
            "raw_score": float(result.raw_score),
            "decision": bool(result.decision),
        })

        count += 1

        if count % 100 == 0:

            elapsed = time.time() - start
            rate = count / elapsed

            print(
                f"Impostor {count}/{MAX_IMPOSTOR} "
                f"| {rate:.2f}/sec"
            )

    if count >= MAX_IMPOSTOR:
        break


print("Impostor complete:", count)


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

fnr = 1 - tpr

eer_index = np.argmin(
    np.abs(fpr - fnr)
)

eer = (
    fpr[eer_index]
    + fnr[eer_index]
) / 2

eer_threshold = thresholds[eer_index]


# --------------------------------------------------
# Results
# --------------------------------------------------

print()
print("========================================")
print("1:1 MYSQL RESULTS")
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

print("ROC-AUC:", auc)

print("Approximate EER:", eer)

print("EER threshold:", eer_threshold)

print()

print("Results:", CSV_PATH)