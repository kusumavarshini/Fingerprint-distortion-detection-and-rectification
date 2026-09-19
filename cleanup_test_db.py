import mysql.connector

connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password=input("MySQL password: "),
    database="fingerprint_authentication"
)

cursor = connection.cursor()

cursor.execute(
    "DELETE FROM persons WHERE identity_code = %s",
    ("TEST-FAMILY-1-CHILD",)
)

connection.commit()

print("Test record deleted. Rows:", cursor.rowcount)

cursor.close()
connection.close()