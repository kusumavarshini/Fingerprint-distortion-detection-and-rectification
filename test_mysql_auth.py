from mysql_auth import get_person_templates


identity = "FAMILY-1/CHILD"

password = input("MySQL password: ")

person = get_person_templates(
    identity,
    password
)

print()
print("Person ID:", person["person_id"])
print("Identity:", person["identity_code"])
print("Family:", person["family_id"])
print("Member type:", person["member_type"])
print("Number of templates:", len(person["templates"]))

print()

for item in person["templates"]:
    template = item["template"]

    print(
        f"Impression {item['impression_id']}: "
        f"{item['image_path']} | "
        f"minutiae={template.minutiae.shape}"
    )