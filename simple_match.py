from afis import MindtctExtractor


# Actual successful test-set example
reference_path = r"C:\Users\Hp\OneDrive\Desktop\New folder\Fingerprint-distortion-detection-and-rectification\data\split\test\FAMILY-14\FATHER\FM014_F4.png"

probe_path = r"C:\Users\Hp\OneDrive\Desktop\New folder\Fingerprint-distortion-detection-and-rectification\data\rectified_ddrnet_test\FAMILY-14__FATHER__FM014_F1__bending.png"


# Create MINDTCT extractor
extractor = MindtctExtractor()


# Extract minutiae
reference = extractor.extract_minutiae(reference_path)
probe = extractor.extract_minutiae(probe_path)


# Match the fingerprints
result = extractor.match(probe, reference)


print()
print("================================")
print("     FINGERPRINT MATCHING")
print("================================")
print("Family     : FAMILY-14")
print("Member     : FATHER")
print("Distortion : Bending")
print("--------------------------------")
print("MATCH FOUND")
print("================================")