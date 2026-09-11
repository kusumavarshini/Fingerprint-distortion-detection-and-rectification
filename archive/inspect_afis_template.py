from pathlib import Path
import cv2
from afis import MindtctExtractor


IMAGE = Path("data/split/test/FAMILY-13/CHILD/FM013_C1.png")


def main():

    print("=" * 80)
    print("AFIS MINDTCT TEMPLATE INSPECTION")
    print("=" * 80)

    image = cv2.imread(
        str(IMAGE),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise RuntimeError(f"Could not read image: {IMAGE}")

    extractor = MindtctExtractor()

    template = extractor.extract_minutiae(image)

    print("\nTemplate type:")
    print(type(template))

    print("\nTemplate attributes:")
    print(dir(template))

    print("\nTemplate dictionary:")
    try:
        print(vars(template))
    except TypeError:
        print("vars() not available")

    print("\nMinutiae attribute:")
    try:
        minutiae = template.minutiae

        print("Type:", type(minutiae))
        print("Length:", len(minutiae))

        if len(minutiae) > 0:

            print("\nFirst minutia:")
            print(minutiae[0])

            print("\nFirst minutia type:")
            print(type(minutiae[0]))

            print("\nFirst minutia attributes:")
            print(dir(minutiae[0]))

            print("\nFirst minutia dictionary:")
            try:
                print(vars(minutiae[0]))
            except TypeError:
                print("vars() not available")

    except Exception as e:
        print("Could not access template.minutiae:")
        print(repr(e))

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()