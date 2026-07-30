"""
Project 4 - Image or Text Recognition (Basic)
-----------------------------------------------
Goal: implement a basic recognition task using a pre-trained/library-based
AI tool, run it on a sample input, and clearly display the output.

I went with Path 1 (OCR) rather than the object detection path, since
pytesseract doesn't require downloading a separate multi-megabyte model
file - it wraps the Tesseract engine directly, which made it a lot easier
to get a fully working pipeline end to end without network access.

Pipeline (matches the 4 validation checks from the brief):
  1. Library integration      -> pytesseract wraps Google's Tesseract OCR
  2. Pre-processing integrity -> grayscale, blur, deskew, adaptive threshold
  3. Accuracy benchmarking    -> 80% confidence gate on every detected word
  4. Visual confirmation      -> annotated image + printed text output

Author: Ishtiaq
"""

import sys
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont

CONFIDENCE_THRESHOLD = 80  # per the brief, this is the minimum accepted standard
SAMPLE_IMAGE_PATH = "sample_input.png"
OUTPUT_DIR = "."


# ---------------------------------------------------------------------------
# STEP 0: If no image is supplied, generate a sample "scanned document"
# style image so the pipeline has something realistic to work on - slightly
# noisy, unevenly lit, and a touch rotated, like a real phone photo of a page
# ---------------------------------------------------------------------------
def generate_sample_image(path=SAMPLE_IMAGE_PATH):
    width, height = 900, 400
    img = Image.new("RGB", (width, height), color=(235, 235, 230))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40
        )
    except OSError:
        font = ImageFont.load_default()

    lines = ["DECODELABS", "Internship Project 4", "OCR Recognition Test"]
    y = 60
    for line in lines:
        draw.text((80, y), line, fill=(20, 20, 20), font=font)
        y += 90

    arr = np.array(img).astype(np.float32)

    # uneven lighting - a soft gradient across the page, like light falling
    # unevenly across a scanned document
    gradient = np.tile(np.linspace(-40, 40, width), (height, 1))
    arr[:, :, 0] += gradient
    arr[:, :, 1] += gradient
    arr[:, :, 2] += gradient

    # a bit of sensor-style noise
    noise = np.random.normal(0, 8, arr.shape)
    arr += noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    noisy_img = Image.fromarray(arr)
    noisy_img = noisy_img.rotate(-2.5, expand=True, fillcolor=(235, 235, 230))
    noisy_img.save(path)
    return path


# ---------------------------------------------------------------------------
# STEP 1: Pre-processing pipeline
# ---------------------------------------------------------------------------
def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def reduce_noise(gray_image):
    # smooths out sensor noise / compression artifacts before thresholding
    return cv2.GaussianBlur(gray_image, (5, 5), 0)


def deskew(gray_image):
    # find the angle of the text blob and rotate it back to horizontal
    inverted = cv2.bitwise_not(gray_image)
    thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))

    if len(coords) < 10:
        return gray_image  # not enough signal to safely estimate an angle

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = gray_image.shape
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray_image, matrix, (w, h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def adaptive_threshold(gray_image):
    # Otsu's method automatically picks the cutoff instead of guessing one
    _, binary = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary


def preprocess_pipeline(image_path):
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    gray = to_grayscale(original)
    cv2.imwrite(f"{OUTPUT_DIR}/step1_grayscale.png", gray)

    blurred = reduce_noise(gray)
    cv2.imwrite(f"{OUTPUT_DIR}/step2_blurred.png", blurred)

    deskewed = deskew(blurred)
    cv2.imwrite(f"{OUTPUT_DIR}/step3_deskewed.png", deskewed)

    binary = adaptive_threshold(deskewed)
    cv2.imwrite(f"{OUTPUT_DIR}/step4_threshold.png", binary)

    return original, binary


# ---------------------------------------------------------------------------
# STEP 2: Run OCR and apply the confidence gate
# ---------------------------------------------------------------------------
def run_ocr(binary_image, threshold=CONFIDENCE_THRESHOLD):
    data = pytesseract.image_to_data(
        binary_image, config="--psm 6", output_type=pytesseract.Output.DICT
    )

    kept_words = []
    dropped_words = []

    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        conf = float(data["conf"][i])

        if not word or conf < 0:  # tesseract uses -1 for non-text regions
            continue

        entry = {
            "text": word,
            "confidence": conf,
            "box": (data["left"][i], data["top"][i], data["width"][i], data["height"][i]),
        }

        if conf >= threshold:
            kept_words.append(entry)
        else:
            dropped_words.append(entry)

    return kept_words, dropped_words


# ---------------------------------------------------------------------------
# STEP 3: Draw the visual confirmation - only the words that cleared the gate
# ---------------------------------------------------------------------------
def annotate_image(binary_image, kept_words, dropped_words):
    annotated = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    for word in kept_words:
        x, y, w, h = word["box"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
        label = f"{word['text']} ({word['confidence']:.0f}%)"
        cv2.putText(
            annotated, label, (x, max(y - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2
        )

    for word in dropped_words:
        x, y, w, h = word["box"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 200), 1)

    cv2.imwrite(f"{OUTPUT_DIR}/final_annotated.png", annotated)


def display_summary(kept_words, dropped_words, threshold=CONFIDENCE_THRESHOLD):
    print("=" * 60)
    print("OCR RECOGNITION RESULTS")
    print("=" * 60)

    total = len(kept_words) + len(dropped_words)
    print(f"Total words detected : {total}")
    print(f"Passed >= {threshold}% confidence : {len(kept_words)}")
    print(f"Dropped (< {threshold}%)          : {len(dropped_words)}")

    if kept_words:
        avg_conf = sum(w["confidence"] for w in kept_words) / len(kept_words)
        print(f"Average confidence (kept)  : {avg_conf:.1f}%")

        print("\nRecognized text (confirmed):")
        recognized_line = " ".join(w["text"] for w in kept_words)
        print(f"  \"{recognized_line}\"")
    else:
        print("\nNothing cleared the confidence threshold.")

    if dropped_words:
        print(f"\nDropped as low-confidence noise: "
              f"{[w['text'] for w in dropped_words]}")

    print("\nSaved files:")
    print("  step1_grayscale.png, step2_blurred.png,")
    print("  step3_deskewed.png, step4_threshold.png, final_annotated.png")


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else None

    if image_path is None:
        print("No image path provided - generating a sample test image.\n")
        image_path = generate_sample_image()

    original, binary = preprocess_pipeline(image_path)
    kept_words, dropped_words = run_ocr(binary)
    annotate_image(binary, kept_words, dropped_words)
    display_summary(kept_words, dropped_words)


if __name__ == "__main__":
    main()
