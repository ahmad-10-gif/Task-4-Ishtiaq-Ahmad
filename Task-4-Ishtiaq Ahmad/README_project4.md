# Project 4 - Image/Text Recognition (Basic)

Internship project for DecodeLabs - optional mastery milestone. Went with
the OCR path (Path 1) rather than object detection, since pytesseract
wraps Tesseract directly and doesn't need a separate model file to be
downloaded, which made it much easier to get a fully working pipeline
running end to end.

## Files

- `image_text_recognition.py` - the main script
- `sample_input.png` - auto-generated if you don't pass your own image
  (simulates a slightly rotated, unevenly lit "scanned document")

## Pipeline

1. **Grayscale conversion** - collapses the image down to a single
   intensity channel, dropping color info that OCR doesn't need
2. **Gaussian blur** - smooths out sensor noise before thresholding picks
   it up as false detail
3. **Deskew** - detects the rotation angle of the text blob and straightens
   it back to horizontal (this matters more than it sounds - Tesseract's
   accuracy drops fast on tilted text)
4. **Adaptive thresholding (Otsu's method)** - converts grayscale into
   clean black-and-white so character edges are unambiguous
5. **OCR + confidence gate** - runs `pytesseract.image_to_data()` to get
   per-word text and confidence, then only keeps words scoring 80% or
   higher (anything below gets dropped, not displayed)
6. **Visual confirmation** - draws green boxes + confidence labels around
   every word that passed, saves it as `final_annotated.png`

## How to run

```bash
pip install opencv-python pytesseract pillow numpy
# tesseract-ocr must also be installed at the system level
python image_text_recognition.py                # uses generated sample
python image_text_recognition.py my_photo.jpg    # or your own image
```

Every run saves 5 images to the working folder so you can see each stage:
`step1_grayscale.png`, `step2_blurred.png`, `step3_deskewed.png`,
`step4_threshold.png`, `final_annotated.png`

## Sample output

```
Total words detected : 7
Passed >= 80% confidence : 7
Dropped (< 80%)          : 0
Average confidence (kept)  : 94.4%

Recognized text (confirmed):
  "DECODELABS Internship Project 4 OCR Recognition Test"
```

## Why the 80% gate matters

Without it, Tesseract will happily return garbage guesses for smudges or
noise with the same confidence formatting as real text. Filtering out
anything below 80% means what actually gets displayed is text the model
is genuinely confident about, at the cost of occasionally dropping a
real word that was too blurry to read clearly - a reasonable trade-off
for a milestone that's meant to prove the model's certainty, not just its
guesswork.

## Notes / things I'd try next

- Test against a real scanned document instead of a synthetic one to see
  how much harder actual paper texture and shadows are compared to
  simulated noise
- Try the object detection path (MobileNet-SSD) as a second recognition
  mode alongside OCR, so the same script can handle both text and
  physical objects
- Experiment with different Tesseract PSM (page segmentation) modes for
  different layouts - single line vs. block vs. sparse text
