# Dataset Preparation Instructions

## 1. Image Collection
- Collect high-resolution images of Indian cattle and buffalo breeds from diverse regions.
- Ensure images cover varied angles, lighting conditions, and backgrounds.
- Target at least 74 distinct breeds (cattle and buffalo).

## 2. Annotation Guidelines
- Label each image with:
  - Breed name
  - Species (cattle/buffalo)
  - Morphological traits:
    - Horn shape
    - Coat color
    - Ear type
- Use a standardized annotation format (e.g., CSV or JSON).

## 3. Data Augmentation
- Apply the following transformations to simulate real-world conditions:
  - Rotation
  - Zoom
  - Brightness adjustment
  - Horizontal and vertical flip
- Use augmentation scripts (see augmentation.py) to automate this process.

## 4. Quality Control
- Review annotated images for accuracy and consistency.
- Remove duplicates and low-quality samples.

---

Follow these guidelines to build a robust and diverse dataset for AI-powered cattle breed recognition.
