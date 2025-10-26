import pytesseract
from PIL import Image
import joblib
import re

# --- Setup Tesseract path ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- Load trained pipeline ---
pipeline = joblib.load(r"D:\BVB\SMS\spam_pipeline.pkl")

# --- Text cleaning function ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '<URL>', text)
    text = re.sub(r'\S+@\S+', '<EMAIL>', text)
    text = re.sub(r'[^a-z\s<>]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- Ask user for input ---
choice = input("Enter '1' for image or '2' for text input: ")

if choice == '1':
    img_path = input("Enter image path: ")
    img = Image.open(img_path)
    text = pytesseract.image_to_string(img)
elif choice == '2':
    text = input("Enter the SMS text: ")
else:
    print("Invalid choice")
    exit()

text = clean_text(text)
pred = pipeline.predict([text])

print("\nExtracted / Input text:\n", text)
print("Prediction:", "Spam" if pred[0] == 1 else "Ham")
