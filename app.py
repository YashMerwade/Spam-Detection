from flask import Flask, render_template, request, redirect, url_for
import pytesseract
from PIL import Image
import joblib
import re
import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load trained pipeline
pipeline = joblib.load(r"D:\BVB\sms\spam_pipeline.pkl")
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '<URL>', text)
    text = re.sub(r'\S+@\S+', '<EMAIL>', text)
    text = re.sub(r'[^a-z\s<>]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def highlight_spam_words(text):
    spam_words = ['bank', 'account', 'click', 'free', 'win', 'urgent', 
                  'prize', 'offer', 'suspended', 'activate', 'payment']
    words = text.split()
    highlighted = []
    for word in words:
        if word.lower() in spam_words:
            highlighted.append(f'<span class="highlight">{word}</span>')
        else:
            highlighted.append(word)
    return ' '.join(highlighted)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    highlighted_text = None
    extracted_text = ""
    uploaded_image_url = None

    if request.method == 'POST':
        sms_text = request.form.get('sms_text')
        sms_image = request.files.get('sms_image')

        # Prioritize image if uploaded
        if sms_image and sms_image.filename != '':
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            # Unique filename to avoid collisions
            unique_filename = str(uuid.uuid4()) + "_" + sms_image.filename
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            sms_image.save(img_path)
            uploaded_image_url = url_for('static', filename=f'uploads/{unique_filename}')
            img = Image.open(img_path)
            extracted_text = pytesseract.image_to_string(img)
        elif sms_text:
            extracted_text = sms_text

        extracted_text = clean_text(extracted_text)
        prediction = "Spam" if pipeline.predict([extracted_text])[0] == 1 else "Legitimate ✅"
        highlighted_text = highlight_spam_words(extracted_text)

    return render_template('index.html', prediction=prediction,
                           highlighted_text=highlighted_text,
                           uploaded_image_url=uploaded_image_url)

if __name__ == '__main__':
    app.run(debug=True)
