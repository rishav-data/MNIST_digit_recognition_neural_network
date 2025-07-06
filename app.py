from flask import Flask, request, render_template, after_this_request
import os
import time
import pandas as pd
from digit_recogniser_neural_network import load_model, make_predictions
from converter import convert_image

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file part"
    file = request.files['image']
    if file.filename == '':
        return "No selected file"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(UPLOAD_FOLDER, 'digit.png')
    input_csv = os.path.join(script_dir, "input.csv")

    try:
        # Save uploaded image
        file.save(img_path)
        print("📝 Image saved at:", img_path)
        print("📏 File exists:", os.path.exists(img_path))
        print("📦 File size:", os.path.getsize(img_path))

        # Wait for file write completion
        for _ in range(10):
            if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                break
            time.sleep(0.1)
        else:
            return "Error: Image not ready for processing."

        print(" Calling converter.py from app...")
        convert_image()

        # Predict
        input_data = pd.read_csv(input_csv, header=None).values.T
        W1, b1, W2, b2 = load_model()
        prediction = make_predictions(input_data, W1, b1, W2, b2)[0]

        @after_this_request
        def cleanup(response):
            try:
                if os.path.exists(input_csv):  #Only delete CSV
                    os.remove(input_csv)
            except Exception as cleanup_error:
                print("⚠️ Cleanup failed:", cleanup_error)
            return response

        return f"""
            <h3>Predicted Digit: {prediction}</h3><br>
            <img src="/static/uploads/digit.png" style="width: 100px; height: 100px;"><br>
            <a href='/'>Try Another</a>
        """

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
