# 🔢 Digit Recognizer (MNIST) using Python and NumPy

This is a simple handwritten digit recognizer built from scratch using **NumPy** and **Flask**. It uses a custom-trained neural network on the **MNIST dataset**, and offers a clean web interface to upload and classify digit images in real time.

---

## 🌟 Highlights

- 🧠 **Built in under a day** using the power of modern **AI development tools** (including LLMs)
- 🚀 Created **without prior machine learning project experience**
- 🛠️ Model and logic implemented **entirely from scratch using NumPy** — no frameworks
- 🌐 Fully functional Flask frontend to upload, predict, and visualize results

---

## 📂 Project Structure

📁 digit-recogniser/
├── app.py # Flask server for UI and predictions
├── converter.py # Converts input image to flattened grayscale CSV
├── digit_recogniser_neural_network.py # Neural net logic and weights
├── model_weights.npz # Saved trained weights
├── train.csv # MNIST training data
├── test.csv # MNIST test data
├── limitations.txt # Notes on model constraints
├── working_principle_of_digit_recognition.txt # Math and logic explanation
├── templates/
│ └── index.html # Upload form interface
├── static/
│ └── uploads/
│ └── digit.png # Uploaded digit image (retained for reference)
└── MNIST images/ # Sample MNIST digit images

🚀 How It Works
User uploads a digit image via index.html (must be 28×28px, grayscale or black background).

app.py receives the image and saves it as static/uploads/digit.png.

converter.py:

Loads and preprocesses the image (resize, invert, normalize).

Saves it as a flattened CSV (input.csv).

digit_recogniser_neural_network.py:

Loads model_weights.npz

Reads the input.csv vector

Feeds it into the trained network to predict the digit

app.py then:

Sends the prediction back to index.html

Displays the uploaded image and guessed digit to the user

📌 Note: digit.png and input.csv are not deleted, so you can inspect them after each upload.
📉 Limitations
Works best with 28×28 grayscale images (similar to MNIST)

Accuracy is ~84.5% (basic 2-layer network, no regularization or dropout)

Prediction may fail or be inaccurate if:

The image has poor contrast

It's not centered

The stroke is too thin or too noisy

The model is trained only on digits 0–9 using the train.csv dataset

📚 Reference
Dataset: MNIST Digits

Neural network built entirely using NumPy, no ML frameworks

Frontend with HTML/CSS

Image processing with OpenCV

✍️ Author
Built by Rishav Dhara — powered by curiosity, creativity, and the capabilities of modern AI tools.
This was a solo project made possible in just one day with the assistance of AI-based development support.

