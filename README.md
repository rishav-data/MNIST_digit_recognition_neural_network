# 🔢 Digit Recognizer (MNIST) using Python and NumPy

This is a simple handwritten digit recognizer built from scratch using **NumPy** and **Flask**. It uses a custom-trained neural network on the **MNIST dataset**, and offers a clean web interface to upload and classify digit images in real time.

<img width="1047" height="730" alt="fr" src="https://github.com/user-attachments/assets/7d121035-6263-4364-a448-a22075308d77" />


---

## 🌟 Highlights

- 🧠 **Built in under a day** using the power of modern **AI development tools** (including LLMs)
- 🚀 Created **without prior machine learning project experience**
- 🛠️ Model and logic implemented **entirely from scratch using NumPy** — no frameworks
- 🌐 Fully functional Flask frontend to upload, predict, and visualize results

---

## 📂 Project Structure

<img width="652" height="390" alt="structure" src="https://github.com/user-attachments/assets/1217d25d-9292-49e6-943b-27a19461ac58" />


## 🚀 How It Works
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


<img width="277" height="282" alt="res" src="https://github.com/user-attachments/assets/31e1d9b4-388c-4fa5-bcb7-711ac56db537" />

## 📌 Note: digit.png and input.csv are not deleted, so you can inspect them after each upload.
## 📉 Limitations
Works best with 28×28 grayscale images (similar to MNIST)

Accuracy is ~84.5% (basic 2-layer network, no regularization or dropout) for training data but drops significantly for images I made on MS Paint to test it .

Prediction may fail or be inaccurate if:

The image has poor contrast

It's not centered

The stroke is too thin or too noisy

The model is trained only on digits 0–9 using the train.csv dataset

<img width="1903" height="752" alt="traindat" src="https://github.com/user-attachments/assets/ccbcf258-47a8-45b8-8686-e584f43b005c" />


## 📚 Reference
Dataset: MNIST Digits

<img width="655" height="325" alt="MNIST_dataset_example" src="https://github.com/user-attachments/assets/45c0304f-1cbe-46ff-9633-9dd87cb162fd" />

Neural network built entirely using NumPy, no ML frameworks

<img width="1092" height="662" alt="conv" src="https://github.com/user-attachments/assets/912fbcb1-1d69-4e6d-b00e-0f0a4ebdda2a" />

<img width="702" height="387" alt="digitrecognn" src="https://github.com/user-attachments/assets/0038bd54-bd3e-44ac-bc94-52fe5d610206" />

Frontend with HTML/CSS

Image processing with OpenCV

## ✍️ Author
Built by Rishav Dhara — powered by curiosity, creativity, and the capabilities of modern AI tools.
This was a solo project made possible in just one day with the assistance of AI-based development support.

PLEASE NOTE THAT THE MODEL IS TRAINED ON train.csv AND test.csv . THESE TWO FILES WERE WAY TOO BIG TO JUSTIFY KEEPING THEM ESPECIALLY WHEN RENDER CAN'T DEPLOY THE BUILD DUE TO THEM . THE TRAINING DATA IS EASILY AVAIALABLE ON KAGGLE . 

Deployed Neural Network : https://mnist-digit-recognition-neural-network.onrender.com/
