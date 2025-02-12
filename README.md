# 🩺 Disease Prediction Using Symptoms

## 📌 Overview
This project utilizes **Machine Learning** to predict diseases based on input symptoms. The system is trained on a dataset of various symptoms and diseases, using a **Random Forest Classifier** to achieve high accuracy. The model supports real-time predictions and provides a ranked list of possible diseases.

## 🚀 Features
- **Preprocessing Module**: Cleans and prepares the dataset.
- **Model Training**: Trains a high-accuracy ML model.
- **Evaluation**: Generates accuracy and classification reports.
- **Real-time Prediction**: Predicts diseases based on user input.
- **Model Saving & Loading**: Saves the trained model for future use.

## 📂 Project Structure
```
📁 Disease-Prediction
│── 📂 data_preprocessing.py
│── 📂 model_training.py
│── 📂 evaluation.py
│── 📂 prediction.py
│── 📂 saved_model.pkl
│── 📄 requirements.txt
│── 📄 README.md
```

## 🛠️ Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/Arunkarthi123/Disease-Prediction.git
   cd Disease-Prediction
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the model training script:
   ```sh
   python model_training.py
   ```

## 🧪 Usage
- **To predict a disease:**
  ```sh
  python prediction.py
  ```
  Enter symptoms as prompted, and the model will return the most probable diseases.

## 📊 Evaluation
The trained model is evaluated using **accuracy score** and **classification report**.

## 📌 Technologies Used
- **Python**
- **Pandas**
- **Scikit-Learn**
- **Joblib**
- **NumPy**

## 🏆 Contribution
Feel free to contribute! Fork the repo, make your changes, and submit a pull request.

## 📜 License
This project is open-source and available under the MIT License.

