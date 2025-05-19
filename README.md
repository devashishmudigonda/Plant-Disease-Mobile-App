# 🌿 Plant Disease Detection Mobile App (iOS)

A powerful and intuitive mobile application for **real-time plant disease detection**, leveraging a **custom 2D CNN model** integrated into a **Flask backend** and a modern **React Native frontend** (iOS only). Designed to assist farmers, researchers, and agricultural experts with accurate diagnosis through a simple mobile interface.

---

## 🚀 Project Overview

- 🎯 **Goal**: Identify plant leaf diseases and provide results directly in the mobile app.
- 🤖 **Model**: Trained 2D CNN for multi-class plant disease classification.
- 🛠️ **Backend**: Flask API for model inference.
- 📱 **Frontend**: React Native app (iOS support).
- 🔬 **Deployment**: Currently supported on macOS via iOS simulator (Xcode).

---

## 🗂️ Dataset Used

We use the **PlantVillage** dataset, which contains thousands of annotated leaf images across multiple crops and diseases.

📎 **Download Dataset:**  
🔗 [https://data.mendeley.com/datasets/tywbtsjrjv/1](https://data.mendeley.com/datasets/tywbtsjrjv/1)

Make sure to preprocess the dataset as required for training before using it in the model pipeline.

---

## 🧾 Clone the Project

```bash
git clone https://github.com/devashishmudigonda/Plant-Disease-Mobile-App
```

---

## 📦 Install Dependencies

### 1. 📁 Navigate to the Project Root  
Make sure you're inside the cloned repository.

### 2. 📄 Install Python Requirements  
Run this in the terminal:
```bash
pip install -r requirements.txt
```

---

## ⚙️ Environment Setup

### 🧪 Create & Activate Virtual Environment

#### 🪟 On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### 🍎 On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 🔁 Run the Backend Server

Navigate to the Flask backend folder (usually named something like `Flask_Backend/` or `api/`):

```bash
cd Flask_Backend
python app.py
```

- This will start the Flask server at `http://localhost:5000`
- Ensure the model and weights are in the correct folder (usually inside `model/` or `static/`)

---

## 📱 Run the iOS Frontend

> ⚠️ Only supported on macOS with Xcode installed.

### 1. Navigate to the React Native folder:
```bash
cd React_Native_FrontEnd
```

### 2. Run the iOS app in the simulator:
```bash
npx react-native run-ios
```

This will open the iOS simulator and launch the mobile application connected to your local Flask server.

---

## 💡 Features

- 📷 **Upload plant leaf image directly from mobile**
- 🔎 **Detects disease using CNN model**
- 🧪 **Returns disease name and confidence score**
- 🧬 **Backend processing with Flask API**
- 📱 **Smooth and native iOS experience**

---

## 📂 Project Structure

```bash
Plant-Disease-Mobile-App/
│
├── Flask Deployed App/               # Python + Flask server
│   ├── app.py                   # Main backend script
│   ├── CNN.py                   # Trained CNN model files
│   └── ...                      # Utils, static, etc.
│
├── React_Native_FrontEnd/      # Mobile app (React Native)
│   ├── App.js                   # Entry point for app
│   └── ...                      # Screens, assets, etc.
│
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 👨‍💻 Author

**Devashish Mudigonda**  
📬 [GitHub](https://github.com/devashishmudigonda)

---

## 🤝 Contributions

Pull requests and issues are welcome! Please open an issue to discuss changes before submitting a PR.

---

## 📸 Demo

Below are key outputs and visual results from the project:

### 📱 Mobile App Output
![Mobile Output](demo/mobile_output.png)

### 📊 Confusion Matrix
![Confusion Matrix](demo/confusion_matrix.png)

### 📉 Training Loss Graph
![Loss Graph](demo/loss_graph.png)


### 🧪 Notebook Inference Output
![Notebook Output](demo/notebook_output.png)

### 🌿 Severity Estimation Calculation
![Severity Calculation](demo/severity_calc.png)

---

