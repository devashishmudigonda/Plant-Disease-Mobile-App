# import os
# from flask import Flask, redirect, render_template, request
# from PIL import Image
# import torchvision.transforms.functional as TF
# import CNN
# import numpy as np
# import torch
# import pandas as pd


# disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
# supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

# model = CNN.CNN(39)    
# model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
# model.eval()

# def prediction(image_path):
#     image = Image.open(image_path)
#     image = image.resize((224, 224))
#     input_data = TF.to_tensor(image)
#     input_data = input_data.view((-1, 3, 224, 224))
#     output = model(input_data)
#     output = output.detach().numpy()
#     index = np.argmax(output)
#     return index


# app = Flask(__name__)

# @app.route('/')
# def home_page():
#     return render_template('home.html')

# @app.route('/contact')
# def contact():
#     return render_template('contact-us.html')

# @app.route('/index')
# def ai_engine_page():
#     return render_template('index.html')

# @app.route('/mobile-device')
# def mobile_device_detected_page():
#     return render_template('mobile-device.html')

# @app.route('/submit', methods=['GET', 'POST'])
# def submit():
#     if request.method == 'POST':
#         image = request.files['image']
#         filename = image.filename
#         file_path = os.path.join('static/uploads', filename)
#         image.save(file_path)
#         print(file_path)
#         pred = prediction(file_path)
#         title = disease_info['disease_name'][pred]
#         description =disease_info['description'][pred]
#         prevent = disease_info['Possible Steps'][pred]
#         image_url = disease_info['image_url'][pred]
#         supplement_name = supplement_info['supplement name'][pred]
#         supplement_image_url = supplement_info['supplement image'][pred]
#         supplement_buy_link = supplement_info['buy link'][pred]
#         return render_template('submit.html' , title = title , desc = description , prevent = prevent , 
#                                image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)

# @app.route('/market', methods=['GET', 'POST'])
# def market():
#     return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
#                            supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))

# if __name__ == '__main__':
#     app.run(debug=True)





# import os
# from flask import Flask, redirect, render_template, request
# from PIL import Image
# import torchvision.transforms.functional as TF
# import CNN  # Import the modified CNN model with severity calculation
# import numpy as np
# import torch
# import pandas as pd
# import cv2

# # Load disease and supplement information
# disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
# supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# # Load the CNN model
# model = CNN.CNN(39)  # 39 disease classes
# model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
# model.eval()

# # Prediction function for disease type
# def predict_disease(image_path):
#     image = Image.open(image_path)
#     image = image.resize((224, 224))
#     input_data = TF.to_tensor(image)
#     input_data = input_data.view((-1, 3, 224, 224))
#     output = model(input_data)
#     output = output.detach().numpy()
#     disease_index = np.argmax(output)
#     return disease_index

# # Severity calculation using OpenCV
# def calculate_severity(image_path):
#     # Load the image using OpenCV
#     image = cv2.imread(image_path)

#     # Convert to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Thresholding to isolate diseased areas (assumed to be darker pixels)
#     _, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)

#     # Calculate total pixels in the image
#     total_pixels = np.size(thresholded_image)

#     # Calculate number of diseased pixels (white pixels in the thresholded image)
#     diseased_pixels = np.count_nonzero(thresholded_image)

#     # Calculate severity percentage
#     severity_percentage = (diseased_pixels / total_pixels) * 100
#     return severity_percentage

# # Initialize Flask app
# app = Flask(__name__)

# @app.route('/')
# def home_page():
#     return render_template('home.html')

# @app.route('/contact')
# def contact():
#     return render_template('contact-us.html')

# @app.route('/index')
# def ai_engine_page():
#     return render_template('index.html')

# @app.route('/mobile-device')
# def mobile_device_detected_page():
#     return render_template('mobile-device.html')

# # Update the submit route to handle the healthy plant case
# @app.route('/submit', methods=['GET', 'POST'])
# def submit():
#     if request.method == 'POST':
#         # Get the uploaded image
#         image = request.files['image']
#         filename = image.filename
#         file_path = os.path.join('static/uploads', filename)
#         image.save(file_path)

#         # Predict disease type
#         disease_pred = predict_disease(file_path)

#         # Calculate severity (N/A for healthy plants)
#         severity = model.calculate_severity(file_path, disease_pred)

#         # Fetch details from the CSV based on the predicted disease index
#         title = disease_info['disease_name'][disease_pred]
#         description = disease_info['description'][disease_pred]
#         prevent = disease_info['Possible Steps'][disease_pred]
#         image_url = disease_info['image_url'][disease_pred]

#         # Fetch supplement info
#         supplement_name = supplement_info['supplement name'][disease_pred]
#         supplement_image_url = supplement_info['supplement image'][disease_pred]
#         supplement_buy_link = supplement_info['buy link'][disease_pred]

#         # Render the result with all details including severity
#         return render_template('submit.html',
#                                title=title,
#                                desc=description,
#                                prevent=prevent,
#                                image_url=image_url,
#                                severity=severity,  # Display the severity percentage or N/A
#                                sname=supplement_name,
#                                simage=supplement_image_url,
#                                buy_link=supplement_buy_link)


# @app.route('/market', methods=['GET', 'POST'])
# def market():
#     return render_template('market.html',
#                            supplement_image=list(supplement_info['supplement image']),
#                            supplement_name=list(supplement_info['supplement name']),
#                            disease=list(disease_info['disease_name']),
#                            buy=list(supplement_info['buy link']))

# if __name__ == '__main__':
#     app.run(debug=True)


# import os
# from flask import Flask, request, jsonify
# from PIL import Image
# import torchvision.transforms.functional as TF
# import CNN
# import numpy as np
# import torch
# import pandas as pd

# # Load disease and supplement data
# disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
# supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# # Load the model
# model = CNN.CNN(39)
# model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
# model.eval()

# def prediction(image_path):
#     """Predict the disease based on the uploaded image"""
#     image = Image.open(image_path)
#     image = image.resize((224, 224))
#     input_data = TF.to_tensor(image)
#     input_data = input_data.view((-1, 3, 224, 224))
#     output = model(input_data)
#     output = output.detach().numpy()
#     index = np.argmax(output)
#     return index

# app = Flask(__name__)

# @app.route('/submit', methods=['POST'])
# def submit():
#     """Handle image upload and return prediction results as JSON"""
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image file uploaded'}), 400
    
#     image = request.files['image']
#     filename = image.filename
#     file_path = os.path.join('static/uploads', filename)
#     image.save(file_path)

#     pred = prediction(file_path)

#     result = {
#         "title": disease_info['disease_name'][pred],
#         "description": disease_info['description'][pred],
#         "prevention": disease_info['Possible Steps'][pred],
#         "image_url": disease_info['image_url'][pred],
#         "supplement_name": supplement_info['supplement name'][pred],
#         "supplement_image_url": supplement_info['supplement image'][pred],
#         "buy_link": supplement_info['buy link'][pred]
#     }
    
#     return jsonify(result)

# if __name__ == '__main__':
#     app.run(debug=True)


# import os
# from flask import Flask, request, jsonify
# import pandas as pd
# import numpy as np
# import joblib
# from flask_cors import CORS
# from PIL import Image
# import torchvision.transforms.functional as TF
# import CNN
# import torch
# from sklearn.preprocessing import StandardScaler

# # Initialize Flask App
# app = Flask(__name__)
# CORS(app)  # ✅ Allow React Native to access Flask API

# # Load Models
# crop_model = joblib.load("crop_recommendation_model.pkl")
# fertilizer_model = joblib.load("fertilizer_recommendation_model.pkl")
# fertilizer_encoders = joblib.load("fertilizer_label_encoders.pkl")  # Load label encoders

# # Load Disease Model
# model = CNN.CNN(39)
# model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
# model.eval()

# # Load Disease and Supplement Data
# disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
# supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# # ✅ **Crop Recommendation API**
# @app.route('/recommend_crop', methods=['POST'])
# def recommend_crop():
#     """Predicts the best crop based on soil and weather conditions."""
#     try:
#         data = request.get_json()
#         features = np.array([[data['N'], data['P'], data['K'], data['temperature'],
#                               data['humidity'], data['pH'], data['rainfall']]])
        
#         # Normalize input data (if required)
#         scaler = StandardScaler()
#         features_scaled = scaler.fit_transform(features)

#         prediction = crop_model.predict(features_scaled)[0]
#         return jsonify({"recommended_crop": prediction})
    
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

# # ✅ **Fertilizer Recommendation API**
# @app.route('/recommend_fertilizer', methods=['POST'])
# def recommend_fertilizer():
#     """Predicts the best fertilizer based on soil type, crop type, and integer soil nutrients."""
#     try:
#         data = request.get_json()
#         print("🚀 Received Data:", data)
#         print("🟢 Model Feature Names:", fertilizer_model.feature_names_in_)

#         # Validate required fields (Matching Model Feature Names)
#         required_fields = ["soil_type", "crop_type", "moisture", "nitrogen", "phosphorous", "potassium", "humidity", "temparature"]
#         for field in required_fields:
#             if field not in data:
#                 return jsonify({"error": f"Missing field: {field}"}), 400
        
#         # Ensure categorical inputs exist in trained model
#         if data["soil_type"] not in fertilizer_encoders["Soil Type"].classes_:
#             return jsonify({"error": f"Invalid soil type: {data['soil_type']}. Expected one of {list(fertilizer_encoders['Soil Type'].classes_)}"}), 400
#         if data["crop_type"] not in fertilizer_encoders["Crop Type"].classes_:
#             return jsonify({"error": f"Invalid crop type: {data['crop_type']}. Expected one of {list(fertilizer_encoders['Crop Type'].classes_)}"}), 400

#         # Encode categorical inputs (Soil Type & Crop Type)
#         soil_type_encoded = fertilizer_encoders["Soil Type"].transform([data["soil_type"]])[0]
#         crop_type_encoded = fertilizer_encoders["Crop Type"].transform([data["crop_type"]])[0]

#         # Convert all numeric inputs
#         try:
#             moisture = int(data["moisture"])
#             nitrogen = int(data["nitrogen"])
#             phosphorous = int(data["phosphorous"])
#             potassium = int(data["potassium"])
#             humidity = int(data["humidity"])
#             temparature = int(data["temparature"])
#         except ValueError:
#             return jsonify({"error": "All numeric fields must be integers."}), 400

#         # ✅ Match column names exactly (fix trailing spaces)
#         feature_columns = ["Temparature", "Humidity ", "Moisture", "Soil Type", "Crop Type", "Nitrogen", "Potassium", "Phosphorous"]
#         features = pd.DataFrame([[temparature, humidity, moisture, soil_type_encoded, crop_type_encoded, nitrogen, potassium, phosphorous]],
#                                 columns=feature_columns)

#         print("✅ Features for Prediction:", features)  # ✅ Debugging

#         # Predict fertilizer
#         prediction_encoded = fertilizer_model.predict(features)[0]

#         # Decode the prediction
#         predicted_fertilizer = fertilizer_encoders["Fertilizer Name"].inverse_transform([prediction_encoded])[0]

#         return jsonify({"recommended_fertilizer": predicted_fertilizer})
    
#     except Exception as e:
#         print("❌ API Error:", str(e))  # ✅ Log full error message in Flask
#         return jsonify({"error": str(e)}), 500



# # ✅ **Plant Disease Prediction API**
# def prediction(image_path):
#     """Predicts the plant disease from an uploaded image."""
#     image = Image.open(image_path)
#     image = image.resize((224, 224))
#     input_data = TF.to_tensor(image)
#     input_data = input_data.view((-1, 3, 224, 224))
    
#     output = model(input_data)
#     output = output.detach().numpy()
#     index = np.argmax(output)

#     return index

# @app.route('/submit', methods=['POST'])
# def submit():
#     """Handles image upload and returns plant disease prediction."""
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image file uploaded'}), 400
    
#     image = request.files['image']
#     filename = image.filename
#     file_path = os.path.join('static/uploads', filename)
#     image.save(file_path)

#     pred = prediction(file_path)

#     result = {
#         "title": disease_info['disease_name'][pred],
#         "description": disease_info['description'][pred],
#         "prevention": disease_info['Possible Steps'][pred],
#         "image_url": disease_info['image_url'][pred],
#         "supplement_name": supplement_info['supplement name'][pred],
#         "supplement_image_url": supplement_info['supplement image'][pred],
#         "buy_link": supplement_info['buy link'][pred]
#     }
    
#     return jsonify(result)

# if __name__ == '__main__':
#     app.run(debug=True)


import os
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from flask_cors import CORS
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import torch
import cv2
from sklearn.preprocessing import StandardScaler

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Load Models
crop_model = joblib.load("crop_recommendation_model.pkl")
fertilizer_model = joblib.load("fertilizer_recommendation_model.pkl")
fertilizer_encoders = joblib.load("fertilizer_label_encoders.pkl")

# Load CNN Disease Model
model = CNN.CNN(39)
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

# Load CSVs
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

#  Disease Prediction Function
def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

# Severity Estimation Function (only for diseased leaves)
def calculate_severity(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    total_pixels = cleaned.size
    diseased_pixels = cv2.countNonZero(cleaned)
    severity_percentage = (diseased_pixels / total_pixels) * 100
    return round(severity_percentage, 2)

# Plant Disease Prediction API
@app.route('/submit', methods=['POST'])
def submit():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400
    
    image = request.files['image']
    filename = image.filename
    file_path = os.path.join('static/uploads', filename)
    image.save(file_path)

    pred = prediction(file_path)
    disease_name = disease_info['disease_name'][pred]

    # Severity only if it's a diseased leaf
    if "healthy" in disease_name.lower() or "leaf" not in disease_name.lower():
        severity = "N/A"
    else:
        severity = f"{calculate_severity(file_path)}%"

    result = {
        "title": disease_name,
        "description": disease_info['description'][pred],
        "prevention": disease_info['Possible Steps'][pred],
        "image_url": disease_info['image_url'][pred],
        "supplement_name": supplement_info['supplement name'][pred],
        "supplement_image_url": supplement_info['supplement image'][pred],
        "buy_link": supplement_info['buy link'][pred],
        "severity": severity 
    }
    
    return jsonify(result)

# Crop Recommendation Endpoint (no change)
@app.route('/recommend_crop', methods=['POST'])
def recommend_crop():
    try:
        data = request.get_json()
        features = np.array([[data['N'], data['P'], data['K'], data['temperature'],
                              data['humidity'], data['pH'], data['rainfall']]])
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        prediction = crop_model.predict(features_scaled)[0]
        return jsonify({"recommended_crop": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Fertilizer Recommendation Endpoint (no change)
@app.route('/recommend_fertilizer', methods=['POST'])
def recommend_fertilizer():
    try:
        data = request.get_json()
        required_fields = ["soil_type", "crop_type", "moisture", "nitrogen", "phosphorous", "potassium", "humidity", "temparature"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        if data["soil_type"] not in fertilizer_encoders["Soil Type"].classes_:
            return jsonify({"error": f"Invalid soil type: {data['soil_type']}"})
        if data["crop_type"] not in fertilizer_encoders["Crop Type"].classes_:
            return jsonify({"error": f"Invalid crop type: {data['crop_type']}"})

        soil_type_encoded = fertilizer_encoders["Soil Type"].transform([data["soil_type"]])[0]
        crop_type_encoded = fertilizer_encoders["Crop Type"].transform([data["crop_type"]])[0]

        moisture = int(data["moisture"])
        nitrogen = int(data["nitrogen"])
        phosphorous = int(data["phosphorous"])
        potassium = int(data["potassium"])
        humidity = int(data["humidity"])
        temparature = int(data["temparature"])

        feature_columns = ["Temparature", "Humidity ", "Moisture", "Soil Type", "Crop Type", "Nitrogen", "Potassium", "Phosphorous"]
        features = pd.DataFrame([[temparature, humidity, moisture, soil_type_encoded, crop_type_encoded, nitrogen, potassium, phosphorous]],
                                columns=feature_columns)

        prediction_encoded = fertilizer_model.predict(features)[0]
        predicted_fertilizer = fertilizer_encoders["Fertilizer Name"].inverse_transform([prediction_encoded])[0]

        return jsonify({"recommended_fertilizer": predicted_fertilizer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
