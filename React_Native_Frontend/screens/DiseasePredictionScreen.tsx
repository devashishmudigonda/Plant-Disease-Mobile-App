import React, { useState } from 'react';
import {
  View,
  Text,
  Button,
  Image,
  ActivityIndicator,
  StyleSheet,
  Alert,
  ScrollView,
} from 'react-native';
import * as ImagePicker from 'react-native-image-picker';
import axios from 'axios';

const DiseasePredictionScreen: React.FC = () => {
  const [image, setImage] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<any>(null);

  const pickImage = () => {
    ImagePicker.launchImageLibrary({ mediaType: 'photo' }, (response) => {
      if (!response.didCancel && response.assets && response.assets.length > 0) {
        const selectedImage = response.assets[0];
        setImage(selectedImage);
      }
    });
  };

  const uploadImage = async () => {
    if (!image) {
      Alert.alert("Error", "Please select an image first");
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append('image', {
      uri: image.uri,
      type: image.type,
      name: image.fileName
    });

    try {
      // const response = await axios.post('http://127.0.0.1:5000/submit', formData, {
        // const response = await axios.post('http://192.168.0.139:5001/submit', formData, {
          const response = await axios.post('http://192.168.41.132:5001/submit', formData, {


        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setPrediction(response.data);
    } catch (error) {
      console.error("Upload Error:", error);
      Alert.alert("Upload Failed", "Failed to get prediction");
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView
      contentContainerStyle={styles.scrollContainer}
      keyboardShouldPersistTaps="handled"
    >
      <Text style={styles.title}>Plant Disease Detector</Text>

      {image && <Image source={{ uri: image.uri }} style={styles.uploadedImage} />}

      <Button title="📸 Pick an Image" onPress={pickImage} />
      <Button title="🔍 Upload & Predict" onPress={uploadImage} disabled={!image || loading} />

      {loading && <ActivityIndicator size="large" color="#00ff00" style={styles.loading} />}

      {prediction && (
        <View style={styles.resultContainer}>
          <Text style={styles.resultTitle}>🔬 Prediction Result</Text>
          <Text style={styles.resultText}><Text style={styles.boldText}>🦠 Disease: </Text>{prediction.title}</Text>
          <Text style={styles.resultText}><Text style={styles.boldText}>📖 Description: </Text>{prediction.description}</Text>
          <Text style={styles.resultText}><Text style={styles.boldText}>🛡 Prevention: </Text>{prediction.prevention}</Text>

          {/* Display Severity */}
          <Text style={styles.resultText}><Text style={styles.boldText}>📈 Severity: </Text>{prediction.severity}</Text>

          {prediction.image_url && (
            <Image source={{ uri: prediction.image_url }} style={styles.resultImage} />
          )}

          <Text style={styles.resultText}><Text style={styles.boldText}>💊 Supplement: </Text>{prediction.supplement_name}</Text>
          {prediction.supplement_image_url && (
            <Image source={{ uri: prediction.supplement_image_url }} style={styles.supplementImage} />
          )}

          <Text style={styles.buyText} onPress={() => Alert.alert("Redirecting", "Open in Browser!")}>
            🛒 Buy Here: {prediction.buy_link}
          </Text>
        </View>
      )}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  scrollContainer: {
    flexGrow: 1,
    padding: 20,
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 26,
    fontWeight: 'bold',
    marginBottom: 20,
    color: '#2c3e50',
    textAlign: 'center'
  },
  uploadedImage: {
    width: 250,
    height: 250,
    borderRadius: 10,
    marginBottom: 20,
    resizeMode: 'contain'
  },
  loading: {
    marginTop: 20
  },
  resultContainer: {
    marginTop: 30,
    padding: 20,
    backgroundColor: '#ffffff',
    borderRadius: 10,
    width: '100%',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 5,
    elevation: 3
  },
  resultTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    marginBottom: 15,
    color: '#16a085',
    textAlign: 'center'
  },
  resultText: {
    fontSize: 18,
    marginBottom: 10,
    color: '#2c3e50',
    textAlign: 'center'
  },
  boldText: {
    fontWeight: 'bold'
  },
  resultImage: {
    width: '100%',
    height: 200,
    borderRadius: 10,
    marginVertical: 15,
    resizeMode: 'contain'
  },
  supplementImage: {
    width: 150,
    height: 150,
    borderRadius: 10,
    marginVertical: 10,
    resizeMode: 'contain'
  },
  buyText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#e74c3c',
    textAlign: 'center',
    marginTop: 10,
    textDecorationLine: 'underline'
  },
});

export default DiseasePredictionScreen;
