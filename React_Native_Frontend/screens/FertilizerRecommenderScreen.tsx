import React, { useState } from 'react';
import { View, Text, TextInput, Button, StyleSheet, ScrollView, Alert } from 'react-native';
import axios from 'axios';

const FertilizerRecommenderScreen = () => {
  const [data, setData] = useState({
    soil_type: '',
    crop_type: '',
    moisture: '',
    nitrogen: '',
    phosphorus: '',
    potassium: '',
    humidity: '',     // ✅ Added missing field
    temperature: ''   // ✅ Added missing field
  });

  const [result, setResult] = useState<string | null>(null); // ✅ Added missing state

  const handleSubmit = async () => {
    try {
      const formattedData = {
        soil_type: data.soil_type.trim(),
        crop_type: data.crop_type.trim(),
        moisture: parseInt(data.moisture) || 0,
        nitrogen: parseInt(data.nitrogen) || 0,
        phosphorous: parseInt(data.phosphorus) || 0,  // ✅ Correct spelling for Flask
        potassium: parseInt(data.potassium) || 0,
        humidity: parseInt(data.humidity) || 0,       // ✅ Now included
        temparature: parseInt(data.temperature) || 0  // ✅ Correct spelling for Flask
      };

      console.log("🚀 Sending Data to Flask:", formattedData);

      const response = await axios.post('http://127.0.0.1:5000/recommend_fertilizer', formattedData);
      
      console.log("✅ API Response:", response.data);
      setResult(response.data.recommended_fertilizer);  // ✅ Store API response

    } catch (error: any) {
      console.error("❌ Fertilizer API Error:", error.response?.data || error.message);
      Alert.alert("Error", `Failed to get fertilizer recommendation.\n\n${error.response?.data?.error || error.message}`);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>💊 Enter Soil & Crop Data</Text>

      <TextInput style={styles.input} placeholder="Soil Type (Clayey, Sandy, etc.)" value={data.soil_type} onChangeText={(text) => setData({ ...data, soil_type: text })} />
      <TextInput style={styles.input} placeholder="Crop Type (Wheat, Rice, etc.)" value={data.crop_type} onChangeText={(text) => setData({ ...data, crop_type: text })} />

      {/* Integer-Only Inputs */}
      <TextInput style={styles.input} placeholder="Moisture (%)" keyboardType="numeric" value={data.moisture} onChangeText={(text) => setData({ ...data, moisture: text })} />
      <TextInput style={styles.input} placeholder="Nitrogen (N) Content" keyboardType="numeric" value={data.nitrogen} onChangeText={(text) => setData({ ...data, nitrogen: text })} />
      <TextInput style={styles.input} placeholder="Phosphorus (P) Content" keyboardType="numeric" value={data.phosphorus} onChangeText={(text) => setData({ ...data, phosphorus: text })} />
      <TextInput style={styles.input} placeholder="Potassium (K) Content" keyboardType="numeric" value={data.potassium} onChangeText={(text) => setData({ ...data, potassium: text })} />

      {/* ✅ New Input Fields for Missing Data */}
      <TextInput style={styles.input} placeholder="Humidity (%)" keyboardType="numeric" value={data.humidity} onChangeText={(text) => setData({ ...data, humidity: text })} />
      <TextInput style={styles.input} placeholder="Temperature (°C)" keyboardType="numeric" value={data.temperature} onChangeText={(text) => setData({ ...data, temperature: text })} />

      <Button title="🔍 Get Recommendation" onPress={handleSubmit} />

      {/* ✅ Display the result if it's available */}
      {result && <Text style={styles.result}>Recommended Fertilizer: {result}</Text>}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: { flexGrow: 1, justifyContent: 'center', alignItems: 'center', padding: 20 },
  title: { fontSize: 22, fontWeight: 'bold', marginBottom: 20 },
  input: { width: '80%', height: 40, borderWidth: 1, marginBottom: 10, padding: 8, borderRadius: 5 },
  result: { fontSize: 18, marginTop: 20, fontWeight: 'bold', color: 'green' }, // ✅ Style result text
});

export default FertilizerRecommenderScreen;
