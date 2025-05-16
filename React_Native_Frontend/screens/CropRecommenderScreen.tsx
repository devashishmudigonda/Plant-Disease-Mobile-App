import React, { useState } from 'react';
import { View, Text, TextInput, Button, StyleSheet, ScrollView, Alert } from 'react-native';
import axios from 'axios';

const CropRecommenderScreen = () => {
  // Define the expected types explicitly
  const [data, setData] = useState<{ N: string; P: string; K: string; temperature: string; humidity: string; pH: string; rainfall: string }>({
    N: '',
    P: '',
    K: '',
    temperature: '',
    humidity: '',
    pH: '',
    rainfall: '',
  });

  const [result, setResult] = useState<string | null>(null);

  const handleSubmit = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:5000/recommend_crop', data);
      setResult(response.data.recommended_crop);
    } catch (error) {
      Alert.alert("Error", "Failed to get recommendation.");
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>🌱 Enter Soil & Weather Data</Text>

      {/** ✅ Fix: Explicitly define key type as keyof typeof data */}
      {(Object.keys(data) as Array<keyof typeof data>).map((key) => (
        <TextInput
          key={key}
          style={styles.input}
          placeholder={key}
          keyboardType="numeric"
          value={data[key]}  // ✅ Now TypeScript knows this key is valid
          onChangeText={(text) => setData({ ...data, [key]: text })}
        />
      ))}

      <Button title="🔍 Get Recommendation" onPress={handleSubmit} />

      {result && <Text style={styles.result}>Recommended Crop: {result}</Text>}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: { flexGrow: 1, justifyContent: 'center', alignItems: 'center', padding: 20 },
  title: { fontSize: 22, fontWeight: 'bold', marginBottom: 20 },
  input: { width: '80%', height: 40, borderWidth: 1, marginBottom: 10, padding: 8, borderRadius: 5 },
  result: { fontSize: 18, marginTop: 20, fontWeight: 'bold' },
});

export default CropRecommenderScreen;
