// import React, { useState } from 'react';
// import { View, Text, Button, Image, ActivityIndicator, StyleSheet } from 'react-native';
// import * as ImagePicker from 'react-native-image-picker';
// import axios from 'axios';
// import { Alert } from 'react-native';


// const App: React.FC = () => {
//   const [image, setImage] = useState<any>(null);
//   const [loading, setLoading] = useState<boolean>(false);
//   const [prediction, setPrediction] = useState<any>(null);

//   // Function to pick an image from the device
//   const pickImage = () => {
//     ImagePicker.launchImageLibrary({ mediaType: 'photo' }, (response) => {
//       if (!response.didCancel && response.assets && response.assets.length > 0) {
//         const selectedImage = response.assets[0];
//         setImage(selectedImage);
//       }
//     });
//   };

//   // Function to upload image to Flask backend
//   const uploadImage = async () => {
//     if (!image) {
//       Alert.alert("Please select an image first");
//       return;
//     }

//     setLoading(true);

//     const formData = new FormData();
//     formData.append('image', {
//       uri: image.uri,
//       type: image.type,
//       name: image.fileName
//     });

//     try {
//       const response = await axios.post('http://127.0.0.1:5000/submit', formData, {
//         headers: { 'Content-Type': 'multipart/form-data' }
//       });

//       setPrediction(response.data);
//     } catch (error) {
//       console.error("Upload Error:", error);
//       Alert.alert("Failed to get prediction");
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <View style={styles.container}>
//       <Text style={styles.title}>Plant Disease Detection</Text>

//       {image && <Image source={{ uri: image.uri }} style={styles.image} />}
      
//       <Button title="Pick an Image" onPress={pickImage} />

//       <Button title="Upload & Predict" onPress={uploadImage} disabled={!image || loading} />

//       {loading && <ActivityIndicator size="large" color="#00ff00" />}

//       {prediction && (
//         <View style={styles.result}>
//           <Text style={styles.resultText}>Disease: {prediction.title}</Text>
//           <Text style={styles.resultText}>Description: {prediction.description}</Text>
//           <Text style={styles.resultText}>Prevention: {prediction.prevention}</Text>
//           <Image source={{ uri: prediction.image_url }} style={styles.image} />
//           <Text style={styles.resultText}>Supplement: {prediction.supplement_name}</Text>
//           <Image source={{ uri: prediction.supplement_image_url }} style={styles.image} />
//           <Text style={styles.resultText}>Buy Here: {prediction.buy_link}</Text>
//         </View>
//       )}
//     </View>
//   );
// };

// const styles = StyleSheet.create({
//   container: {
//     flex: 1,
//     justifyContent: 'center',
//     alignItems: 'center',
//     padding: 20,
//   },
//   title: {
//     fontSize: 24,
//     fontWeight: 'bold',
//     marginBottom: 20,
//   },
//   image: {
//     width: 200,
//     height: 200,
//     marginVertical: 10,
//   },
//   result: {
//     marginTop: 20,
//     padding: 10,
//     backgroundColor: '#eee',
//     borderRadius: 5,
//   },
//   resultText: {
//     fontSize: 16,
//     marginBottom: 5,
//   },
// });

// export default App;


// import React from 'react';
// import { View, Text, Button, StyleSheet } from 'react-native';
// import { NavigationContainer } from '@react-navigation/native';
// import { createStackNavigator, StackNavigationProp } from '@react-navigation/stack';
// import CropRecommenderScreen from './screens/CropRecommenderScreen';
// import DiseasePredictionScreen from './screens/DiseasePredictionScreen';
// import { RouteProp } from '@react-navigation/native';

// // ✅ Define the type for navigation
// type RootStackParamList = {
//   Home: undefined;
//   CropRecommender: undefined;
//   DiseasePrediction: undefined;
// };

// // ✅ Define props for HomeScreen
// type HomeScreenProps = {
//   navigation: StackNavigationProp<RootStackParamList, 'Home'>;
// };

// const Stack = createStackNavigator<RootStackParamList>();

// const HomeScreen: React.FC<HomeScreenProps> = ({ navigation }) => {
//   return (
//     <View style={styles.container}>
//       <Text style={styles.title}>🌾 Smart Farming App</Text>
//       <Button title="🌱 Crop Recommender" onPress={() => navigation.navigate('CropRecommender')} />
//       <Button title="🦠 Disease Prediction" onPress={() => navigation.navigate('DiseasePrediction')} />
//       <Button title="💊 Fertilizer Recommender" onPress={() => Alert.alert('Coming Soon!')} />
//     </View>
//   );
// };

// export default function App() {
//   return (
//     <NavigationContainer>
//       <Stack.Navigator>
//         <Stack.Screen name="Home" component={HomeScreen} />
//         <Stack.Screen name="CropRecommender" component={CropRecommenderScreen} />
//         <Stack.Screen name="DiseasePrediction" component={DiseasePredictionScreen} />
//       </Stack.Navigator>
//     </NavigationContainer>
//   );
// }

// const styles = StyleSheet.create({
//   container: {
//     flex: 1,
//     justifyContent: 'center',
//     alignItems: 'center',
//   },
//   title: {
//     fontSize: 24,
//     fontWeight: 'bold',
//     marginBottom: 20,
//   },
// });



import 'react-native-gesture-handler';
import React from 'react';
import { View, Text, Button, StyleSheet } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator, StackNavigationProp } from '@react-navigation/stack';
import CropRecommenderScreen from './screens/CropRecommenderScreen';
import DiseasePredictionScreen from './screens/DiseasePredictionScreen';
import FertilizerRecommenderScreen from './screens/FertilizerRecommenderScreen';
import { RouteProp } from '@react-navigation/native';

// ✅ Define navigation type
type RootStackParamList = {
  Home: undefined;
  CropRecommender: undefined;
  DiseasePrediction: undefined;
  FertilizerRecommender: undefined;
};

// ✅ Define props for HomeScreen
type HomeScreenProps = {
  navigation: StackNavigationProp<RootStackParamList, 'Home'>;
};

const Stack = createStackNavigator<RootStackParamList>();

// ✅ Explicitly type navigation in HomeScreen
const HomeScreen: React.FC<HomeScreenProps> = ({ navigation }) => {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>🌾 Smart Farming App</Text>
      <Button title="🌱 Crop Recommender" onPress={() => navigation.navigate('CropRecommender')} />
      <Button title="🦠 Disease Prediction" onPress={() => navigation.navigate('DiseasePrediction')} />
      <Button title="💊 Fertilizer Recommender" onPress={() => navigation.navigate('FertilizerRecommender')} />
    </View>
  );
};

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen name="Home" component={HomeScreen} />
        <Stack.Screen name="CropRecommender" component={CropRecommenderScreen} />
        <Stack.Screen name="DiseasePrediction" component={DiseasePredictionScreen} />
        <Stack.Screen name="FertilizerRecommender" component={FertilizerRecommenderScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
});
