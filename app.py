import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import joblib
import os
from sklearn.base import BaseEstimator

# Create a wrapper class for older scikit-learn models
class ModelCompatibilityWrapper(BaseEstimator):
    def __init__(self, model):
        self.model = model
    
    def predict(self, X):
        # Handle the prediction without using newer attributes
        try:
            return self.model.predict(X)
        except AttributeError:
            # If the model is a tree-based model
            if hasattr(self.model, 'estimators_'):
                predictions = []
                for tree in self.model.estimators_:
                    pred = tree.predict(X)
                    predictions.append(pred)
                return np.round(np.mean(predictions, axis=0))
            # If it's a single tree
            return self.model.predict(X)

# Load and wrap models
try:
    diabetes_model = ModelCompatibilityWrapper(joblib.load('diabetespred_model.sav'))
    heart_disease_model = ModelCompatibilityWrapper(joblib.load('heart_disease_model.sav'))
    parkinsons_model = ModelCompatibilityWrapper(joblib.load('parkinsons_model.sav'))
    diabetes_scaler = joblib.load('diabetes_scaler.sav')
    heart_scaler = joblib.load('heart_scaler.sav')
    parkinsons_scaler = joblib.load('parkinsons_scaler.sav')
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                          ['Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Parkinsons Prediction'],
                          icons=['activity', 'heart', 'person'],
                          default_index=0)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    # page title
    st.title('Diabetes Prediction using ML')
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
        
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
        
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
        
    with col2:
        Insulin = st.text_input('Insulin Level')
        
    with col3:
        BMI = st.text_input('BMI value')
        
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
        
    with col2:
        Age = st.text_input('Age of the Person')
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    if st.button('Diabetes Test Result'):
        try:
            # Convert input to float and validate
            user_input = [float(Pregnancies), float(Glucose), float(BloodPressure), 
                         float(SkinThickness), float(Insulin), float(BMI), 
                         float(DiabetesPedigreeFunction), float(Age)]
            
            # Validate input ranges
            if any(x < 0 for x in user_input):
                st.error("All values must be non-negative")
                st.stop()
            
            # Scale the features
            features_scaled = diabetes_scaler.transform([user_input])
            
            # Convert to numpy array if needed
            features_scaled = np.array(features_scaled)
            
            # Ensure correct shape
            if len(features_scaled.shape) == 1:
                features_scaled = features_scaled.reshape(1, -1)
            
            try:
                diab_prediction = diabetes_model.predict(features_scaled)
                if diab_prediction[0] == 1:
                    diab_diagnosis = 'The person is diabetic'
                else:
                    diab_diagnosis = 'The person is not diabetic'
                st.success(diab_diagnosis)
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                
        except ValueError:
            st.error("Please enter valid numerical values for all fields")

# Heart Disease Prediction Page
elif selected == 'Heart Disease Prediction':
    # page title
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('Chest Pain types') 
                       # (['Typical Angina', 'Atypical Angina', 
                         # 'Non-anginal Pain', 'Asymptomatic'])
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        #, ['Yes', 'No'])
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
                         #,
                         #  ['Upsloping', 'Flat', 'Downsloping'])
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('Thalassemia')
        #,
                      #     ['Normal', 'Fixed Defect', 'Reversible Defect'])
    
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    if st.button('Heart Disease Test Result'):
        try:
            user_input = [float(x) for x in [age, sex, cp, trestbps, chol, fbs, 
                         restecg, thalach, exang, oldpeak, slope, ca, thal]]
            
            # Scale the features if needed
            try:
                features_scaled = heart_scaler.transform([user_input])
                heart_prediction = heart_disease_model.predict(features_scaled)
            except:
                # If scaling fails, try without scaling
                heart_prediction = heart_disease_model.predict([user_input])
            
            if heart_prediction[0] == 1:
                heart_diagnosis = 'The person is having heart disease'
            else:
                heart_diagnosis = 'The person does not have any heart disease'
            st.success(heart_diagnosis)
            
        except ValueError:
            st.error("Please enter valid numerical values for all fields")
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            
# Parkinsons Prediction Page
else:
    # page title
    st.title('Parkinsons Prediction using ML')
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        try:
            features = [float(x) for x in [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                       RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA,
                       NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]]
            
            # Scale the features
            features_scaled = parkinsons_scaler.transform([features])
            
            # Ensure correct shape
            features_scaled = np.array(features_scaled)
            if len(features_scaled.shape) == 1:
                features_scaled = features_scaled.reshape(1, -1)
            
            try:
                parkinsons_prediction = parkinsons_model.predict(features_scaled)
                if parkinsons_prediction[0] == 1:
                    parkinsons_diagnosis = "The person has Parkinson's disease"
                else:
                    parkinsons_diagnosis = "The person does not have Parkinson's disease"
                st.success(parkinsons_diagnosis)
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                
        except ValueError:
            st.error("Please enter valid numerical values for all fields")
