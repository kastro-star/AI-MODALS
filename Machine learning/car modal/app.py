import pandas as pd 
import numpy as np 
import pickle as pk 
import streamlit as st

# Load model
model = pk.load(open('model.pkl','rb'))

# Set up page header
st.header('Car Price Prediction ML Model')

# Load and preprocess data
cars_data = pd.read_csv('Cardetails.csv')

def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()
cars_data['name'] = cars_data['name'].apply(get_brand_name)

# Input fields
## Basic car details
name = st.selectbox('Select Car Brand', cars_data['name'].unique())
year = st.slider('Car Manufactured Year', 1994, 2024)
km_driven = st.slider('No of kms Driven', 11, 200000)
fuel = st.selectbox('Fuel type', cars_data['fuel'].unique())
seller_type = st.selectbox('Seller  type', cars_data['seller_type'].unique())
transmission = st.selectbox('Transmission type', cars_data['transmission'].unique())
owner = st.selectbox('Seller  type', cars_data['owner'].unique())
mileage = st.slider('Car Mileage', 10, 40)
engine = st.slider('Engine CC', 700, 5000)
max_power = st.slider('Max Power', 0, 200)
seats = st.slider('No of Seats', 5, 10)

## Scratch details
num_scratches = st.number_input('Number of scratches', min_value=0, max_value=10, value=0, step=1)

scratch_lengths = []
scratch_costs = []
if num_scratches > 0:
    st.write("Enter scratch lengths (in cm):")
    for i in range(num_scratches):
        scratch_length = st.number_input(f'Length of scratch {i+1}', min_value=0.0, max_value=100.0, value=0.0, step=0.5)
        scratch_lengths.append(scratch_length)
        # Calculate cost for each scratch (500 for 5cm, decreasing for longer scratches)
        scratch_cost = 500 * (5/scratch_length) if scratch_length > 5 else 500
        scratch_costs.append(scratch_cost)

total_scratch_length = sum(scratch_lengths)
total_scratch_cost = sum(scratch_costs)

# Prediction logic
if st.button("Predict"):
    # Prepare input data
    input_data_model = pd.DataFrame(
        [[name,year,km_driven,fuel,seller_type,transmission,owner,mileage,engine,max_power,seats,num_scratches,total_scratch_length]],
        columns=['name','year','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats','num_scratches','total_scratch_length']
    )
    
    # Encode categorical variables
    category_mappings = {
        'owner': {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4, 'Test Drive Car': 5},
        'fuel': {'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4},
        'seller_type': {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3},
        'transmission': {'Manual': 1, 'Automatic': 2},
        'name': {
            'Maruti': 1, 'Skoda': 2, 'Honda': 3, 'Hyundai': 4, 'Toyota': 5, 'Ford': 6, 'Renault': 7,
            'Mahindra': 8, 'Tata': 9, 'Chevrolet': 10, 'Datsun': 11, 'Jeep': 12, 'Mercedes-Benz': 13,
            'Mitsubishi': 14, 'Audi': 15, 'Volkswagen': 16, 'BMW': 17, 'Nissan': 18, 'Lexus': 19,
            'Jaguar': 20, 'Land': 21, 'MG': 22, 'Volvo': 23, 'Daewoo': 24, 'Kia': 25, 'Fiat': 26,
            'Force': 27, 'Ambassador': 28, 'Ashok': 29, 'Isuzu': 30, 'Opel': 31
        }
    }
    
    for col, mapping in category_mappings.items():
        input_data_model[col].replace(mapping, inplace=True)

    # Make prediction
    car_price = model.predict(input_data_model.drop(['num_scratches', 'total_scratch_length'], axis=1))
    final_price = car_price[0] - total_scratch_cost

    # Display results
    st.markdown('Car Price is going to be '+ str(final_price))
    st.write(f'Number of scratches: {num_scratches}')
    if num_scratches > 0:
        st.write(f'Total length of scratches: {total_scratch_length} cm')
        st.write(f'Total cost of scratch repairs: {total_scratch_cost:.2f}')