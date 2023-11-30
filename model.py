import numpy as np
import pickle
import streamlit as st

with open('final.pkl', 'rb') as model_file:
    rf = pickle.load(model_file)



st.title("Predcting the Price of used Car")
st.subheader("Fiil the form for Prediction")

car_data={'Maruti': ['800', 'Wagon', 'Alto', 'Celerio', 'Ciaz', 'Vitara', 'Swift', 'Eeco', 'Omni', 'SX4', 'Ertiga', 'Zen', 'Baleno', 'Esteem', 'S-Cross', 'A-Star', 'Ritz', 'Estilo', 'Gypsy', 'Ignis', 'S-Presso', 'Grand'], 
 'Hyundai': ['Verna', 'Xcent', 'Creta', 'Venue', 'i10', 'Elantra', 'Santro', 'Grand', 'i20', 'EON', 'Getz', 'Elite', 'Sonata', 'Tucson', 'Accent', 'Santa'],
 'Datsun': ['RediGO', 'GO', 'redi-GO'], 
 'Honda': ['Amaze', 'City', 'Civic', 'Brio', 'Mobilio', 'Jazz', 'Accord', 'BR-V', 'WR-V', 'CR-V', 'BRV'],
 'Tata': ['Indigo', 'Tigor', 'Indica', 'Nano', 'Bolt', 'Nexon', 'Zest', 'Sumo', 'Tiago', 'Manza', 'Safari', 'New', 'Hexa', 'Venture', 'Xenon', 'Aria', 'Harrier', 'Altroz', 'Spacio', 'Winger'],
 'Chevrolet': ['Sail', 'Enjoy', 'Tavera', 'Beat', 'Cruze', 'Spark', 'Optra', 'Captiva', 'Aveo'], 
 'Toyota': ['Corolla', 'Innova', 'Etios', 'Fortuner', 'Camry', 'Yaris', 'Qualis'], 
 'Jaguar': ['XF', 'XJ'], 
 'Mercedes-Benz': ['New', 'E-Class', 'S-Class', 'GL-Class', 'C-Class', 'M-Class', 'B', 'GLS'], 
 'Audi': ['Q5', 'A6', 'Q7', 'A8', 'A4', 'Q3', 'A5', 'RS7'], 
 'Skoda': ['Superb', 'Rapid', 'Fabia', 'Yeti', 'Laura', 'Octavia'],
 'Jeep': ['Compass'], 'BMW': ['3', 'X1', '7', '5', 'X5'],
 'Mahindra': ['Scorpio', 'Jeep', 'XUV500', 'Bolero', 'Xylo', 'Quanto', 'XUV300', 'Renault', 'Marazzo', 'Supro', 'KUV', 'TUV', 'Verito', 'Thar', 'Alturas', 'NuvoSport', 'Ingenio'],
 'Ford': ['EcoSport', 'Figo', 'Fiesta', 'Aspire', 'Endeavour', 'Ecosport', 'Ikon', 'Freestyle', 'Fusion', 'Classic'],
 'Nissan': ['Terrano', 'Micra', 'Sunny', 'Evalia', 'Kicks', 'X-Trail'],
 'Renault': ['Duster', 'KWID', 'Scala', 'Pulse', 'Lodgy', 'Captur', 'Fluence', 'Koleos', 'Triber'],
 'Fiat': ['Avventura', 'Linea', 'Punto', 'Grande', 'Palio', '500'],
 'Volkswagen': ['Jetta', 'Vento', 'Ameo', 'Polo', 'Passat', 'CrossPolo'], 
 'Volvo': ['V40', 'XC60', 'XC'], 
 'Mitsubishi': ['Outlander', 'Pajero', 'Montero'],
 'Land': ['Rover'],
 'Daewoo': ['Matiz'],
 'MG': ['Hector'], 
 'Force': ['One'],
 'Isuzu': ['D-Max'],
 'OpelCorsa': ['1.6Gls', '1.4'], 
 'Ambassador': ['Classic', 'Grand', 'CLASSIC'], 
 'Kia': ['Seltos']}

model_enco = {'model_enc': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 
                            59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                            78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
                            97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
                            113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
                            128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
                            144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
                            160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
                            176, 177, 178, 179, 180, 181, 182, 183, 184], 
              'model': ['1.4', '1.6Gls', '3', '5', '500', '7', '800', 'A-Star', 'A4', 'A5', 'A6', 'A8',
                        'Accent', 'Accord', 'Alto', 'Altroz', 'Alturas', 'Amaze', 'Ameo', 'Aria', 'Aspire',
                        'Aveo', 'Avventura', 'B', 'BR-V', 'BRV', 'Baleno', 'Beat', 'Bolero', 'Bolt', 'Brio',
                        'C-Class', 'CLASSIC', 'CR-V', 'Camry', 'Captiva', 'Captur', 'Celerio', 'Ciaz', 'City',
                        'Civic', 'Classic', 'Compass', 'Corolla', 'Creta', 'CrossPolo', 'Cruze', 'D-Max', 
                        'Duster', 'E-Class', 'EON', 'EcoSport', 'Ecosport', 'Eeco', 'Elantra', 'Elite',
                        'Endeavour', 'Enjoy', 'Ertiga', 'Esteem', 'Estilo', 'Etios', 'Evalia', 'Fabia',
                        'Fiesta', 'Figo', 'Fluence', 'Fortuner', 'Freestyle', 'Fusion', 'GL-Class',
                        'GLS', 'GO', 'Getz', 'Grand', 'Grande', 'Gypsy', 'Harrier', 'Hector', 'Hexa',
                        'Ignis', 'Ikon', 'Indica', 'Indigo', 'Ingenio', 'Innova', 'Jazz', 'Jeep', 'Jetta',
                        'KUV', 'KWID', 'Kicks', 'Koleos', 'Laura', 'Linea', 'Lodgy', 'M-Class', 'Manza',
                        'Marazzo', 'Matiz', 'Micra', 'Mobilio', 'Montero', 'Nano', 'New', 'Nexon', 'NuvoSport', 
                        'Octavia', 'Omni', 'One', 'Optra', 'Outlander', 'Pajero', 'Palio', 'Passat', 'Polo', 'Pulse',
                        'Punto', 'Q3', 'Q5', 'Q7', 'Qualis', 'Quanto', 'RS7', 'Rapid', 'RediGO', 'Renault',
                        'Ritz', 'Rover', 'S-Class', 'S-Cross', 'S-Presso', 'SX4', 'Safari', 'Sail', 'Santa', 
                        'Santro', 'Scala', 'Scorpio', 'Seltos', 'Sonata', 'Spacio', 'Spark', 'Sumo', 'Sunny',
                        'Superb', 'Supro', 'Swift', 'TUV', 'Tavera', 'Terrano', 'Thar', 'Tiago', 'Tigor',
                        'Triber', 'Tucson', 'V40', 'Vento', 'Venture', 'Venue', 'Verito', 'Verna', 'Vitara',
                        'WR-V', 'Wagon', 'Winger', 'X-Trail', 'X1', 'X5', 'XC', 'XC60', 'XF', 'XJ', 'XUV300',
                        'XUV500', 'Xcent', 'Xenon', 'Xylo', 'Yaris', 'Yeti', 'Zen', 'Zest', 'i10', 'i20', 'redi-GO']}

brand_encoded = {
        'Ambassador': 0,
        'Mitsubishi': 0,
        'Kia' : 0,
        'Chevrolet' : 0,
        'Force' : 0,
        'Honda' : 0,
        'Ford': 0,
        'Isuzu':  0,
        'Jeep': 0,
        'Renault':  0,
        'Toyota' :0,
        'Hyundai' : 0,
        'Nissan' : 0,
         'OpelCorsa':  0,
         'Volkswagen':  0,
         'Daewoo':  0, 
         'Mercedes-Benz':  0,
         'MG': 0,
         'Datsun':  0,
         'Fiat':  0,
         'Audi': 0,
         'Volvo':  0,
         'Jaguar': 0,
         'Skoda': 0,
         'BMW':  0,
         'Maruti': 0,
        'Mahindra':  0,
        'Tata': 0,
        'Land': 0}

fuel_encoded = {
    'CNG': 0,
    'Diesel': 0,
    'Electric': 0,
    'LPG': 0,
    'Petrol': 0 }

seller_type_encoded = {
        'Dealer': 0,
        'Individual':  0,
        'Trustmark Dealer': 0
    }
    
    
transmission_encoded = {
        'Automatic': 0,
        'Manual': 0,
    }
    
owner_encoded = {
        'First Owner': 0,
        'Fourth & Above Owner': 0,
        'Second Owner':0,
        'Test Drive Car': 0,
        'Third Owner': 0}
        

# Input fields
year = st.number_input("Year", min_value=1992, max_value=2020)
km_driven = st.number_input("Kilometers Driven",min_value=1,max_value=None)



brand = st.selectbox("Select your brand",list(car_data.keys()))
st.write(f"You selected: {brand} as the brand.")
if brand:
    model = car_data.get(brand, [])
    if model:
        model = st.selectbox("Select a Model", model)
        st.write(f"You selected: {model} as the model.")
        selected_model_enc = model_enco['model_enc'][model_enco['model'].index(model)]
        
else:
    st.write("Select a brand first to see available models.")

fuel = st.selectbox("Fuel Type", ['CNG', 'Diesel', 'Electric', 'LPG', 'Petrol'])
seller_type = st.selectbox("Seller Type", ['Dealer', 'Individual', 'Trustmark Dealer'])
transmission=st.selectbox('Transmission',['Automatic', 'Manual'])
owner=st.selectbox("Owner",['First Owner','Fourth & Above Owner','Second Owner',
                        'Test Drive Car','Third Owner'])



if st.button("Predict Price"):
    # Encode the selected fuel type
    fuel_encoded = {
        'CNG': 1 if fuel == 'CNG' else 0,
        'Diesel': 1 if fuel == 'Diesel' else 0,
        'Electric': 1 if fuel == 'Electric' else 0,
        'LPG': 1 if fuel == 'LPG' else 0,
        'Petrol': 1 if fuel == 'Petrol' else 0
    }

    # Encode the selected seller type
    seller_type_encoded = {
        'Dealer': 1 if seller_type == 'Dealer' else 0,
        'Individual': 1 if seller_type == 'Individual' else 0,
        'Trustmark Dealer': 1 if seller_type == 'Trustmark Dealer' else 0
    }
    
    brand_encoded = {
        'Ambassador': 1 if brand == 'Ambassador' else 0,
        'Mitsubishi': 1 if brand == 'Mitsubishi' else 0,
        'Kia' : 1 if brand == 'Kia' else 0,
        'Chevrolet' : 1 if brand == 'Chevrolet' else 0,
        'Force' : 1 if brand == 'Force' else 0,
        'Honda' : 1 if brand == 'Honda' else 0,
        'Ford': 1 if brand == 'Ford' else 0,
        'Isuzu': 1 if brand == 'Isuzu' else 0,
        'Jeep': 1 if brand == 'Jeep' else 0,
        'Renault': 1 if brand == 'Renault' else 0,
        'Toyota' : 1 if brand == 'Toyota' else 0,
        'Hyundai' : 1 if brand == 'Hyundai' else 0,
        'Nissan' : 1 if brand == 'Nissan' else 0,
         'OpelCorsa': 1 if brand == 'OpelCorsa' else 0,
         'Volkswagen': 1 if brand == 'Volkswagen' else 0,
         'Daewoo': 1 if brand == 'Daewoo' else 0, 
         'Mercedes-Benz': 1 if brand == 'Mercedes-Benz' else 0,
         'MG': 1 if brand == 'MG' else 0,
         'Datsun': 1 if brand == 'Datsun' else 0,
         'Fiat': 1 if brand == 'Fiat' else 0,
         'Audi': 1 if brand == 'Audi' else 0,
         'Volvo': 1 if brand == 'Volvo' else 0,
         'Jaguar': 1 if brand == 'Jaguar' else 0,
         'Skoda': 1 if brand == 'Skoda' else 0,
         'BMW': 1 if brand == 'BMW' else 0,
         'Maruti': 1 if brand == 'Maruti' else 0,
        'Mahindra': 1 if brand == 'Mahindra' else 0,
        'Tata': 1 if brand == 'Tata' else 0,
        'Land': 1 if brand == 'Land' else 0
    }
    
    transmission_encoded = {
        'Automatic': 1 if transmission == 'Automatic' else 0,
        'Manual': 1 if transmission == 'Manual' else 0,
    }
    
    owner_encoded = {
        'First Owner': 1 if owner == 'First Owner' else 0,
        'Fourth & Above Owner': 1 if owner == 'Fourth & Above Owner' else 0,
        'Second Owner': 1 if owner == 'Second Owner' else 0,
        'Test Drive Car': 1 if owner == 'Test Drive Car' else 0,
        'Third Owner': 1 if owner == 'Third a' else 0,
        
    }
    
    test = np.array([year, km_driven,selected_model_enc]+ list(fuel_encoded.values()) + list(seller_type_encoded.values()) + list(transmission_encoded.values()) + list(owner_encoded.values()) + list(brand_encoded.values()) )
    test = test.reshape(1, -1)

    st.success(rf.predict(test)[0])


