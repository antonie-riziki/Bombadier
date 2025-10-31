import pandas as pd 
import cv2
import streamlit as st 
import matplotlib.pyplot as plt 
import seaborn as sb 
import numpy as np  
import csv
import os
import sys
import tempfile
import warnings
import joblib
import random
import africastalking
import calendar

from datetime import datetime, timedelta


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, r2_score, mean_absolute_error, mean_squared_error

from PIL import Image


sys.path.insert(1, './pages')
print(sys.path.insert(1, '../pages/'))

# Get the absolute path of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the pages directory to sys.path
sys.path.insert(0, os.path.join(current_dir, '../pages'))

from ct_model import  prepare_train_test_for_soil_quality, prepare_train_test_for_soil_ph, prepare_train_test_for_soil_type, prepare_train_test_for_crop_type, get_recommended_crop, get_ai_content, the_explainer, get_crop_summary

from dotenv import load_dotenv

load_dotenv()

africastalking.initialize(
    username='EMID',
    api_key = os.getenv("AT_API_KEY")
)

sms = africastalking.SMS
airtime = africastalking.Airtime


pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 1000)

sb.set_style("darkgrid")

warnings.filterwarnings("ignore")

header = st.container()
home_sidebar = st.container()
main_body = st.container()


with header:
	st.title("BOMBADIER: PROOF OF CONCEPT")

	st.markdown('Discover the Secrets Beneath Your Feet')

	st.write('''
		Bombadier is your intelligent companion for understanding and optimizing soil health. Powered by advanced IoT sensors and 
		Machine Learning algorithms, Bombadier analyzes your soil's type, moisture levels, pH, and nutrient content. It then provides 
		tailored recommendations for the best crops to plant, ensuring maximum yield and sustainability. Whether you're a seasoned farmer 
		or a first-time grower, Bombadier takes the guesswork out of farming and puts science in your hands.

		''')
	
	# tab1,tab2 = st.tabs(['Capture Image', 'Take a Video'])

with main_body:
	
	current_dir = os.path.dirname(os.path.abspath(__file__))

	# Construct the absolute path to the CSV file
	csv_path = os.path.join(current_dir, '../src/crop_yield_dataset.csv')
	
	df = pd.read_csv(csv_path)

	le = LabelEncoder()
	def get_categorical_columns(df):
	    category = []
	    for i in df.select_dtypes(include=["object"]):
	        df[i] = le.fit_transform(df[i])
	    plt.figure(figsize=(12, 8))
	    return sb.heatmap(df.corr(), annot=True, linewidths=0.5, cmap="viridis")

	get_categorical_columns(df)


	##############################################################################################################################


	# SOIL QUALITY MODEL


	###############################################################################################################################

	# x_train, x_test, y_train, y_test = prepare_train_test_for_soil_quality(df)

	def get_soil_quality(data):

	    input_data_array = np.asarray(data)
	    input_data_array_reshaped = input_data_array.reshape(1, -1)
		
	    current_dir = os.path.dirname(os.path.abspath(__file__))

	    # Construct the absolute path to the CSV file
	    model_path = os.path.join(current_dir, '../model/sqm.pkl')

	    load_sqm = joblib.load(model_path)

	    pred = load_sqm.predict(input_data_array_reshaped)
	    
	    data = input_data_array_reshaped.flatten().tolist()  # Convert the reshaped array to a flat list
	    pred.flatten().tolist()
	    data.append(float(pred[0]))  # Append the prediction (assuming `pred` is an array or list with one element)
	    data = [float(value) for value in data]

	    st.write("**Soil Quality**", unsafe_allow_html=True)

	    if pred >= 70:
	        st.success(f"✅ Soil Quality Score: {round(pred[0])}, which indicates excellent fertility")
	    elif 50 <= pred < 70:
	        st.success(f"✅ Soil Quality Score: {round(pred[0])}, which reflects good fertility.")
	    elif 30 <= pred < 50:
	        st.info(f"ℹ️ Soil Quality Score: {round(pred[0])}, indicating moderate fertility.")
	    elif 15 <= pred < 30:
	        st.warning(f"⚠️ Soil Quality Score: {round(pred[0])}, suggesting low fertility.")
	    else:
	        st.error(f"❌ The soil quality score is {round(pred[0])}, which indicates very poor fertility.")
	    st.write("")

	    return pred, data


	##############################################################################################################################


	# SOIL PH MODEL


	############################################################################################################################### 

	# x_train_2, x_test_2, y_train_2, y_test_2 = prepare_train_test_for_soil_ph(df)

	def get_soil_ph(data):
	    input_data_array = np.asarray(data)
	    input_data_array_reshaped = input_data_array.reshape(1, -1)

	    current_dir = os.path.dirname(os.path.abspath(__file__))

	    # Construct the absolute path to the CSV file
	    model_path = os.path.join(current_dir, '../model/sph.pkl')	

	    load_sph = joblib.load(model_path)

	    pred = load_sph.predict(input_data_array_reshaped)

	    data = input_data_array_reshaped.flatten().tolist()  # Convert the reshaped array to a flat list
	    pred.flatten().tolist()
	    data.append(float(pred[0]))  # Append the prediction (assuming `pred` is an array or list with one element)
	    data = [float(value) for value in data]
	    
	    st.markdown("**Soil PH**", unsafe_allow_html=True)
	    if 6.2 <= pred <= 6.8:
	        st.write(f"The predicted soil pH is {round(pred[0])}, which is within the optimal range for most crops (6.2 - 6.8). "
	                "This means your soil is slightly acidic, making it ideal for nutrient absorption. "
	                "To maintain or improve this balance, consider regular pH testing and avoid overusing acidic or alkaline fertilizers.")
	    elif pred < 6.2:
	        st.write(f"The predicted soil pH is {round(pred[0])}, which is lower than the optimal range (6.2 - 6.8). "
	                "This indicates your soil is more acidic, which can limit nutrient availability. "
	                "Consider applying lime to raise the pH and improve soil conditions.")
	    else:
	        st.write(f"The predicted soil pH is {round(pred[0])}, which is higher than the optimal range (6.2 - 6.8). "
	                "This indicates alkaline soil, which might reduce nutrient uptake. "
	                "Adding sulfur or organic matter can help lower the pH.")
	    st.write("")
	    return pred, data


	##############################################################################################################################

	
	# SOIL TYPE MODEL
	

	############################################################################################################################### 

	# x_train_3, x_test_3, y_train_3, y_test_3 = prepare_train_test_for_soil_type(df)

	def get_soil_type(data):
	    input_data_array = np.asarray(data)
	    input_data_array_reshaped = input_data_array.reshape(1, -1)

	    current_dir = os.path.dirname(os.path.abspath(__file__))

	    # Construct the absolute path to the CSV file
	    model_path = os.path.join(current_dir, '../model/styp.pkl')

	    load_styp = joblib.load(model_path)

	    pred = load_styp.predict(input_data_array_reshaped)

	    new_pred_val = pred[0]

	    st.write("**Soil Type**", unsafe_allow_html=True)

	    try:
	        if new_pred_val == 0:
	  
	            st.write(f'''The soil type is identified as Clay Soil. Clay soils are nutrient-rich and retain water well, 
				            but they can become compacted and drain poorly if not managed properly. To improve conditions, 
				            consider adding organic matter like compost or aged manure to enhance soil structure and aeration. 
				            Growing crops like rice, legumes, and sunflowers can be beneficial.''')

	        elif new_pred_val == 1:
	            st.write(f''''The soil type is identified as Loam Soil. Loam is the ideal soil type for most crops because it provides 
				            excellent drainage while retaining adequate moisture and nutrients. To maintain this balance, 
				            regularly add organic matter and avoid over-tilling. Crops such as vegetables, fruits, and grains thrive in loam soil.''')
	        
	        elif new_pred_val == 2:
	            st.write(f'''The soil type is identified as Peaty Soil. Peaty soils are high in organic matter and retain moisture well, 
				            but they can be acidic, which may limit nutrient availability. To optimize crop growth, consider adding lime to 
				            reduce acidity and improve pH balance. Suitable crops include root vegetables, berries, and shrubs. ''')

	        elif new_pred_val == 3:
	            st.write(f''' The soil type is identified as Saline Soil. Saline soils have high salt content, which can hinder plant growth. 
				            To improve conditions, focus on leaching salts by providing adequate drainage and using freshwater irrigation. 
				            Salt-tolerant crops such as barley, sugar beet, and certain grasses are recommended for this soil type. ''')
	        
	        elif new_pred_val == 4:
	            st.write(f''' The soil type is identified as Sand Soil. Sandy soils drain quickly and are easy to work with, but they retain 
				            fewer nutrients and moisture, making them less fertile. To enhance fertility, add organic matter like compost or 
				            mulch. Crops such as carrots, potatoes, and watermelon grow well in sandy soils. ''')
	        else:
	            st.write(f"Value out of range {pred}")
	    except ValueError as e:
	        st.write(f"Unexpected Error Occurred: {e}")

	    data = input_data_array_reshaped.flatten().tolist()  # Convert the reshaped array to a flat list
	    # pred.flatten().tolist()
	    data.append(new_pred_val)  # Append the prediction (assuming `pred` is an array or list with one element)
	    data = [float(value) for value in data]
	    st.write("")
	    return pred, data


	##############################################################################################################################

	
	# CROP TYPE MODEL
	

	############################################################################################################################### 

	# x_train_4, x_test_4, y_train_4, y_test_4 = prepare_train_test_for_crop_type(df)

	def get_crop_type(data):
	    input_data_array = np.asarray(data)
	    input_data_array_reshaped = input_data_array.reshape(1, -1)

	    current_dir = os.path.dirname(os.path.abspath(__file__))

	    # Construct the absolute path to the CSV file
	    model_path = os.path.join(current_dir, '../model/ctyp.pkl')

	    load_ctyp = joblib.load(model_path)

	    pred = load_ctyp.predict(input_data_array_reshaped)

	    new_pred_val = pred[0]

	    st.write("**Crop Type Prediction**", unsafe_allow_html=True)
	    
	    try:
	        if new_pred_val == 0:
	            st.write(f''' The predicted crop type is Barley. Barley grows best in well-drained, loam or sandy-loam soils with a pH range of 6.0 to 7.5.
                   			It thrives in moderately fertile soils with good moisture retention. To ensure optimal growth, avoid waterlogging, and apply nitrogen fertilizers during early growth stages. ''')

	        elif new_pred_val == 1:
	            st.write(f'''The predicted crop type is Maize. Maize grows well in loam or sandy-loam soils with good drainage and a pH range of 5.5 to 7.0. 
                  			This crop needs adequate nitrogen and potassium for optimal growth. Regular irrigation and weed control are crucial to ensure a healthy yield. ''')
	        
	        elif new_pred_val == 2:
	            st.write(f''' The predicted crop type is Cotton. Cotton prefers well-drained, loam or sandy-loam soils with a pH range of 5.8 to 7.0.
                   			This crop requires good fertility and warm temperatures for growth. Regular irrigation and the application of potassium-rich fertilizers can improve fiber quality. ''')

	        elif new_pred_val == 3:
	            st.write(f''' The predicted crop type is Potato. Potatoes prefer well-drained, sandy-loam soils with a pH range of 5.0 to 6.5.
                   			This crop requires soils rich in organic matter and potassium. Avoid waterlogging by improving soil drainage, 
                   			and consider mulching to retain soil moisture.''')

	        elif new_pred_val == 4:
	            st.write(f''' The predicted crop type is Rice. Rice thrives in clay or silty soils that can hold water for extended periods. 
		                 It requires soils with a pH range of 5.5 to 6.5 and high organic matter. Ensure proper water management and nutrient application, 
		                 including nitrogen, phosphorus, and potassium, for a high yield. ''')

	        elif new_pred_val == 5:
	            st.write(f''' The predicted crop type is Soybean. Soybeans grow well in loam or clay-loam soils with a pH range of 6.0 to 7.0. 
                    		These crops require moderate fertility and good drainage. Incorporating rhizobium inoculants can enhance nitrogen fixation and improve yield. ''')
	        
	        elif new_pred_val == 6:
	            st.write(f'''The predicted crop type is Sugarcane. Sugarcane thrives in deep, well-drained, loam or clay-loam soils with a pH range of 6.0 to 7.5. 
                      		It requires high fertility and abundant water for optimal growth. Regular irrigation, proper weed control, and the application of nitrogen and phosphorus fertilizers 
                      		are crucial for high yields. ''')

	        elif new_pred_val == 7:
	            st.write(f'''The predicted crop type is Sunflower. Sunflowers are well-suited for a wide range of soils, 
                      including loam and clay soils, as long as they are well-drained. Sunflowers thrive in soils with 
                      a pH range of 6.0 to 7.5 and moderate to high fertility. To maximize yield, ensure adequate sunlight 
                      and apply nitrogen-based fertilizers during the early growth stages. ''')

	        elif new_pred_val == 8:
	            st.write(f''' The predicted crop type is Tomato. Tomatoes thrive in well-drained, fertile, sandy-loam soils with a pH range of 6.0 to 6.8. 
                   			They require sufficient sunlight, regular watering, and balanced fertilizers to support their growth. Adding organic mulch can help retain moisture. ''')

	        elif new_pred_val == 9:
	            st.write(f'''The predicted crop type is Wheat. Wheat requires well-drained, loam or clay-loam soils with a pH between 6.0 and 7.0. 
		                  It grows best in moderately fertile soils with good moisture retention. Ensure proper irrigation during critical growth stages 
		                  and apply phosphorus fertilizers to enhance grain yield. ''')

	        else:
	            st.write(f"Value out of range {pred}")
	    except ValueError as e:
	        st.write(f"Unexpected Error Occurred: {e}")

	    data = input_data_array_reshaped.flatten().tolist()  # Convert the reshaped array to a flat list
	    # pred.flatten().tolist()
	    data.append(new_pred_val)  # Append the prediction (assuming `pred` is an array or list with one element)
	    data = [float(value) for value in data]
	    
	    st.write("")
	    # st.write(data)
	    st.dataframe(pd.DataFrame([data], columns=['N', 'P', 'K', 'Crop_Yield', 'Soil_Quality', 'Soil_pH', 'Soil_Type', "Crop_Type"]))
	    return pred, data



	with st.form("Self Service Soil Testing Form"):
		col1, col2 = st.columns(2)

		with col1:
			nitrogen_values = st.number_input("Enter nitrogen level", df['N'].min(), df['N'].max(), (df['N'].mean()))
			phosphorus_values = st.number_input("Enter phosphorus level", df['P'].min(), df['P'].max(), (df['P'].mean()))

		with col2:
			potassium_values = st.number_input("Enter potassium level", df['K'].min(), df['K'].max(), (df['K'].mean()))
			crop_yield_values = st.number_input("Enter crop yield level", df['Crop_Yield'].min(), df['Crop_Yield'].max(), (df['Crop_Yield'].mean()))
		
		
		def get_crop_images(subpath):
			current_dir = os.path.dirname(os.path.abspath(__file__))

			# Construct the absolute path to the CSV file
			img_path = os.path.join(current_dir, "../assets/img/crop_type")
		  
			corn_folder = os.path.join(img_path, subpath)
			return(corn_folder)

		def get_soil_images(subpath):
			current_dir = os.path.dirname(os.path.abspath(__file__))

			# Construct the absolute path to the CSV file
			img_path = os.path.join(current_dir, "../assets/img/soil_type")
			
			soil_folder = os.path.join(img_path, subpath)
			return(soil_folder)


		if st.form_submit_button("Cultivate Wisdom", use_container_width=True, type="primary"):
			col1, col2 = st.columns(2)

			with st.expander('Full Report', expanded=True):

				with col1:
					data = (nitrogen_values, phosphorus_values, potassium_values, crop_yield_values)
				
					soil_quality_prediction, soil_quality_data = get_soil_quality(data)

					soil_ph_prediction, soil_ph_data = get_soil_ph(soil_quality_data)

					soil_type_prediction, soil_type_data = get_soil_type(soil_ph_data)

					crop_type_prediction, crop_type_data = get_crop_type(soil_type_data)

					


				with col2:
					st.write("Visual Insights: Soil & Crop")
					# crop_type_prediction, crop_type_data = get_crop_type(soil_type_data)

					crop_dict = {
						'barley': 0,
						'corn': 1,
						'cotton': 2,
						'potato': 3,
						'rice':4,
						'soyabeans': 5,
						'sugarcane': 6,
						'sunflower': 7,
						'tomato': 8,
						'wheat': 9,
					}

					soil_dict = {
						'clay': 0,
						'loam': 1,
						'peaty': 2,
						'saline': 3,
						'sand': 4,
					}

					for val, key in enumerate(soil_dict):
						if soil_type_prediction == val:  # Check the prediction condition

							path = get_soil_images(key)
							
							if os.path.exists(path) and os.path.isdir(path):
								
								images = [file for file in os.listdir(path) if file.endswith(('.png', '.jpg', '.jpeg'))]
								
								if images:
									random_image = random.choice(images)
									image_path = os.path.join(path, random_image)

									img = Image.open(image_path)
									st.image(img, caption=f"{key} soil", use_container_width=True)
								else:
									st.warning("No images found in the folder.")
							else:
								st.error(f"The folder does not exist")
								break
						else:
							pass

					for val, key in enumerate(crop_dict):
						if crop_type_prediction == val:  # Check the prediction condition

							path = get_crop_images(key)
							
							if os.path.exists(path) and os.path.isdir(path):
								
								images = [file for file in os.listdir(path) if file.endswith(('.png', '.jpg', '.jpeg'))]
								
								if images:
									random_image = random.choice(images)
									image_path = os.path.join(path, random_image)

									img = Image.open(image_path)
									st.image(img, caption=f"{key}", use_container_width=True)
								else:
									st.warning("No images found in the folder.")
							else:
								st.error("The folder does not exist.")
						else:
							pass

			recommended_crop = get_recommended_crop(soil_type_data)


			for val, key in enumerate(crop_dict):
				if crop_type_prediction == val:
					tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Ploughing", "Sowing", "Adding Nutrients", "Irrigation", "Pests & Disease", "Harvest", "Storage", "Summary"])


					with tab1:
						st.title("Land Preparation")
						
						prompt = f''' What are the best practices for preparing the soil before planting {key}, 
						including ploughing techniques, depth, and timing? '''
						get_ai_content(prompt)

					with tab2:
						st.title("Seed Planting")
						
						prompt = f''' What is the recommended spacing, planting depth, and seed variety selection for 
						{key} sowing to ensure optimal growth? '''
						get_ai_content(prompt)
						

					with tab3:
						st.title("Nutrition Boosting")
						
						prompt = f''' What types of fertilizers or organic nutrients are recommended for {key} at different 
						growth stages, and how should they be applied? '''
						get_ai_content(prompt)

					with tab4:
						st.title("Land Watering")
						
						prompt = f''' What is the ideal irrigation schedule for {key} to ensure healthy growth without 
						overwatering or underwatering? '''
						get_ai_content(prompt)

					with tab5:
						st.title("Integrated Pest Management (IPM)")
						
						prompt = f''' What are the common pests and diseases that affect {key}, and how can they be effectively 
						controlled using sustainable methods? '''
						get_ai_content(prompt)

					with tab6:
						st.title("Abadunt Reaping")
						
						prompt = f''' When is the best time to harvest {key}, and what are the signs that indicate readiness for harvesting? '''
						get_ai_content(prompt)

					with tab7:
						st.title("Repository")
						
						prompt = f''' What are the best storage practices for {key} to maintain quality and prevent spoilage or pest infestation? '''
						get_ai_content(prompt)


					with tab8:
						st.title("Summary Report Details")
						
						prompt = f''' Generate a short planting report for {key} based on Kenyan ecosystem. Simplify the details into one line per point, 
								using quantified metrics where applicable. Ensure the report is concise, easy to understand, and suitable for sending via SMS. Example format:

									1. Crop: [Crop Type]
									2. Optimal Planting Time: [Time/Month related to Kenyan climatic conditions during each season]
									3. Required Fertilizer: [Amount and Type, also recommend best agro-input company in Kenya for fertilizer]
									4. Expected Yield: [Quantity in kg/acre for smallscale farmers in Kenya]
									5. Water Requirement: [Liters per acre/week or other relevant metrics]
									Provide additional points if relevant, while keeping each line brief and clear '''
						
						get_crop_summary(prompt)

	col1, col2, col3 = st.columns(3, gap="large", vertical_alignment="bottom")

	with col1:
		# get_sms =  st.button("Send to SMS", icon=":material/sms:")

		# if get_sms:

		with st.popover("Send to SMS", icon=":material/sms:"):

			with st.form(key="report"):
				phone_number = st.number_input('Phone Number', value=0, min_value=0, max_value=int(10e10))

				submit_report = st.form_submit_button("Send")

				def send_report():
					amount = "10"
					currency_code = "KES"


					recipients = [f"+254{str(phone_number)}"]
					# airtime_rec = "+254" + str(phone_number)
					print(recipients)
					print(phone_number)

					# Set your message
					message = f"Your report has been created and a link will be sent shortly. Meanwhile visit www.echominds.africa for more information";
					# Set your shortCode or senderId
					sender = 20880
					try:
						# responses = airtime.send(phone_number=airtime_rec, amount=amount, currency_code=currency_code)
						response = sms.send(message, recipients, sender)
						
						print(response)
						# print(responses)
					except Exception as e:
						print(f'Houston, we have a problem: {e}')

			if submit_report not in st.session_state:
				send_report()
	
	with col2:
		with st.popover("Chat on Whatsapp", icon=":material/chat:"):

			st.info("Coming soon")

				

	with col3:
		voice_command = st.button("Sauti ya Shamba", icon=":material/mic:")

				


	# gemini_chatbot()



############################################################################

# Loan functions 

############################################################################


def generate_synthetic_data(n=3000, random_state=42):
    rng = np.random.RandomState(random_state)
    # Features:
    # farm_size (acres), avg_monthly_yield (kg), repayment_history_pct (0-100),
    # group_member (0/1), mobile_money_usage_pct (0-100), assets_value_ksh,
    # existing_loans_ksh, soil_quality_score (0-100), age, crop_type_encoded (0-4)
    farm_size = rng.gamma(2.0, 1.5, size=n)  # skewed smallholder sizes
    avg_yield = np.clip(rng.normal(500, 300, size=n), 50, None)
    repayment_hist = np.clip(rng.normal(85, 20, size=n), 0, 100)
    group_member = rng.binomial(1, 0.6, size=n)
    mm_usage = np.clip(rng.normal(70, 30, size=n), 0, 100)
    assets = np.clip(rng.normal(100000, 80000, size=n), 0, None)
    existing_loans = np.clip(rng.normal(20000, 30000, size=n), 0, None)
    soil_score = np.clip(rng.normal(60, 20, size=n), 0, 100)
    age = np.clip(rng.normal(40, 12, size=n), 18, 80)
    crop_type = rng.randint(0, 5, size=n)

    # Outcome: default (1) or not (0). Construct using intuitive rules + noise.
    # Higher yield, better repayment_hist, group_member, mm_usage reduce default.
    risk_score = (
        0.35 * (1 / (1 + farm_size)) +
        0.30 * (1 - (repayment_hist / 100)) +
        0.15 * (existing_loans / (existing_loans + assets + 1e-6)) +
        0.10 * (1 - (soil_score / 100)) +
        0.05 * (1 - (mm_usage / 100)) +
        0.05 * (age < 25)  # younger farmers slightly more risky
    )
    # noise and clip
    risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())
    # probability of default
    p_default = 0.1 + 0.7 * risk_score  # baseline 10% up to ~80%
    y = rng.binomial(1, p_default)
    df = pd.DataFrame({
        "farm_size": farm_size,
        "avg_yield": avg_yield,
        "repayment_hist": repayment_hist,
        "group_member": group_member,
        "mm_usage": mm_usage,
        "assets": assets,
        "existing_loans": existing_loans,
        "soil_score": soil_score,
        "age": age,
        "crop_type": crop_type,
        "default": y
    })
    return df

# ---------------------------
# Helper: train model
# ---------------------------
@st.cache_data(show_spinner=False)
def train_model(df):
    X = df.drop(columns=["default"])
    y = df["default"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=150, random_state=7)
    model.fit(X_train_scaled, y_train)

    return model, scaler, X_test, y_test

# ---------------------------
# Scoring & interpretation
# ---------------------------
def compute_credit_score(model, scaler, features_df):
    X_scaled = scaler.transform(features_df)
    # We predict probability of default; we want score where higher is better credit
    p_default = model.predict_proba(X_scaled)[:, 1]
    credit_score = np.clip((1 - p_default) * 100, 0, 100)  # 0..100 (higher better)
    return credit_score, p_default

def risk_band_from_score(score):
    if score >= 80:
        return "Low risk"
    if score >= 60:
        return "Moderate risk"
    if score >= 40:
        return "High risk"
    return "Very high risk"

def generate_recommendation(score, p_default, row):
    recs = []
    if score >= 80:
        recs.append("Eligible for standard loan products with normal rates.")
    elif score >= 60:
        recs.append("Eligible for small loan; consider co-borrower or cooperative guarantee.")
    else:
        recs.append("High risk — recommend small pilot loan, require group guarantee or collateral.")

    # actionable items
    if row["repayment_hist"] < 70:
        recs.append("Improve repayment record by clearing past loans and paying on time.")
    if row["soil_score"] < 50:
        recs.append("Improve soil through composting and soil testing to increase productivity.")
    if row["mm_usage"] < 50:
        recs.append("Increase mobile money usage for better digital transaction history.")

    # brief AI style sentence
    ai_sentence = f"Estimated default probability: {p_default:.1%}. " + " ".join(recs[:2])
    return ai_sentence

# ---------------------------
# Streamlit UI
# ---------------------------
# st.set_page_config(page_title="AgriLoan Analytics", layout="wide")
# st.title("AgriLoan Analytics — Credit Scoring for Smallholder Farmers")
# st.write("Enter farmer loan parameters to compute analytics, credit score, and recommendations.")

# Left: input form
with st.form("loan_form"):
    st.header("Farmer Loan Inputs")
    col1, col2, col3 = st.columns(3)
    with col1:
        farmer_name = st.text_input("Farmer name", value="John Doe")
        phone = st.text_input("Phone (2547...)", value="254700000000")
        age = st.number_input("Age", min_value=18, max_value=100, value=40)
        farm_size = st.number_input("Farm size (acres)", min_value=0.01, max_value=100.0, value=1.0, step=0.1)
        crop_type = st.selectbox("Crop type", ["Maize","Beans","Kale","Tomato","Sorghum"])
    with col2:
        avg_yield = st.number_input("Avg monthly yield (kg)", min_value=1.0, max_value=10000.0, value=500.0)
        repayment_hist = st.slider("Repayment history (%)", min_value=0, max_value=100, value=85)
        group_member = st.radio("Member of cooperative / group?", ("Yes","No"))
        mm_usage = st.slider("Mobile money usage (%)", 0, 100, 70)
    with col3:
        assets = st.number_input("Assets value (KSH)", min_value=0.0, value=100000.0, step=1000.0)
        existing_loans = st.number_input("Existing loans (KSH)", min_value=0.0, value=20000.0, step=1000.0)
        soil_score = st.slider("Soil quality score (0-100)", 0, 100, 60)
        requested_amount = st.number_input("Requested loan amount (KSH)", min_value=100.0, value=10000.0, step=100.0)

    submitted = st.form_submit_button("Analyze & Score", use_container_width=True, type="primary")

# Train model once
df = generate_synthetic_data()
model, scaler, X_test, y_test = train_model(df)

# On submit: compute score and show analytics
if submitted:
    # prepare single-row dataframe with same features as training
    crop_map = {"Maize":0,"Beans":1,"Kale":2,"Tomato":3,"Sorghum":4}
    features = pd.DataFrame([{
        "farm_size": farm_size,
        "avg_yield": avg_yield,
        "repayment_hist": repayment_hist,
        "group_member": 1 if group_member == "Yes" else 0,
        "mm_usage": mm_usage,
        "assets": assets,
        "existing_loans": existing_loans,
        "soil_score": soil_score,
        "age": age,
        "crop_type": crop_map[crop_type]
    }])

    score_arr, p_default_arr = compute_credit_score(model, scaler, features)
    score = float(score_arr[0])
    p_default = float(p_default_arr[0])
    band = risk_band_from_score(score)
    recommendation = generate_recommendation(score, p_default, features.iloc[0])

    # Layout results
    st.subheader("Credit Score & Risk")
    colA, colB = st.columns([1,2])
    with colA:
        st.metric("Credit Score (0-100)", f"{score:.1f}")
        st.write("Risk band:", band)
        st.write(f"Estimated default probability: {p_default:.1%}")
        st.write("")
        st.write("Quick recommendation:")
        st.write(recommendation)

    # Feature importance
    feat_names = features.columns.tolist()
    importances = model.feature_importances_
    fi_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)

    with colB:
        st.write("Feature importances (model)")
        fig, ax = plt.subplots(figsize=(6,3))
        ax.barh(fi_df["feature"], fi_df["importance"])
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        st.pyplot(fig)

    # Show sample analytics: where this farmer sits in score distribution
    # compute scores for test set quickly
    X_test_scaled = scaler.transform(X_test)
    p_default_test = model.predict_proba(X_test_scaled)[:, 1]
    scores_test = (1 - p_default_test) * 100

    st.write("Score distribution vs this farmer")
    fig2, ax2 = plt.subplots(figsize=(8,2.5))
    ax2.hist(scores_test, bins=30)
    ax2.axvline(score, color="k", linewidth=2, linestyle="--")
    ax2.set_xlabel("Credit Score")
    st.pyplot(fig2)

    # Decision guidance (loan sizing suggestion)
    # Simple rule: max exposure = assets * 0.5 * (score/100)
    suggested_max = features["assets"].iloc[0] * 0.5 * (score/100)
    suggested_amount = min(requested_amount, suggested_max)
    st.write(f"Suggested maximum loan based on simple exposure rule: KSH {suggested_max:,.0f}")
    st.write(f"Suggested approved amount for this request: KSH {suggested_amount:,.0f}")

    # Offer to export report
    report = {
        "farmer_name": farmer_name,
        "phone": phone,
        "score": score,
        "default_prob": p_default,
        "risk_band": band,
        "suggested_amount": suggested_amount,
        "requested_amount": requested_amount,
        "recommendation": recommendation
    }
    report_df = pd.DataFrame([report])

    csv = report_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download report (CSV)", data=csv, file_name=f"{farmer_name}_loan_report.csv", mime="text/csv")

    # Short plain-language message suitable for SMS/USSD
    sms_msg = (f"{farmer_name}: Credit score {score:.0f}/100 ({band}). "
               f"Default risk {p_default:.0%}. Recommended loan KSH {suggested_amount:,.0f}.")
    st.write("Short message for farmer (SMS/USSD):")
    st.code(sms_msg)

# Sidebar: quick info and model performance
with st.sidebar:
    st.header("Model / App Info")
    st.write("Prototype model trained on synthetic data.")
    st.write("Use real historical data for production.")
    if st.button("Show sample data (first 5 rows)"):
        st.dataframe(df.head())

    # display a tiny model performance summary
    try:
        X = df.drop(columns=["default"])
        y = df["default"]
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        acc = (preds == y).mean()
        st.write(f"Model approximate accuracy (on synthetic data): {acc:.2%}")
    except Exception:
        st.write("Model not ready.")


##############################################################################################################################

# LOAN CALCULATING LOGIC

###############################################################################################################################

with st.form(key='form1'):
	col1, col2, col3 = st.columns(3)
	with col1:
		loan_amount = st.number_input('amount borrowed', value=0, min_value=0, max_value=int(10e10))

	with col2:
		payment_rate = st.slider('Interest rate', 0.0, 10.0)/100.0

	with col3:
		monthly_amount = st.number_input('Monthly re-payment', min_value=0, max_value=int(10e10))

	# submit_label = f'<i class="fas fa-calculator">Calculate</i> '
	# submit = st.form_submit_button(submit_label)

	submit = st.form_submit_button(label='Loan Metrics', use_container_width=True, type="primary")

	#Determine the total period it takes to repay off a loan
	# bal = 5000
	# interestRate = 0.13
	# monthlyPayment = 500

if submit not in st.session_state:
	df = pd.DataFrame(columns=['End Month', 'Loan Amount', 'Interest Charge'])
	
	current_date = datetime.today()
	# print(current_date)

	end_month_day = calendar.monthrange(current_date.year, current_date.month)[1]
	days_left = end_month_day - current_date.day

	next_month_start_date = current_date + timedelta(days=days_left + 1)
	end_month = next_month_start_date

	period_count = 0
	total_int = 0
	data = []

	while loan_amount > 0:
		int_charge = (payment_rate / 12) * loan_amount
		loan_amount += int_charge
		loan_amount -= monthly_amount

		if loan_amount <= 0:
			loan_amount = 0
		total_int += int_charge
		print(end_month, round(loan_amount, 2), round(int_charge, 2))

		period_count += 1
		new_date = calendar.monthrange(end_month.year, end_month.month)[1]
		end_month += timedelta(days=new_date)

		# df = df.append({'End Month': end_month, 'Loan Amount': round(loan_amount, 2), 'Interest Charge': round(int_charge, 2)}, ignore_index=True)

		data.append([end_month.date(), round(loan_amount, 2), round(int_charge, 2)])

		if loan_amount == 0:
			break

	print('Total Interest Rate paid: ', total_int)
	df = pd.DataFrame(data, columns=['next_pay_date', 'amount_remaining', 'interest_amount'])

	
	years = int(period_count // 12)
	months_remaining = round(period_count % 12)
	print(f"{years} years and {months_remaining} months")

	col1, col2 = st.columns(2)
	with col1:
		st.dataframe(df, use_container_width=True)

	with col2:
		st.write('Loan payment due')
		col1, col2, col3 = st.columns(3)
		col1.metric("", str(years), " yrs")
		col2.metric("", str(months_remaining), " months")
		st.metric("Total Interest Paid", "sh. " + str(round(total_int)), "")
		# col2.metric("Wind", "9 mph", "-8%")
		# col3.metric("Humidity", "86%", "4%")
