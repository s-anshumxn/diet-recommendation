import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

data = pd.read_csv("wweia_data.csv")
data_category = pd.read_csv("wweia_food_categories_addtl.csv")
updated = data.iloc[:,3:]
updt = data.iloc[:,:3]
df=data
# 1st method makes features between -1 to 1
data_norm_changed = updated.apply(lambda iterator: ((iterator - iterator.mean())/iterator.std()).round(2))
data_norm = pd.concat([updt, data_norm_changed], axis ="columns")


synonims = {1002: {1002, 1004, 1006, 1008, 1202, 1204, 1206, 1208, 1402, 1404},
 1004: {1002, 1004, 1006, 1008, 1202, 1204, 1206, 1208, 1402, 1404},
 1006: {1002, 1004, 1006, 1008, 1202, 1204, 1206, 1208, 1402, 1404},
 1008: {1002, 1004, 1006, 1008, 1202, 1204, 1206, 1208, 1402, 1404},
 1202: {1002, 1004, 1006, 1008, 1202, 1204, 1206, 1208, 1402, 1404},
 1204: {1002, 1004, 1006, 1008, 1202, 1204, 1206, 1208, 1402, 1404},
 1206: {1002, 1004, 1006, 1008, 1202, 1204, 1206, 1208, 1402, 1404},
 1208: {1002, 1004, 1006, 1008, 1202, 1204, 1206, 1208, 1402, 1404},
 1402: {1002, 1004, 1006, 1008, 1202, 1204, 1206, 1208, 1402, 1404},
 1404: {1002, 1004, 1006, 1008, 1202, 1204, 1206, 1208, 1402, 1404},
 1602: {1602, 1604},
 1604: {1602, 1604},
 1820: {1820, 1822},
 1822: {1820, 1822},
 2002: {2002, 2004},
 2004: {2002, 2004},
 2006: {2006},
 2008: {2008},
 2010: {2010},
 2202: {2202, 2204, 2206, 3004},
 2204: {2202, 2204, 2206, 3004},
 2206: {2202, 2204, 2206, 3004},
 2402: {2402, 2404, 3006},
 2404: {2402, 2404, 3006},
 2502: {2502},
 2602: {2602},
 2604: {2604},
 2606: {2606},
 2608: {2608},
 2802: {2802},
 2804: {2804},
 2806: {2806},
 3002: {3002},
 3004: {2202, 2204, 2206, 3004},
 3006: {2402, 2404, 3006},
 3202: {3202, 4002},
 3204: {3204, 4004},
 3206: {3206},
 3208: {3208},
 3402: {3402},
 3404: {3404},
 3406: {3406},
 3502: {3502},
 3504: {3504},
 3506: {3506},
 3602: {3602},
 3702: {3702},
 3703: {3703, 3704, 3706, 3708, 3720, 3722},
 3704: {3703, 3704, 3706, 3708, 3720, 3722},
 3706: {3703, 3704, 3706, 3708, 3720, 3722},
 3708: {3703, 3704, 3706, 3708, 3720, 3722},
 3720: {3703, 3704, 3706, 3708, 3720, 3722},
 3722: {3703, 3704, 3706, 3708, 3720, 3722},
 3802: {3802},
 4002: {3202, 4002},
 4004: {3204, 4004},
 4202: {4202, 4402},
 4204: {4204},
 4206: {4206},
 4208: {4208},
 4402: {4202, 4402},
 4404: {4404},
 4602: {4602, 4604, 4804},
 4604: {4602, 4604, 4804},
 4802: {4802},
 4804: {4602, 4604, 4804},
 5002: {5002, 5004},
 5004: {5002, 5004},
 5006: {5006},
 5008: {5008},
 5202: {5202, 5204},
 5204: {5202, 5204},
 5402: {5402, 5404},
 5404: {5402, 5404},
 5502: {5502},
 5504: {5504},
 5506: {5506},
 5702: {5702, 5704},
 5704: {5702, 5704},
 5802: {5802},
 5804: {5804},
 5806: {5806},
 6002: {6002},
 6004: {6004},
 6006: {6006},
 6008: {6008},
 6010: {6010},
 6012: {6012},
 6014: {6014},
 6016: {6016},
 6018: {6018},
 6402: {6402},
 6404: {6404},
 6406: {6406},
 6408: {6408},
 6410: {6410},
 6412: {6412},
 6414: {6414},
 6416: {6416},
 6418: {6418},
 6420: {6420},
 6422: {6422},
 6802: {6802, 6804, 6806},
 6804: {6802, 6804, 6806},
 6806: {6802, 6804, 6806},
 7002: {7002, 7004, 7006, 7008},
 7004: {7002, 7004, 7006, 7008},
 7006: {7002, 7004, 7006, 7008},
 7008: {7002, 7004, 7006, 7008},
 7102: {7102, 7104, 7106},
 7104: {7102, 7104, 7106},
 7106: {7102, 7104, 7106},
 7202: {7202},
 7204: {7204},
 7206: {7206},
 7208: {7208},
 7220: {7220},
 7302: {7302},
 7304: {7304},
 7502: {7502},
 7504: {7504},
 7506: {7506},
 7702: {7702, 7704, 7802, 7804},
 7704: {7702, 7704, 7802, 7804},
 7802: {7702, 7704, 7802, 7804},
 7804: {7702, 7704, 7802, 7804},
 8002: {8002},
 8004: {8004},
 8006: {8006, 8008},
 8008: {8006, 8008},
 8010: {8010},
 8012: {8012},
 8402: {8402, 8404, 8406},
 8404: {8402, 8404, 8406},
 8406: {8402, 8404, 8406},
 8408: {8408},
 8410: {8410},
 8412: {8412},
 8802: {8802},
 8804: {8804},
 8806: {8806},
 9002: {9002},
 9004: {9004},
 9006: {9006},
 9008: {9008},
 9010: {9010},
 9012: {9012},
 9202: {9202},
 9204: {9204},
 9402: {9402, 9404, 9406},
 9404: {9402, 9404, 9406},
 9406: {9402, 9404, 9406},
 9602: {9602},
 9802: {9802},
 9999: {9999}}

def recommend_1(df,final):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    
    tfidf = TfidfVectorizer(stop_words="english")
    vectors = tfidf.fit_transform(data["description"]).toarray()
    knn = NearestNeighbors(n_neighbors=100, algorithm='brute', metric='cosine')
    knn.fit(vectors)

    recommendations = []

    p=0

    def recommend(index):
        calories = []
        energy = 0
        remaining = cal_per_cat
        recipe_index = index
        _, indices = knn.kneighbors(vectors[recipe_index].reshape(1, -1))
        
        for i in indices[0][1::10]:
            if energy < cal_per_cat:
                if data.iloc[i].energy_kcal < remaining:
                    recommendation = f"{data.iloc[i].description}, -- calories = {data.iloc[i].energy_kcal} Kcal"
                    calories.append(data.iloc[i].energy_kcal)
                    energy += data.iloc[i].energy_kcal
                    remaining = cal_per_cat - energy
                    recommendations.append(recommendation)
        p=i
        return np.sum(calories)

    total_cal = []
    st.write("THE RECOMMENDED FOODS ARE")
    for item in final:
        energy = recommend(item)
        total_cal.append(energy)

    total_energy = np.sum(total_cal)
    error = abs(total_energy - cal) / cal
    b = p

    while error > 0.15:
        b += 1
        recommendation = f"{data.iloc[b].description}, -- calories = {data.iloc[b].energy_kcal} Kcal"
        total_energy += data.iloc[b].energy_kcal
        recommendations.append(recommendation)
        error = abs(total_energy - cal) / cal

    st.write("--------------------")
    st.write("The total calories of the recommended food ", total_energy, "Kcal")
    st.write("Error percentage between calories actually needed and recommended calories = {:.2f}%".format(error * 100))
    
    return recommendations

def Select(df3):
    final = []
    new_indexes = list(df3.index)
    randm = []
    for b in range(10):
        randm.append(random.choice(new_indexes))
    user_sel = []
    st.write("Select one among these that you prefer the most:")
    for x in randm:
        st.write("{}   -> {}".format(x, data["description"][x]))
    z = st.text_input("Enter index:", key=str(df3.shape))
    if z.strip():  # Check if z is not an empty string
        user_sel = list(map(int, z.split(" ")))
        return user_sel
    else:
        # Handle the case where z is an empty string
        return []
    return user_sel

# Streamlit UI
st.set_page_config(
    page_title="Food Recommendation App",
    page_icon=":tomato:",
    layout="wide",
)

# ... (Your existing code)

# Define your custom theme
custom_theme = """
<style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f7f7f7;
        color: #333333;
        margin: 0;
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: white;
    }
    .sidebar .sidebar-content a {
        color: #ecf0f1;
    }
    .header {
        background-color: #3498db;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .subheader {
        background-color: #2ecc71;
        color: white;
        padding: 10px;
        border-radius: 8px;
        margin-top: 10px;
    }
    .button {
        background-color: #3498db;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
</style>
"""

# Apply the custom theme
st.markdown(custom_theme, unsafe_allow_html=True)

# Streamlit UI
st.title("Food Recommendation App")
st.markdown("Enter your Bio Metrics.")

# User input for BMI
gender = st.radio("Select your gender:", ("MALE", "FEMALE"))
weight = st.number_input("Enter your weight in KG:", min_value=0)
height = st.number_input("Enter your height in CM:", min_value=0)
age = st.number_input("Enter your age:", min_value=0)

if height > 0:
    bmi = weight / ((height / 100) ** 2)
else:
    bmi = None

if bmi is not None:
    st.write(f"Your body mass index is: {bmi:.2f}")
else:
    st.write("Unable to calculate BMI. Please check your input values.")

if gender == "MALE":
    BMR_men = 93.362 + (14.597 * weight) + (5.989 * height) - (5.277 * age)
    cal = int(BMR_men * 1.2)
else:
    BMR_women = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
    cal = int(BMR_women * 1.2)

cal_per_cat = cal / 6

# Display total calories needed
st.subheader(f"Total Calories needed = {cal} Kcal/day")

# User input for food category selection
categories_range = [0, 14, 35, 82, 91, 105, 155]
food_categories = [f"{data_category['larger_category'][i]} - {data_category['wweia_food_category_description'][i]}" for i in range(len(data_category))]
selected = st.multiselect("Select food categories:", food_categories)
selected_indices = []
for category in selected:
        # Split the selected category to extract larger_category and wweia_food_category_description
        larger_category, food_category_description = category.split(" - ", 1)
        # Find the corresponding index based on the two columns
        index = data_category[(data_category['larger_category'] == larger_category) & (data_category['wweia_food_category_description'] == food_category_description)].index.tolist()
        if index:
            selected_indices.extend(index)
if len(selected) == 6:
    header_list = ["wweia_food_category_code", "wweia_food_category_description", "larger_category", "same_category"]
    df1 = pd.read_csv('wweia_food_categories_addtl.csv', skiprows=lambda x: x not in selected_indices, names=header_list)

    def category_code():
        return df1["wweia_food_category_code"]

    list_of_categories = category_code().to_list()

    new_list = []
    for j in list_of_categories:
        categ = synonims[j]
        categ = list(categ)
        new_list.append(categ)

    # Your previous code
    select_user = []
    final = []
    check = []
    food2 = []
    for x in new_list:
        for y in x:
            z = data.loc[data["wweia_category_code"]==y]
            food2.append(z)
        df3 = pd.concat(food2)
        check.append(df3)
        select_user.append(food2)
        final.append(Select(df3))
        food2 = []


else:
    st.write("Please select exactly 6 food categories.")
# Streamlit app command
if st.button("Get Food Recommendations", key="recommend_button"):
    # Perform recommendation based on selected categories
    recommendations = recommend_1(df, final)
    # Display recommendations
    st.subheader("Recommended Foods:")
    for recommendation in recommendations:
        st.success(recommendation)

# Additional Streamlit formattingS
st.sidebar.header("About")
st.sidebar.text("Welcome to our\nDIET RECOMMENDATION SYSTEM.\nEnter your details and\nget your fitness details.")
st.sidebar.text("Created by:\nAnshuman Singh\nPrerit Sharma\nRishabh Jain")
