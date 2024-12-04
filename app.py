import streamlit as st
from st_on_hover_tabs import on_hover_tabs
import pandas as pd 
from datetime import datetime,timedelta
import time
import os
from transformers import AutoFeatureExtractor, SwinForImageClassification
from PIL import Image
import requests
import numpy as np
import io
import cv2

st.set_page_config(layout="wide")


st.markdown('''
<style>
    section[data-testid='stSidebar'] {
        background-color: #111;
        min-width: unset !important;
        width: unset !important;
        flex-shrink: unset !important;
    }

    button[kind="header"] {
        background-color: transparent;
        color: rgb(180, 167, 141);
    }

    @media (hover) {
        /* header element to be removed */
        header["data"-testid="stHeader"] {
            display: none;
        }

        /* The navigation menu specs and size */
        section[data-testid='stSidebar'] > div {
            height: 100%;
            width: 95px;
            position: relative;
            z-index: 1;
            top: 0;
            left: 0;
            background-color: #111;
            overflow-x: hidden;
            transition: 0.5s ease;
            padding-top: 60px;
            white-space: nowrap;
        }

        /* The navigation menu open and close on hover and size */
        /* section[data-testid='stSidebar'] > div {
        height: 100%;
        width: 75px; /* Put some width to hover on. */
        /* } 

        /* ON HOVER */
        section[data-testid='stSidebar'] > div:hover{
        width: 300px;
        }

        /* The button on the streamlit navigation menu - hidden */
        button[kind="header"] {
            display: none;
        }
    }

    @media (max-width: 272px) {
        section["data"-testid='stSidebar'] > div {
            width: 15rem;
        }/.
    }
</style>
''', unsafe_allow_html=True)

entered_style = """
        display: flex;
        justify-content: center;
"""

st.markdown(
    """
<div style='border: 2px solid #994C00; border-radius: 5px; padding: 10px; background-color: #FFCC99;'>
    <h1 style='text-align: center; color: #FF8000; font-size:30px;'>
    🍜🥕 Food Prediction and Recommendation System 🍚🌟
    </h1>
</div>
    """, unsafe_allow_html=True)

with open("assets/css/style.css") as f:
    st.markdown(f"<style> {f.read()} </style>",unsafe_allow_html=True)
with open("assets/webfonts/font.txt") as f:
    st.markdown(f.read(),unsafe_allow_html=True)
# end def

with st.sidebar:
    tabs = on_hover_tabs(tabName=['Information','Food Allergies', 'Eating list', 'Reset'], 
    iconName=['information','search', 'list', 'refresh'], 
    styles={'navtab': {'background-color': '#111', 'color': '#818181', 'font-size': '18px', 
                    'transition': '.3s', 'white-space': 'nowrap', 'text-transform': 'uppercase'}, 
                    'tabOptionsStyle': 
                    {':hover :hover': {'color': '#FFCC99', 'cursor': 'pointer'}}, 'iconStyle': 
                    {'position': 'fixed', 'left': '7.5px', 'text-align': 'left'}, 'tabStyle': 
                    {'list-style-type': 'none', 'margin-bottom': '30px', 'padding-left': '30px'}}, 
                    key="1",default_choice=0)
    st.markdown(
    """
        <div style='border: 2px solid green; padding: 10px; white; margin-top: 5px; margin-buttom: 5px; margin-right: 20px; bottom: 50; position: buttom;'>
            <h1 style='text-align: center; color: white; font-size: 100%;'> 💻 AI Innovator Award 2023 🪙 </h1>
        </div>
    """, unsafe_allow_html=True)

df_label = pd.read_csv(".\data.csv")

if tabs == 'Information' and len(df_label) < 1:
    def calculate_age(birthdate):
        today = datetime.now()
        birthdate = datetime.strptime(birthdate, "%Y-%m-%d")
        age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
        return age

    def write_to_csv(data):
        existing_data = pd.read_csv("data.csv")
        new_data = pd.DataFrame(data, columns=existing_data.columns)
        result_df = pd.concat([existing_data, new_data], ignore_index=True)
        result_df.to_csv("data.csv", index=False)

    def main():
        # Custom styling for larger input labels
        input_label_style = "font-size: 18px; color: blue;"

        # Create two columns
        left_column, right_column = st.columns(2)

        # Input fields in the left column
        left_column.markdown(
            f'<p style="color: white; text-align: center; font-size: 20px; font-family: \'Comic Sans MS\', cursive, sans-serif;">👦 Gender 👧</p>',
            unsafe_allow_html=True
        )
        gender_options = ["Male", "Female"]
        gender = left_column.selectbox("- Select Your Gender (male/female) ? : ", options=gender_options)
        
        min_year = datetime.now() - timedelta(days=365 * 100)
        left_column.markdown(
            f'<p style="color: white; text-align: center; font-size: 20px; font-family: \'Comic Sans MS\', cursive, sans-serif;">📆 Birth 📆</p>',
            unsafe_allow_html=True
        )
        birthdate = left_column.date_input("-  Please select your date of birth. : ", min_value=min_year)
        age = calculate_age(str(birthdate))
        right_column.markdown(
            f'<p style="color: white; text-align: center; font-size: 20px; font-family: \'Comic Sans MS\', cursive, sans-serif;">🧍‍♀️ Height 🧍‍♂️</p>',
            unsafe_allow_html=True
        )
        height = right_column.number_input("- Enter your height (cm) ?: ", step=1, min_value=0)
        right_column.markdown(
            f'<p style="color: white; text-align: center; font-size: 20px; font-family: \'Comic Sans MS\', cursive, sans-serif;">⚖️ Weight ⚖️</p>',
            unsafe_allow_html=True
        )
        weight = right_column.number_input("- Enter your weight (kg) ?: ", step=1, min_value=0)

        left_column_2, center ,right_column_2 = st.columns(3)
        center.markdown(
            f'<p style="color: white; text-align: center; font-size: 20px; font-family: \'Comic Sans MS\', cursive, sans-serif;">📑 Select Eating Plan 🍴</p>',
            unsafe_allow_html=True
        )
        plan_options = ["Maintain weight", "Mild weight loss (0.25 kg/week)", "Weight loss (0.5 kg/week)", "Extreme weight loss (1 kg/week)"]
        plan = center.selectbox("- What weight loss plan do you want to be on?:", options=plan_options)

        # Button to trigger data writing
        if st.button("Submit", key="write_button",use_container_width=True):
            new_data = {
                "gender": [gender],
                "age": [age],
                "height": [height],
                "weight": [weight],
                "plan": [plan]
            }
            write_to_csv(new_data)
            st.balloons()
            st.experimental_rerun()

    if __name__ == "__main__":
        main()
    

if tabs == 'Information' and len(df_label) >= 1:
    st.markdown(
        """
    <div style='border: 2px solid #33FF33; border-radius: 5px; padding: 5px; background-color: white;'>
        <h3 style='text-align: center; color: #00CC00; font-size: 180%'> 📝 Welcome ✅ </h3>
    </div>
        """, unsafe_allow_html=True) 
    # Function to calculate BMR
    def calculate_bmr(gender, age, height, weight):
        if gender.lower() == 'male':
            bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
        elif gender.lower() == 'female':
            bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
        else:
            st.error("Invalid gender. Please enter 'Male' or 'Female'.")
            return None

        return bmr

    # Function to calculate daily calorie intake based on plan
    def calculate_daily_calories(bmr, plan):
        if plan == "Maintain weight":
            kcal = bmr * 1.2
        elif plan == "Mild weight loss (0.25 kg/week)":
            kcal = bmr * 1.2 - 250
        elif plan == "Weight loss (0.5 kg/week)":
            kcal = bmr * 1.2 - 500
        elif plan == "Extreme weight loss (1 kg/week)":
            kcal = bmr * 1.2 - 1000
        else:
            st.error("Invalid plan. Please choose a valid plan.")
            return None

        return kcal
    data = pd.read_csv('.\data.csv')  
    # Get selected row's data
    selected_row_data = data.iloc[0]
    # Display input fields with existing data
    gender = selected_row_data["gender"]
    age = selected_row_data["age"]
    height = selected_row_data["height"]
    weight = selected_row_data["weight"]
    plan = selected_row_data["plan"]
    # Calculate BMR
    bmr = calculate_bmr(gender, age, height, weight)
    st.write("")
    key_morning = hash("uploaded_file_Morning")
    key_lunch = hash("uploaded_file_Lunch")
    key_evening = hash("uploaded_file_Evening")
    left_column_3, center, right_column_3 = st.columns(3)

    # Use the generated keys for file uploaders
    left_column_3.markdown(
            f'<p style="color: white; text-align: center; font-size: 36px; font-weight: bold;"> BREAKFAST </p>',
            unsafe_allow_html=True
        )
    uploaded_file_Morning = left_column_3.file_uploader("", key=key_morning, accept_multiple_files=False)
    #Open file and save it 
    selecting_kcal = 0
    file_path = ".\labels_kcal.csv"
    df_label = pd.read_csv(file_path)
    selected_categories = left_column_3.multiselect("- Select Categories:", 
                                                    options=[f"{category} - {kcal} kcal" 
                                                            for category, kcal in zip(df_label["Category"], df_label["kcal"])],
                                                    key="categories_multiselect")
    if selected_categories:
        selected_categories = [category.split(" - ")[0] for category in selected_categories]
        filtered_df = df_label[df_label["Category"].isin(selected_categories)]
        selecting_kcal = filtered_df["kcal"].sum()

        # Button to save selected foods to "get_kcal.csv"
        if left_column_3.button("Enter", use_container_width=True):
            selected_foods = filtered_df[["Category", "kcal"]]

            # Add a "date" column with the current timestamp
            selected_foods["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            selected_foods["time"] = "BREAKFAST"
            # Save to "get_kcal.csv"
            csv_filename = "get_kcal.csv"
            mode = "a" if os.path.exists(csv_filename) else "w"
            header = not os.path.exists(csv_filename)
            selected_foods.to_csv(csv_filename, mode=mode, index=False, header=header)

            with st.spinner("Selected foods saved"):
                selecting_kcal = 0
                time.sleep(1.25)
    get_kcal_file_path = "get_kcal.csv"
    df_get_kcal = pd.read_csv(get_kcal_file_path)
    total_kcal = df_get_kcal["kcal"].sum()
    daily_calories = calculate_daily_calories(bmr, plan)
    daily_calories = daily_calories - total_kcal - selecting_kcal
    center.markdown(
            f'<p style="color: white; text-align: center; font-size: 36px; font-weight: bold;"> LUNCH </p>',
            unsafe_allow_html=True
        )
    uploaded_file_Lunch = center.file_uploader("", key=key_lunch, accept_multiple_files=False)
    file_path = ".\labels_kcal.csv
    df_label = pd.read_csv(file_path)
    category_sum = df_label.groupby('Category')['kcal'].sum()
    selected_categories = pd.Series()
    total_kcal_top3 = 0
    while total_kcal_top3 < daily_calories / 2 and len(selected_categories) < 3:
        random_category = category_sum.sample()
        selected_categories = selected_categories.append(random_category)
        total_kcal_top3 = selected_categories.sum()

    center.write(selected_categories)


    right_column_3.markdown(
            f'<p style="color: white; text-align: center; font-size: 36px; font-weight: bold;"> DINNER </p>',
            unsafe_allow_html=True
        )
    uploaded_file_Evening = right_column_3.file_uploader("", key=key_evening, accept_multiple_files=False)
    file_path = ".\labels_kcal.csv
    df_label = pd.read_csv(file_path)
    category_sum = df_label.groupby('Category')['kcal'].sum()
    selected_categories = pd.Series()
    total_kcal_top3 = 0
    while total_kcal_top3 < daily_calories / 2 and len(selected_categories) < 3:
        random_category = category_sum.sample()
        selected_categories = selected_categories.append(random_category)
        total_kcal_top3 = selected_categories.sum()

    right_column_3.write(selected_categories)

    if bmr is not None:
        # Calculate daily calories based on plan
        daily_calories = calculate_daily_calories(bmr, plan)
        daily_calories = daily_calories - total_kcal - selecting_kcal
        st.markdown(
            f"""
            <style>
                .bottom-text {{
                    position: fixed;
                    bottom: 0;
                    left: 50%;
                    transform: translateX(-50%);
                    width: 100%;
                    text-align: center;
                    border: 2px solid #FFFF00;
                    border-radius: 5px;
                    padding: 5px;
                    background-color: white;
                }}
            </style>
            <div class="bottom-text">
                <h3 style='text-align: center; color: #FF8000; font-size: 180%'>Your Energy Gap : {daily_calories:.2f} kcal</h3>
            </div>
            """, unsafe_allow_html=True
        )


if tabs == "Food Allergies":
    st.markdown(
        """
    <div style='border: 2px solid #FF007F; border-radius: 5px; padding: 5px; background-color: white;'>
        <h3 style='text-align: center; color: #000000; font-size: 180%'> 🦐 Check food allergy information 🍲 </h3>
    </div>
        """, unsafe_allow_html=True) 
    uploaded_files = st.file_uploader(" x", 
        type=["jpg", "jpeg", "png", "dcm"], accept_multiple_files=True)

    if uploaded_files is not None:
        processor = AutoFeatureExtractor.from_pretrained('Santipab/Test_100_Food_AI_Innovation')
        model = SwinForImageClassification.from_pretrained('Santipab/Test_100_Food_AI_Innovation')
        answer = []
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.getvalue()
            img = Image.open(io.BytesIO(file_bytes))
            img_out = img.resize((224,224))
            img_out = np.array(img_out)
            image = img.convert('RGB')
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            # model predicts one of the 1000 ImageNet classes
            predicted_class_idx = logits.argmax(-1).item()
            print("Predicted class:", model.config.id2label[predicted_class_idx])
            answer.append(model.config.id2label[predicted_class_idx])
            base_folder_path = '.\Train_Food2k'
            folder_name = answer[0]
            folder_path = os.path.join(base_folder_path, folder_name)
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            selected_images = image_files[:5]

            st.markdown(
                """
            <div style='border: 2px solid #00CC00; border-radius: 5px; padding: 5px; background-color: #99FF99;'>
                <h3 style='text-align: center; color: #000000; font-size: 180%'> 🔎 Prediction 🔍 </h3>
            </div>
                """, unsafe_allow_html=True) 
            st.markdown(" ")

            col1, col2, col3, col4, col5 = st.columns(5)
            for i, image_file in enumerate(selected_images):
                image_path = os.path.join(folder_path, image_file)
                image = cv2.imread(image_path)
                size = min(image.shape[0], image.shape[1])
                cropped_image = image[:size, :size, :]
                image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                resized_image = cv2.resize(image_rgb, (256, 256))
                pil_image = Image.fromarray(resized_image)
                col = [col1, col2, col3, col4, col5][i]
                col.image(pil_image, use_column_width=True)
            labels_df = pd.read_csv('labels_kcal.csv')
            desired_id = answer[0]
            desired_id = int(desired_id)
            row = labels_df[labels_df['id'] == desired_id]
            if not row.empty:
                category_value = row.iloc[0]['Category']
                st.markdown(
                    f"""
                    <div style='border: 2px solid #994C00; border-radius: 5px; padding: 5px; background-color: white;'>
                        <h3 style='text-align: center; color: #FF8000; font-size: 180%'> Category : {category_value} </h3>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(
                    f"""
                    <div style='border: 2px solid #FF3333; border-radius: 5px; padding: 5px; background-color: white;'>
                        <h3 style='text-align: center; color: #FF0000; font-size: 180%'> ❌ ID not found </h3>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown(" ")
            df = pd.read_csv("Allergens.csv")
            allergen_info = df.loc[int(answer[0]), "Allergens"]

            if allergen_info != "No allergen information available":
                color = "red"
            else:
                color = "#33FF33"

            st.markdown(
                f"""
                <div style='border: 2px solid {color}; border-radius: 5px; padding: 5px; background-color: white;'>
                    <h3 style='text-align: center; color: {color}; font-size: 180%'> {allergen_info} </h3>
                </div>
                """,
                unsafe_allow_html=True
            )




if tabs == "Eating list":
    csv_file_path = "get_kcal.csv"
    df = pd.read_csv(csv_file_path)
    st.markdown(
        """
    <div style='border: 2px solid #FF007F; border-radius: 5px; padding: 5px; background-color: #FF99FF;'>
        <h3 style='text-align: center; color: #000000; font-size: 180%'> 📃 Eating Lists 🍉 </h3>
    </div>
        """, unsafe_allow_html=True) 
    st.table(df)

if tabs == "Reset":
    df = pd.read_csv('get_kcal.csv')
    df = pd.DataFrame(columns=df.columns)
    df.to_csv('get_kcal.csv', index=False)
    st.markdown(
            """
        <div style='border: 2px solid #00FFFF; border-radius: 5px; padding: 5px; background-color: white;'>
            <h3 style='text-align: center; color: blue; font-size: 180%'> 🔃 The information has been cleared. ✅ </h3>
        </div>
            """, unsafe_allow_html=True)



