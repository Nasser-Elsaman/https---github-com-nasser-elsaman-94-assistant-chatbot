# Start of the project

import streamlit as st
from streamlit_option_menu import option_menu
import torch
import matplotlib.pyplot as plt
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("streamlit-ml-pa-06731684e124.json", scope)
client = gspread.authorize(creds)
spreadsheet = client.open("Streamlit LLMs Personality Assessment")  # Replace with your spreadsheet name

# Sidebar
st.sidebar.title(":hammer_and_wrench: Configuration")

selected = option_menu (menu_title=None, options= ["Home", "Project", "About"], icons= ["house", "book", "file-person"],
    menu_icon = "cast", default_index=0, orientation = "horizontal",
    styles= {"container": {"padding": "0!important", "background-color": "#5c0303"},
             "icon": {"color": "orange", "font-size": "15px"},
             "nav-link": {"font-size": "15px", "text-align": "center", "margin":"0px", "--hover-color": "#aba9a9"},
             "nav-link-selected": {"background-color": "#121212"}})

if selected == "Home":
    st.title ("Welecome to LLMs-Based Personality Assessment :brain: :bar_chart:")
    url = "https://www.researchgate.net/publication/7014171_The_Mini-IPIP_Scales_Tiny-yet-Effective_Measures_of_the_Big_Five_Factors_of_Personality"
    st.write(":white_check_mark: This project is based on The Mini IPIP personality measure, continue reading to know why I chose this assessment and how it is worth it.")
    st.write(":white_check_mark: The following are some main arguments for why the Mini-IPIP personality test could be a wise option:")
    st.write(":one: It is a short 20-item scale compared to longer Big Five inventories, making it faster to complete. This can improve user experience.")
    st.write(":two: In testing, the Mini-IPIP has shown strong psychometric qualities, including factorial validity, convergent validity with larger measures, and reliability. Therefore, despite its briefness, it properly assesses the Big Five variables.")
    st.write (":three: The brief length makes it suitable for situations where assessment time is limited, like surveys, research studies, or screening.")
    st.write (":four: It is a well-respected short scale, having been utilised and mentioned in hundreds of published investigations.")
    st.write (":white_check_mark: To sum up, the Mini-IPIP is a well-researched, reliable, and valuable brief Big Five assessment that may be a suitable option when simplicity and convenience of use are top concerns. Its correctness is supported by the facts, even with its short length.")
    st.write (":white_check_mark: For more details, please check this research paper:- [The Mini-IPIP Scales: Tiny-yet-Effective Measures of the Big Five Factors of Personality.](%s)" % url)
    
    st.write (":sparkles: _To start the assessment please make sure to activate model in the sidebar (Upper Left Arrow)_ :arrow_forward:")          

st.sidebar.write (":ok: Activate Model To Start:- ")
show_assessment = st.sidebar.toggle(":medal: Nasserelsaman/microsoft-finetuned-personality")
model_link = "https://huggingface.co/microsoft/MiniLM-L12-H384-uncased"
st.sidebar.write (":100: To Fine-tune my model; I depended on the pretrained Model (Base Model):- [microsoft/MiniLM-L12-H384-uncased.](%s)" %model_link)
my_model = "https://huggingface.co/Nasserelsaman/microsoft-finetuned-personality"
st.sidebar.write(":male-technologist: Check out My Fine-tuned model:- [Nasserelsaman/microsoft-finetuned-personality.](%s)" %my_model)
if selected == "Project":
    if show_assessment == False:
        st.write(":sparkles: _To start the assessment please make sure to activate model in the sidebar (Upper Left Arrow)_ :arrow_forward:")
    else:
        with st.spinner('Loading...'):
            time.sleep(3)
        with st.spinner('In progress...'):
            time.sleep(2)
        st.sidebar.write(':balloon: Model Activated successfuly! Assessement Ready Now!')
        st.sidebar.write (":cool: Instructions: The questions that characterize people's actions are listed now. Please score how well each question represents you using the scoring scale [Strongly Disagree - Disagree - Neutral - Agree - Strongly Agree]. Give a brief description of who you are today, not who you hope to become. Give a candid description of yourself in comparison to other persons around your age and of the same sex. After carefully reading each question, please indicate to what extent you believe it characterizes you by placing an accurate choice.")
        quest_link = "https://rockford.instructure.com/courses/8365/files/419203/download?verifier=dCgoMhAAmj7FnEqVb4mKBEHZ1ia3Uh4YCSBtwwLX&wrap=1"
        st.sidebar.write (":books: _Kindly note that this project is storing assessment data results for research purposes only, and to ensure from the original questions and learn how to calculate your score, check out this PDF file:-_ [The Mini-IPIP Scale (Donnellan, Oswald, Baird, & Lucas).](%s)" %quest_link)
        def personality_detection(text, threshold=0.05, endpoint= 1.0):
            token="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"
            tokenizer = AutoTokenizer.from_pretrained("Nasserelsaman/microsoft-finetuned-personality",token=token)
            model = AutoModelForSequenceClassification.from_pretrained("Nasserelsaman/microsoft-finetuned-personality",token=token)
            
            inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
            outputs = model(**inputs)
            predictions = outputs.logits.squeeze().detach().numpy()
        
            # Get raw logits
            logits = model(**inputs).logits
        
            # Apply sigmoid to squash between 0 and 1
            probabilities = torch.sigmoid(logits)
        
            # Set values less than the threshold to 0.05
            predictions[predictions < threshold] = 0.05
            predictions[predictions > endpoint] = 1.0
        
            label_names = ['Agreeableness', 'Conscientiousness', 'Extraversion', 'Neuroticism', 'Openness']
            result = {label_names[i]: f"{predictions[i]*100:.0f}%" for i in range(len(label_names))}
        
            return result
            
        def radar_chart(personality_prediction):
            # Create empty list 
            labels = []
            values = []
            # Iterate through dict items
            for trait, pred in personality_prediction.items():
        
            # Extract just the number 
              label= str(trait)
              num = float(pred.rstrip('%'))
          
            # Append number to list
              labels.append(label)
              values.append(num)
        
            
            num_vars = len(labels)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        
            # Include the first element of the list to close the circular graph
            values += [values[0]]
            angles += [angles[0]]
            
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True, facecolor='white'))
            
            # Set background color to white
            ax.plot(angles, values, color='blue', linewidth=3, linestyle='solid')
            ax.fill(angles, values, color='blue', alpha=0.3)
            
            # Add radial gridlines
            ax.set_yticklabels([])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, color='black') # Set labels color to black
            
            # Add range numbers on the radar chart
            range_numbers = np.linspace(0, 100, 5)
            ax.set_yticks(range_numbers)
            ax.set_yticklabels([f"{num/100:.1%}" for num in range_numbers], color='black') # Set range numbers color to black
            
            # Remove the outer box (spines)
            ax.spines['polar'].set_visible(False)
            
            plt.title("Personality Traits Radar/ Spider Web Chart \u2745 \u270F \u2713", size=16, color='black', y=1.1) # Set title color to black
            
            # Footer
            figtext_x = 0.5
            figtext_y = 0.05
            figtext_text = "This Data Visualization Chart is created by Nasser Elsaman for the result of the personality assessment traits (20 questions) of the user!"
            plt.figtext(figtext_x, figtext_y, figtext_text, fontsize=12, ha='center', va='center', color='black')
            
            st.pyplot(fig)
        
        def questionnaire():
            st.title("Personality Assessment")
        
            # Introduction
            st.subheader("Please fill out the following questionnaire to help us understand your preferences.")
        
            # Questions
            questions = [
                "I am the life of the party.",
                "I sympathize with others’ feelings.",
                "I get chores done right away.",
                "I have frequent mood swings.",
                "I have a vivid imagination.",
                "I don’t talk a lot.",
                "I am not interested in other people’s problems.",
                "I often forget to put things back in their proper place.",
                "I am relaxed most of the time.",
                "I am not interested in abstract ideas.",
                "I talk to a lot of different people at parties.",
                "I feel others’ emotions.",
                "I like order.",
                "I get upset easily.",
                "I have difficulty understanding abstract ideas.",
                "I keep in the background.",
                "I am not really interested in others",
                "I make a mess of things.",
                "I seldom feel blue.",
                "I do not have a good imagination."
            ]
        
            reverse_questions = [6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20]
            answers = []
        
            for i, question in enumerate(questions, start=1):
                st.markdown("--------------------------------------------------------------")
                st.write(f"**{i}**. {question}")
        
                answer = st.radio("", ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"], key=f"question_{i}", index=None, horizontal=True)
        
                # Check if the current question needs reversing
                if i in reverse_questions:
                    if answer == "Strongly Disagree":
                        answer = "Strongly Agree"
                    elif answer == "Disagree":
                        answer = "Agree"
                    elif answer == "Agree":
                        answer = "Disagree"
                    elif answer == "Strongly Agree":
                        answer = "Strongly Disagree"
        
                answers.append(answer)
        
            merged_responses = " ".join([f"{a} with that {q}" for q, a in zip(questions, answers)])
        
         
            # Submit button
            if 'disabled' not in st.session_state:
                st.session_state.disabled = False
            
            # Check if all questions are answered
                # Check if all questions are answered
            if None in answers:
                st.error("Please answer all 20 questions before submitting.")
            else:
            # Display the button with the disabled state from session state
                submit_button = st.button("Submit", key="Submit", disabled=st.session_state.disabled)
                # Check if the button is clicked
                if submit_button:
                    # Update session state to disable the button
                    # st.session_state.disabled = True
                    with st.spinner('Loading...'):
                        time.sleep(3)
                    with st.spinner('In progress...'):
                        time.sleep(2)
                    with st.spinner('The result will appear in seconds...'):
                        time.sleep(3)
                        st.balloons ()
                        st.success("Thank you for completing the questionnaire!")
                        
                        # Display merged responses
                        st.write("Your Responses:")
                        st.write(merged_responses)
            
                        # Perform personality detection
                        personality_prediction = personality_detection(merged_responses, threshold=0.05, endpoint= 1.0)
                        
                        # Display personality predictions
                        st.write("Personality Predictions:")
                        st.write(personality_prediction)
                        # Draw radar chart
                        radar_chart(personality_prediction)
                        sheet = spreadsheet.sheet1
                        # Append answers as one row
                        sheet.append_row(answers)       
        if __name__ == "__main__":
            questionnaire()
        
if selected == "About":
    st.divider()
    st.header(":mortar_board: Overview")
    st.write(":bookmark_tabs: Kindly note that this project is storing assessment data results for research purposes only and it is the third out of 3 projects. If you need to check the other 2 projects:-")
    Google_Form_link = "https://docs.google.com/forms/d/e/1FAIpQLSd6Cqq1uknZ27wMYVZsYhIu0asUz5sI4WQ8m4sXHKsdWlqfUA/viewform?pli=1"
    st.write(":one: Rule-based Personality Assessment using Google Form and Google Script (IF-Then Rule) with the data stored in a Google Sheet for research purposes only [(link).](%s)" % Google_Form_link)
    ml_link = "https://nasserelsaman.pythonanywhere.com/model-based"
    st.write(":two: ML-based Personality Assessment using Flask, python, and its libraries to create a model, uploaded on pythonanywhere with The data stored in a Google Sheet for research purposes only [(link).](%s)" % ml_link)
    st.divider()
    st.header(":mailbox: Get In Touch With Me!")
    st.write(":calling: Phone Numbers:- (+2) 0155 385 6595 / (+2) 0128 205 4321.")
    st.write(":email: Emails:- nasser.elsaman1994@gmail.com / nasser.mohamed2012@feps.edu.eg.")
    st.write (":house_with_garden: Addresses:- El Eshreen Street, ElSalam Area, Cairo Governorate, Egypt. / El Mostaqbal Area, Faisal District, Suez Governorate, Egypt.")
    st.write(":globe_with_meridians: Website:- https://elsamaninfo.wordpress.com.")
    st.divider()
    
# Footer Format
footer="""<style>
a:link , a:visited{
color: blue;
background-color: black;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: black;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: black;
color: red;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ❤ by <a style='display: block; text-align: center;' href="https://elsamaninfo.wordpress.com/" target="_blank">Nasser Elsaman</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

# Hamburger Menu
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


st.markdown("""
<style>
[data-testid="stSidebar"] {
 background-color: #5c0303;
 opacity: 1;
} 
</style>
""", unsafe_allow_html=True)

# End of the project
