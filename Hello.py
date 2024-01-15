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
spreadsheet = client.open("Streamlit ML Personality Assessment")  # Replace with your spreadsheet name

# Sidebar
st.sidebar.title("Configuration")

selected = option_menu (menu_title=None, options= ["Home", "Project", "About"], icons= ["house", "book", "file-person"],
    menu_icon = "cast", default_index=0, orientation = "horizontal",
    styles= {"container": {"padding": "0!important", "background-color": "#5c0303"},
             "icon": {"color": "orange", "font-size": "15px"},
             "nav-link": {"font-size": "15px", "text-align": "center", "margin":"0px", "--hover-color": "#aba9a9"},
             "nav-link-selected": {"background-color": "#121212"}})


if selected == "Home":
    st.title ("Welecome to LLMs-Based Personality Assessment :brain: :bar_chart:")
    url = "https://www.researchgate.net/publication/7014171_The_Mini-IPIP_Scales_Tiny-yet-Effective_Measures_of_the_Big_Five_Factors_of_Personality"
    st.text(":white_check_mark: This project is based on Mini IPIP personality measure, check out this article to know why I chose this assessment and how it is worth it??")
    st.text(":white_check_mark: The following are some main arguments for why the Mini-IPIP personality test could be a wise option:")
    st.text(":one: It is a short 20-item scale compared to longer Big Five inventories, making it faster to complete. This can improve user experience.")
    st.text(":two: In testing, the Mini-IPIP has shown strong psychometric qualities, including factorial validity, convergent validity with larger measures, and reliability. Therefore, despite its briefness, it properly assesses the Big Five variables.")
    st.text (":three: The brief length makes it suitable for situations where assessment time is limited, like surveys, research studies, or screening.")
    st.text (":four: It is a well-respected short scale, having been utilised and mentioned in hundreds of published investigations.")
    st.text (":white_check_mark: To sum up, the Mini-IPIP is a well-researched, reliable, and valuable brief Big Five assessment that may be a suitable option when simplicity and convenience of use are top concerns. Its correctness is supported by the facts, even with its short length.")
    st.text (":white_check_mark: For more details, please check this research paper:- [The Mini-IPIP Scales: Tiny-yet-Effective Measures of the Big Five Factors of Personality.](%s)" % url)
    st.divider()
    st.text(":sparkles: _To start the assessment please activate model in the sidebar (Upper left arrow)_:arrow_forward:",)             

st.sidebar.text ("Activate Model To Start'")
show_assessment = st.sidebar.toggle("Nasserelsaman/microsoft-finetuned-personality")
model_link = "https://huggingface.co/microsoft/MiniLM-L12-H384-uncased"
st.sidebar.text ("To Fine-tune my model; I depended on The pretrained Model (Base Model) [microsoft/MiniLM-L12-H384-uncased.] (%s)" % model_link)
if selected == "Project":
    if show_assessment == False:
        st.text(":sparkles: _To start the assessment please activate model in the sidebar (Upper left arrow)_:arrow_forward:",)
    else:
        with st.spinner('Loading...'):
            time.sleep(3)
        with st.spinner('In progress...'):
            time.sleep(2)
        st.sidebar.write('Model Activated successfuly! Assessement Ready Now!')
        
        def personality_detection(text, threshold=0.05, endpoint= 1.0):
            tokenizer = AutoTokenizer.from_pretrained("Nasserelsaman/microsoft-finetuned-personality",token="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")
            model = AutoModelForSequenceClassification.from_pretrained("Nasserelsaman/microsoft-finetuned-personality",token="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")
            
            inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
            outputs = model(**inputs)
            predictions = outputs.logits.squeeze().detach().numpy()
        
            # Get raw logits
            logits = model(**inputs).logits
        
            # Apply sigmoid to squash between 0 and 1
            probabilities = torch.sigmoid(logits)
        
            # Set values less than the threshold to zero
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
            figtext_text = "This Data Visualization Chart Created by Nasser Elsaman for the result of the personality assessment traits (20 questions) by the user!"
            plt.figtext(figtext_x, figtext_y, figtext_text, fontsize=12, ha='center', va='center', color='black')
            
            st.pyplot(fig)
        
        # def radar_chart(personality_prediction):
        #   # Create empty list 
        #   traits = []
        #   values = []
        #   # Iterate through dict items
        #   for trait, pred in personality_prediction.items():
        
        #   # Extract just the number 
        #       label= str(trait)
        #       num = float(pred.rstrip('%'))
          
        #   # Append number to list
        #       traits.append(label)
        #       values.append(num)
            
        #   N = len(traits)
        #   angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
          
        #   values += values[:1]
        
        #   # Determining the angle of each spoke
        #   angles = [n / float(N) * 2 * np.pi for n in range(N)]
        #   angles += angles[:1]
        
        #   # Initialize the polar plot
        #   fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True, facecolor='white'))
          
        #   # Draw one axe per variable + add labels
        #   plt.xticks(angles[:-1], traits, color='grey', size=10)
        #   ax.tick_params(axis='x', which='major', pad=15)
          
        #   # Plot data
            
        #   ax.plot(angles, values, linewidth=1, linestyle='solid',color="green")
        #   ax.fill(angles, values, color='blue', alpha=0.3)
        #   # # Add range numbers on the radar chart
        #   range_numbers = np.linspace(0, 1, 5)
        #   ax.set_yticks(range_numbers)
        #   # ax.set_yticklabels([f"{num:.1%}" for num in range_numbers], color='black') # Set range numbers color to black
        
        #   # Fill area
        #   ax.fill(angles, values, "yellow" , alpha=0.2)
        #   ax.spines['polar'].set_visible(False)
        #   plt.title("Personality Traits Radar/ Spyder Chart", size=16, color='black', y=1.1) # Set title color to black
        #   st.pyplot(fig)
        
        # def circular_barplot(personality_prediction):
        
        #   # Get data
        #   labels = list(personality_prediction.keys()) 
        #   values = list(personality_prediction.values())
        
        #   # Calculate angles
        #   num_vars = len(labels)
        #   angles = np.linspace(0.05, 2*np.pi- 0.05, num_vars, endpoint=False)
        #   # Figure 
        #   fig, ax = plt.subplots(figsize=(8,8), subplot_kw={'polar': True})
        #   fig.set_facecolor('black')
        #   ax.set_facecolor('white')
            
        #   # Bars
        #   bars = ax.bar(angles, values, width=0.5, color='#645F8C')
            
        #   # Bar labels
        #   for bar, angle, label in zip(bars, angles, labels):
        #     rotation = np.rad2deg(angle)
        #     alignment = 'center' if -90 < rotation < 90 else 'right'
        #     ax.text(angle, 1.1, label, ha=alignment, va='center', rotation=rotation, rotation_mode='anchor', color='w', fontsize=12)
              
        #   # Title
        #   ax.set_title("Personality Traits", pad=25, fontsize=18, y=1.12, color='#4B3F6B')
            
        #   # Background
        #   ax.patch.set_alpha(0)
        #   ax.set_theta_offset(np.pi / 2)
        #   ax.set_theta_direction(-1)
            
        #   # Remove axes
        #   # Remove ticks and labels
        #   ax.set_xticks(angles)
        #   # ax.set_xticklabels(labels, size=13)
        #   ax.xaxis.grid(False)
            
        #   ax.set_yticklabels([])
        #   ax.set_yticks([0, 25, 50, 75, 100])
        
            
        #   # Show plot
        #   st.pyplot(fig)
        
          # # Create figure
          # fig, ax = plt.subplots(figsize=(9,12), subplot_kw={"projection": "polar"})
          # fig.patch.set_facecolor("white")
          # ax.set_facecolor("white")
          # ax.set_theta_offset (1.2 * np.pi/2)
          # ax.set_ylim (-1500, 3500)
        
          # ax.bar (angles, values, alpha=0.9, width= 0.52, zorder=11)
          # # ax.vlines (angles, 3000, color = "white", ls= (0, (4,4)), zorder=11)
        
        
          # # # Draw bars
          # # bars = ax.bar(angles, values, width=0.5, bottom=0.1)
        
          # # # Customize bars
          # # for bar, angle, label in zip(bars, angles, labels):
          # #   bar.set_facecolor('#4C72B0') 
          # #   bar.set_alpha(0.8)
          # #   ax.text(angle, 0.35, label, ha='center', va='center')
        
          # # Remove ticks and labels
          # ax.set_xticks(angles)
          # ax.set_xticklabels(labels, size=13)
          # ax.xaxis.grid(False)
            
          # ax.set_yticklabels([])
          # ax.set_yticks([0, 25, 50, 75, 100])
          # ax.spines ["start"].set_color ("none")
          # ax.spines ["polar"].set_color ("none")
          # XTICKS = ax.xaxis.get_major_ticks()
          # for tick in XTICKS:
          #   tick.set_pad (10)
          # PAD = 10
          # ax.text (-0.2 * np.pi/2, 25 + PAD, "25", ha= "center", size = 12)
          # ax.text (-0.2 * np.pi/2, 25 + PAD, "50", ha= "center", size = 12)
          # ax.text (-0.2 * np.pi/2, 25 + PAD, "75", ha= "center", size = 12)
          # ax.text (-0.2 * np.pi/2, 25 + PAD, "100", ha= "center", size = 12)
          # # Set title
          # ax.set_title("Personality Traits", size=18, y=1.08)
          # st.pyplot(fig) 
        
          # # Show plot
          # plt.show()
         
        
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
        
            # Collect answers
            # answers = []
            # for i, question in enumerate(questions, start=1):
            #     st.markdown("--------------------------------------------------------------")
            #     st.write(f"**{i}**. {question}")
                
            #     answer = st.radio("", ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"], key=f"question_{i}", index=None, horizontal=True)
            #     answers.append(answer)
        
            #     Merge questions and answers in one sentence
            # merged_responses = " ".join([f"{a} with that {q}" for q, a in zip(questions, answers)])
        
            # reverse_questions = [5, 8, 10, 11, 12, 18, 19]
            # answers = []
        
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
                submit_button = st.button("Submit", key="Submit") #, disabled=st.session_state.disabled)
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
    st.header(":mailbox: Get In Touch With Me!")
    
# Footer Format
footer="""<style>
a:link , a:visited{
color: blue;
background-color: black;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: blacck;
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
[data-testid="stsidebar"] {
 # background-color: #5c0303;
  background-color: rgba(76, 175, 80, 1);
  opacity: 1;
} 
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style> [data-testid=stSidebar] {background-color: #ff000050;}
</style>
""", unsafe_allow_html=True)

# # To hide "fork my app on github" icon
# hide_github_icon = """

# .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob, .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137, .viewerBadge_text__1JaDK{ display: none; } #MainMenu{ visibility: hidden; } footer { visibility: hidden; } header { visibility: hidden; }
# """
# st.markdown(hide_github_icon, unsafe_allow_html=True)
