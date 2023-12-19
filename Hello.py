import streamlit as st
import matplotlib.pyplot as plt
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("streamlit-ml-pa-06731684e124.json", scope)
client = gspread.authorize(creds)
spreadsheet = client.open("Streamlit ML Personality Assessment")  # Replace with your spreadsheet name

def personality_detection(text):
    tokenizer = BertTokenizer.from_pretrained("Minej/bert-base-personality")
    model = BertForSequenceClassification.from_pretrained("Minej/bert-base-personality")

    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    outputs = model(**inputs)
    predictions = outputs.logits.squeeze().detach().numpy()

    label_names = ['Extroversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    result = {label_names[i]: predictions[i] for i in range(len(label_names))}

    return result

def radar_chart(personality_prediction):
    labels = list(personality_prediction.keys())
    values = list(personality_prediction.values())

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Include the first element of the list to close the circular graph
    values += [values[0]]
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True, facecolor='white'))  # Set background color to white

    ax.plot(angles, values, color='blue', linewidth=2, linestyle='solid')
    ax.fill(angles, values, color='blue', alpha=0.4)

    # Add radial gridlines
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color='black')  # Set labels color to black

    # Add range numbers on the radar chart
    range_numbers = np.linspace(0, 1, 5)
    ax.set_yticks(range_numbers)
    ax.set_yticklabels([f"{num:.1%}" for num in range_numbers], color='black')  # Set range numbers color to black

    # Remove the outer box (spines)
    ax.spines['polar'].set_visible(False)

    plt.title("Personality Traits Radar Chart", size=16, color='black', y=1.1)  # Set title color to black

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

    # Collect answers
    # answers = []
    # for i, question in enumerate(questions, start=1):
    #     st.markdown("--------------------------------------------------------------")
    #     st.write(f"**{i}**. {question}")
        
    #     answer = st.radio("", ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"], key=f"question_{i}", index=None, horizontal=True)
    #     answers.append(answer)

    # # Merge questions and answers in one sentence
    # merged_responses = " ".join([f"{a} with that {q}" for q, a in zip(questions, answers)])
    reverse_questions = [5, 8, 10, 11, 12, 18, 19]

    answers = []

    for i, question in enumerate(questions, start=1):

    st.markdown("--------------------------------------------------------------")
    st.write(f"**{i}**. {question}")
    
    answer = st.radio("", ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"], 
                      key=f"question_{i}", index=None, horizontal=True)
                      
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
    if None in answers:
        st.error("Please answer all 20 questions before submitting.")
    else:
        # Display the button with the disabled state from session state
        submit_button = st.button("Submit", key="Submit", disabled=st.session_state.disabled)
        # Check if the button is clicked
        if submit_button:
            # Update session state to disable the button
            st.session_state.disabled = True
            st.success("Thank you for completing the questionnaire!")
            
            # Display merged responses
            st.write("Your Responses:")
            st.write(merged_responses)

            # Perform personality detection
            personality_prediction = personality_detection(merged_responses)
            
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

# # To hide "fork my app on github" icon
# hide_github_icon = """

# .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob, .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137, .viewerBadge_text__1JaDK{ display: none; } #MainMenu{ visibility: hidden; } footer { visibility: hidden; } header { visibility: hidden; }
# """
# st.markdown(hide_github_icon, unsafe_allow_html=True)
