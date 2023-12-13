import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

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

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.plot(angles, values, color='blue', linewidth=2, linestyle='solid')
    ax.fill(angles, values, color='blue', alpha=0.4)

    # Add radial gridlines
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Add range numbers on the radar chart
    range_numbers = np.linspace(0, 1, 5)
    ax.set_yticks(range_numbers)
    ax.set_yticklabels([f"{num:.1%}" for num in range_numbers])

    plt.title("Personality Traits Radar Chart", size=16, color='blue', y=1.1)

    st.pyplot(fig)

def questionnaire():

    st.header("Personality Questionnaire", divider="rainbow")

    # Introduction
    st.subheader("Please fill out the following questionnaire to help us understand your preferences.")

    # Questions
    questions = [
        "I am the life of the party",
        "I sympathize with others’ feelings",
        "I get chores done right away",
        "I have frequent mood swings",
        "I have a vivid imagination",
        "I don’t talk a lot",
        "I am not interested in other people’s problems",
        "I often forget to put things back in their proper place",
        "I am relaxed most of the time",
        "I am not interested in abstract ideas",
        "I talk to a lot of different people at parties",
        "I feel others’ emotions",
        "I like order",
        "I get upset easily",
        "I have difficulty understanding abstract ideas",
        "I keep in the background",
        "I am not really interested in others",
        "I make a mess of things",
        "I seldom feel blue",
        "I do not have a good imagination"
    ]

    # Collect answers
    answers = []
    for i, question in enumerate(questions, start=1):
        st.markdown("--------------------------------------------------------------")
        st.write(f"**{i}**. {question}")
        
        answer = st.radio("", ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"], key=f"question_{i}", index=None, horizontal=True)
        answers.append(answer)

    # Merge questions and answers in one sentence
    merged_responses = " ".join([f"{a} with that {q}" for q, a in zip(questions, answers)])

    # Check if all questions are answered
    if None in answers:
        st.error("Please answer all 20 questions before submitting.")
    else:
        # Submit button
        if st.button("Submit"):
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

if __name__ == "__main__":
    questionnaire()
