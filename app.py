import streamlit as st
import os
import base64
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from deep_translator import GoogleTranslator
from langdetect import detect
from gtts import gTTS

# Custom theme and configuration
st.set_page_config(page_title="Welcome to KidzCareHub, I'm Rhea", page_icon="👶", layout="wide")

# Theme and voice preference
if 'theme' not in st.session_state:
    st.session_state.theme = "light"
if 'voice_enabled' not in st.session_state:
    st.session_state.voice_enabled = True

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

def get_custom_css():
    if st.session_state.theme == "light":
        return """
        <style>
            /* Light theme styles */
            @import url('https://fonts.googleapis.com/css2?family=Comic+Neue:wght@400;700&display=swap');
            body, .main, .stSidebar, [data-testid="stSidebar"] {
                font-family: 'Comic Neue', cursive !important;
                font-size: 18px !important;
                background-color: #FFF0F5;
                color: #333333;
            }
            .stButton > button {
                background-color: #FF69B4;
                color: white;
                border-radius: 20px;
                font-size: 20px;
                font-family: 'Comic Neue', cursive !important;
                border: 2px solid #FF1493;
            }
            .stTextInput > div > div > input, .stTextArea > div > div > textarea {
                background-color: #FFE4E1;
                border-radius: 10px;
                border: 2px solid #FF69B4;
                color: #333333;
                font-family: 'Comic Neue', cursive !important;
                font-size: 18px !important;
            }
            h1, h2, h3 {
                color: #FF1493;
                font-family: 'Comic Neue', cursive !important;
                font-weight: 700;
            }
            .stSidebar {
                background-color: #FFB6C1;
            }
            /* Add more custom styles as needed */
        </style>
        """
    else:
        return """
        <style>
            /* Dark theme styles */
            @import url('https://fonts.googleapis.com/css2?family=Comic+Neue:wght@400;700&display=swap');
            body, .main, .stSidebar, [data-testid="stSidebar"] {
                font-family: 'Comic Neue', cursive !important;
                font-size: 18px !important;
                background-color: #1E1E1E;
                color: #FFFFFF;
            }
            .stButton > button {
                background-color: #FF69B4;
                color: white;
                border-radius: 20px;
                font-size: 20px;
                font-family: 'Comic Neue', cursive !important;
                border: 2px solid #FF1493;
            }
            .stTextInput > div > div > input, .stTextArea > div > div > textarea {
                background-color: #2E2E2E;
                color: #FFFFFF !important;
                border-radius: 10px;
                border: 2px solid #FF69B4;
                font-family: 'Comic Neue', cursive !important;
                font-size: 18px !important;
            }
            h1, h2, h3 {
                color: #FF69B4;
                font-family: 'Comic Neue', cursive !important;
                font-weight: 700;
            }
            .stSidebar {
                background-color: #2E2E2E;
            }
            /* Add more custom styles as needed */
        </style>
        """

st.markdown(get_custom_css(), unsafe_allow_html=True)

# Set up OpenAI API key
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception as e:
    st.error(f"Error accessing OpenAI API key: {str(e)}")
    st.stop()

# Initialize the OpenAI model
try:
    llm = OpenAI(temperature=0.7, api_key=api_key, streaming=True)
except Exception as e:
    st.error(f"Error initializing OpenAI model: {str(e)}")
    st.stop()

# Define prompt template
prompt_template = PromptTemplate(
    input_variables=["patient_info", "question"],
    template="""
    You are a friendly pediatrician with access to the following patient information:
    {patient_info}
    
    Based on this information, please answer the following question about pediatric care: {question}
    
    Provide a short, concise, informative, and child-friendly answer, considering the patient's age, medical history, and relevant pediatric guidelines. Include home remedies or local tips when appropriate. For questions about sex from those under 18, clearly state it's not appropriate for their age. For those 18 and over, provide cautious guidance if necessary. Only discuss sex if directly asked.
    """
)

# Create chain
chain = (
    {"patient_info": RunnablePassthrough(), "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

# Helper functions
def get_supported_languages():
    try:
        translator = GoogleTranslator()
        supported_languages = translator.get_supported_languages(as_dict=True)
        supported_languages['English'] = 'en'
        return supported_languages
    except Exception as e:
        st.error(f"Could not fetch supported languages. Error: {str(e)}")
        return {'English': 'en'}

def get_pediatric_response(patient_info, question, target_lang):
    try:
        detected_lang = detect(question)
        if detected_lang != 'en':
            translator = GoogleTranslator(source=detected_lang, target='en')
            translated_question = translator.translate(question)
        else:
            translated_question = question
        
        response = chain.invoke({"patient_info": patient_info, "question": translated_question})
        
        if target_lang != 'en':
            translator = GoogleTranslator(source='en', target=target_lang)
            translated_response = translator.translate(response.strip())
        else:
            translated_response = response.strip()
        
        return translated_response
    except Exception as e:
        return f"Oops! Something went wrong: {str(e)}"

def text_to_speech(text, lang='en'):
    if not st.session_state.voice_enabled:
        return
    
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save("temp.mp3")
        
        with open("temp.mp3", "rb") as audio_file:
            audio_bytes = audio_file.read()
        
        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_html = f'<audio autoplay="true" src="data:audio/mp3;base64,{audio_base64}">'
        st.markdown(audio_html, unsafe_allow_html=True)
        
        os.remove("temp.mp3")
    except Exception as e:
        st.error(f"Text-to-Speech failed: {str(e)}")
        st.warning("Text-to-Speech is currently unavailable. Please read the response instead.")

def get_pediatric_facilities():
    return [
        {
            "name": "Pitter Patter Pediatric Care",
            "phone": "+1 (869) 766-7876",
            "location": "Basseterre, St. Kitts",
            "address": "Central Street, Basseterre, St. Kitts"
        },
        {
            "name": "Dr. Patrick Martin's Clinic",
            "phone": "+1 (869) 662-2600",
            "location": "Basseterre, St. Kitts",
            "address": "Cayon Street, Basseterre, St. Kitts"
        },
        {
            "name": "Smithen's Medical Clinic",
            "phone": "+1 (869) 668-5881",
            "location": "Basseterre, St. Kitts",
            "address": "Liverpool Row, Basseterre, St. Kitts"
        },
        {
            "name": "Joseph N. France General Hospital - Pediatric Ward",
            "phone": "+1 (869) 465-2551",
            "location": "Buckleys Site, Basseterre, St. Kitts",
            "address": "Buckleys Site, Basseterre, St. Kitts"
        }
    ]

# Get supported languages
supported_languages = get_supported_languages()

# Streamlit UI
st.header("🤍 KidzCareHub 🩺")
st.title("👩🏽KidzCareHub Chat👨🏽")
st.write("Hi, I'm Rhea, your virtual health assistant! 🌟")
st.write("I'm here to help with your health questions and provide guidance. While I'm here to assist, please consult a doctor for any medical decisions. 👨‍⚕️👩‍⚕️")

col1, col2 = st.columns(2)

with col1:
    st.header("📋 Your Information")
    patient_name = st.text_input("Your Name", placeholder="Enter child's name")
    patient_age = st.number_input("How old are you (years)", min_value=0, max_value=18, step=1)
    medical_history = st.text_area("Your Medical History", placeholder="Eg: Asthma", height=100)

with col2:
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    medications = st.text_area("Current Medications", placeholder="Eg: Albuterol, Ibuprofen", height=100)

patient_info = f"""
Your Name: {patient_name}
Your Age: {patient_age} years
Medical History: {medical_history}
Medications: {medications}
"""

st.header("❓ Ask Rhea")
question = st.text_input("Ask your pediatric care question:")

with st.sidebar:
    st.header("⚙️ Settings")
    st.button("Light/Dark Mode 🌓", on_click=toggle_theme)
    
    language_options = list(supported_languages.keys())
    default_index = language_options.index('English')
    selected_language = st.selectbox(
        "Select Language 🌐",
        options=language_options,
        index=default_index
    )
    selected_lang_code = supported_languages.get(selected_language, 'en')
    
    st.header("🔊 Voice Settings")
    voice_enabled = st.toggle("Enable Voice", value=st.session_state.voice_enabled)
    st.session_state.voice_enabled = voice_enabled
    
    st.header("📊 Patient Summary")
    st.text(patient_info)

    st.header("📞 Pediatric Facilities")
    facilities = get_pediatric_facilities()
    for facility in facilities:
        st.markdown(f"""
        **{facility['name']}**
        - Phone: <a href="tel:{facility['phone']}">{facility['phone']}</a>
        - Location: {facility['location']}
        """, unsafe_allow_html=True)
        
        maps_query = f"{facility['name']}, {facility['address']}".replace(' ', '+')
        maps_url = f"https://www.google.com/maps/search/?api=1&query={maps_query}"
        
        st.markdown(f"[See on the map]({maps_url})")
        
        st.markdown("---")

    st.markdown("*SKN Health Operator:*")
    if st.button("Connect to SKN Health Operator"):
        st.markdown('<a href="https://sknhealth.my3cx.us/jnfoperator" target="_blank">Click here to connect to SKN Health Operator</a>', unsafe_allow_html=True)

if st.button("Get Answer 🚀"):
    if question:
        with st.spinner("Let's see... 🤔"):
            try:
                answer = get_pediatric_response(patient_info, question, selected_lang_code)
                st.subheader("👩‍⚕️ Rhea's response:")
                st.write(answer)
                if st.session_state.voice_enabled:
                    text_to_speech(answer, lang=selected_lang_code)
            except Exception as e:
                st.error(f"An error occurred while generating the response: {str(e)}")
    else:
        st.warning("Please ask a question first! 😊")

st.header("📚 Health Tips and Reminders")
health_tips = """
**Health Tips for Kids:**
- Ensure regular pediatric check-ups and adhere to vaccination schedules.
- Encourage a balanced diet with plenty of vegetables, fruits, and proteins.
- Promote at least 1 hour of physical activity each day.
- Establish a consistent bedtime routine to ensure adequate sleep.
- Teach proper hand washing techniques to prevent infections.

**Vaccination Reminders:**
- Keep track of immunizations such as MMR, DTP, and annual flu shots.
- Follow up on booster doses as recommended by your pediatrician.
"""
st.markdown(health_tips)

# Add a footer
st.markdown("---")
st.markdown("Made with 💖 for kids and parents everywhere!")
