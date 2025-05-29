import streamlit as st
import speech_recognition as sr
import asyncio
import edge_tts
import os
import sys

# Add the parent directory to the Python path to allow importing from 'core'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.agent import get_financial_agent_response

async def text_to_speech(text, voice="en-US-AriaNeural", outfile="output.mp3"):
    """Converts text to speech using edge-tts and saves it to an mp3 file."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(outfile)

def speech_to_text():
    """Captures audio from the microphone and converts it to text."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = r.listen(source)
        st.info("Processing...")
    try:
        text = r.recognize_whisper(audio_data=audio, model="tiny.en") # Using whisper via speech_recognition
        st.success("Recognized: " + text)
        return text
    except sr.UnknownValueError:
        st.error("Whisper could not understand audio")
        return None
    except sr.RequestError as e:
        st.error(f"Could not request results from Whisper service; {e}")
        return None

st.title("🎙️💰 Finance Assistant")

st.write("Ask about your risk exposure, earnings surprises, or other financial queries.")

input_method = st.radio("Input method:", ('Text', 'Voice'))

user_query = None
if input_method == 'Text':
    user_query = st.text_input("Your question:")
elif input_method == 'Voice':
    if st.button("Speak"): 
        user_query = speech_to_text()
        if user_query:
            st.write(f"You asked: {user_query}")

if user_query:
    with st.spinner("Thinking..."):
        response_text = get_financial_agent_response(user_query)
        st.subheader("Response:")
        st.write(response_text)

        # Text-to-speech output
        audio_file_path = "output.mp3"
        try:
            # Create a new event loop for asyncio if one is not available
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            loop.run_until_complete(text_to_speech(response_text, outfile=audio_file_path))
            
            if os.path.exists(audio_file_path):
                with open(audio_file_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3", autoplay=True)
            else:
                st.error("TTS audio file was not created.")

        except Exception as e:
            st.error(f"Error in text-to-speech: {e}")
        finally:
            if os.path.exists(audio_file_path):
                try:
                    os.remove(audio_file_path) # Clean up the audio file
                except Exception as e:
                    st.warning(f"Could not remove temporary audio file: {e}") 