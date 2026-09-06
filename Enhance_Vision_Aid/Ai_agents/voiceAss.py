"""
Voice Assistant Module
======================
Speech recognition and text-to-speech voice assistant.
Processes voice commands and provides spoken responses.
"""

import speech_recognition as sr
import pyttsx3
import datetime
import webbrowser
from chatbot import chat_respond

def speak(text):
    """Convert text to speech and play audio."""
    engine = pyttsx3.init()
    engine.setProperty('rate', 175)  # Increase speech speed
    engine.say(text)
    engine.runAndWait()

def listen():
    """
    Listen to microphone input and return recognized speech.
    
    Returns:
        str: Recognized command or empty string on error
    """
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300  # Adjust sensitivity
    recognizer.pause_threshold = 0.6  # Reduce pause time

    with sr.Microphone() as source:
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=5)
            command = recognizer.recognize_google(audio).lower()
            print("You said:", command)
            return command
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand.")
            return ""
        except sr.RequestError:
            print("Network error. Check your internet connection.")
            return ""
        except sr.WaitTimeoutError:
            print("No speech detected, please try again.")
            return ""

def respond(command):
    """
    Process user command and generate appropriate response.
    
    Handles: time, date, web searches, chatbot responses.
    """
    if "time" in command:
        speak(f"The time is {datetime.datetime.now().strftime('%I:%M %p')}")
    elif "date" in command:
        speak(f"Today's date is {datetime.datetime.now().strftime('%B %d, %Y')}")
    elif "open google" in command:
        webbrowser.open("https://www.google.com")
        speak("Opening Google")
    elif "exit" in command or "bye" in command:
        speak("Goodbye!")
        exit()
    else:
        response = chat_respond(command)
        if response:
            speak(response)
        else:
            speak("I'm sorry, I didn't get that.")

        
def main():
    """Start voice assistant main loop."""
    speak("Hello, I am your voice assistant. How can I help you?")
    while True:
        command = listen()
        if command:
            respond(command)

if __name__ == "__main__":
    main()
