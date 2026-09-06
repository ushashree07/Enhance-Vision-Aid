# Enhance Vision Aid

> An AI-powered assistive vision system designed to help visually impaired users understand their surroundings through computer vision, object detection, spatial awareness, voice interaction, and AI-generated responses.

## Overview

**Enhance Vision Aid** is a Python-based assistive technology project that combines computer vision, object detection, speech recognition, text-to-speech, and generative AI to provide auditory information about the user's environment.

The core vision pipeline uses **YOLOv10** for real-time object detection. Detected objects are classified according to their approximate position in the camera frame — **left, center, or right** — and converted into spoken environmental descriptions.

The project also contains an experimental AI-agent layer that provides:

* Conversational assistance
* Voice interaction
* Web-based information retrieval
* AI-powered image understanding
* Video/image analysis experiments
* Task persistence using SQLite

The repository is currently a research/prototype implementation with multiple experimental modules rather than a single unified production application.

---

## Key Features

### Computer Vision

* Real-time webcam capture
* YOLOv10-based object detection
* Bounding-box visualization
* Object confidence filtering
* Approximate spatial classification:

  * Left
  * Center
  * Right
* Environment description generation
* Audio feedback for detected objects

### Voice Assistance

* Speech-to-text using `SpeechRecognition`
* Microphone input
* Text-to-speech using `pyttsx3`
* Basic voice commands
* Conversational fallback through an LLM

### Generative AI

The project experiments with multiple AI providers and frameworks:

* Groq
* LangChain
* Phidata
* Google Gemini
* Landing AI
* DuckDuckGo search

### Data Persistence

An SQLite-based task database is included for storing:

* Task name
* Task status
* Creation timestamp

---

# Architecture

The current repository can be viewed as four major layers:

```text
                    ┌──────────────────────────┐
                    │       User / Camera      │
                    │    Microphone / Image    │
                    └────────────┬─────────────┘
                                 │
                ┌────────────────┴────────────────┐
                │                                 │
                ▼                                 ▼
       ┌─────────────────┐              ┌──────────────────┐
       │ Computer Vision │              │  Voice Interface │
       │     Layer       │              │      Layer       │
       └────────┬────────┘              └────────┬─────────┘
                │                                 │
                ▼                                 ▼
       ┌─────────────────┐              ┌──────────────────┐
       │    YOLOv10      │              │ SpeechRecognition│
       │ Object Detection│              │      + TTS       │
       └────────┬────────┘              └────────┬─────────┘
                │                                 │
                ▼                                 ▼
       ┌─────────────────┐              ┌──────────────────┐
       │ Spatial Analysis│              │ Conversational AI│
       │ Left / Center / │              │ Groq / LangChain │
       │      Right      │              └────────┬─────────┘
       └────────┬────────┘                       │
                │                                │
                └───────────────┬────────────────┘
                                ▼
                       ┌──────────────────┐
                       │ Audio / Assistant│
                       │     Response     │
                       └──────────────────┘
```

---

# Repository Structure

```text
Enhance_Vision_Aid/
│
├── Ai_agents/
│   ├── agent.py
│   ├── chatbot.py
│   ├── database.py
│   ├── requirement.txt
│   ├── system_prompt.txt
│   ├── test.py
│   ├── test_db.py
│   ├── vediosummarie.py
│   ├── vision_agent.py
│   ├── voiceAss.py
│   └── .env
│
├── capture.py
├── detect.py
├── object_clasify.py
├── object_detection.py
├── test.py
├── Requirements.txt
│
└── weights/
    └── YOLOv10 model weights
```

The repository currently has three primary root-level vision implementations (`detect.py`, `object_clasify.py`, and `object_detection.py`) plus supporting capture/testing modules.

---

# Module Documentation

## 1. `capture.py`

### Purpose

Handles webcam capture and saving an image frame for subsequent processing.

### Main responsibilities

* Opens the default webcam using OpenCV.
* Captures a frame.
* Allows the camera to stabilize before saving.
* Saves the captured image.
* Loads and displays the captured image.

The current implementation saves the image as:

```text
captured_frame.jpg
```

and uses OpenCV's `VideoCapture(0)` interface.

### Main functions

```python
capture_and_save_frame()
context_aware(image_path)
```

### Flow

```text
Webcam
   ↓
Capture frame
   ↓
Save image
   ↓
Load image
   ↓
Display / process
```

---

# 2. `detect.py`

### Purpose

Provides a basic real-time YOLOv10 object-detection pipeline.

### Implementation

The module loads YOLOv10 models and processes webcam frames continuously.

It currently references:

```text
yolov10s.pt
yolov10b.pt
```

and uses the larger model for webcam inference.

### Processing pipeline

```text
Webcam
   ↓
OpenCV frame
   ↓
YOLOv10
   ↓
Detections
   ↓
Supervision
   ↓
Bounding boxes + labels
   ↓
Display
```

### Libraries

* OpenCV
* Ultralytics YOLO
* Supervision

### Output

The detected objects are displayed visually using bounding boxes and labels.

This module is primarily a **visual detection prototype** and does not currently provide the full voice-feedback pipeline.

---

# 3. `object_clasify.py`

### Purpose

Adds spatial reasoning and audio feedback on top of YOLOv10 detection.

Despite the filename `object_clasify.py`, the module is primarily performing **object position classification**, not semantic object classification.

### YOLO model

The module loads:

```text
weights/yolov10m.pt
```

### Spatial classification

The camera frame is divided into three regions:

```text
┌────────────┬────────────┬────────────┐
│            │            │            │
│    LEFT    │   CENTER   │    RIGHT   │
│            │            │            │
└────────────┴────────────┴────────────┘
```

Objects are assigned to one of:

```python
{
    "left": [],
    "center": [],
    "right": []
}
```

The implementation uses confidence filtering and bounding-box coordinates to determine the object's approximate position.

### Audio feedback

Detected objects are converted into natural-language descriptions such as:

```text
1 person on the left.
2 cars on the center.
1 chair on the right.
```

The description is converted to speech using Google TTS and played using `sounddevice`.

### Overall flow

```text
Camera
  ↓
YOLOv10m
  ↓
Object detections
  ↓
Confidence filtering
  ↓
Bounding-box position
  ↓
Left / Center / Right
  ↓
Environment description
  ↓
Google TTS
  ↓
Audio output
```

---

# 4. `object_detection.py`

### Purpose

Provides a more complete voice-controlled real-time object-detection prototype.

This module combines:

* YOLOv10
* Speech recognition
* Spatial object classification
* Text-to-speech
* Audio playback

It loads:

```text
weights/yolov10s.pt
```

and uses `SpeechRecognition` for voice commands.

### Voice commands

The detection system is designed around commands such as:

```text
start detection
stop detection
exit
```

### Spatial analysis

Detected objects are divided into:

```text
Left       Center       Right
 33%         33%         34%
```

with approximately:

```python
left_boundary = frame_width * 0.33
right_boundary = frame_width * 0.66
```

Objects below the configured confidence threshold are filtered out.

### Audio generation

The system creates a textual environment description and converts it to speech:

```text
YOLO detections
       ↓
Object positions
       ↓
Object counts
       ↓
Environment description
       ↓
gTTS
       ↓
WAV file
       ↓
sounddevice
       ↓
User
```

### Significance

This is one of the closest modules in the repository to the intended assistive-vision workflow because it combines **vision + spatial reasoning + voice control + audio output**.

---

# 5. `test.py`

### Purpose

Experimental image-based object detection pipeline.

It loads an image, resizes it to:

```text
640 × 640
```

and performs YOLOv10 detection.

### Processing

```text
Input image
    ↓
Resize
    ↓
YOLOv10
    ↓
Confidence filtering
    ↓
Position classification
    ↓
Environment description
```

The module uses a confidence threshold of approximately `0.5` and classifies objects into left, center, and right regions.

### Intended use

Useful for testing the detection logic without requiring continuous webcam input.

---

# 6. `Ai_agents/agent.py`

### Purpose

Experimental research/web-information agent.

The module uses **Phidata**, Groq and external tools.

### Components

```text
Phidata Agent
      │
      ├── Groq / DeepSeek model
      │
      ├── DuckDuckGo
      │
      └── Newspaper4k
```

The configured model is:

```text
deepseek-r1-distill-llama-70b
```

The agent is instructed to search for sources, read articles, and produce a short response. Its result is passed to the voice `speak()` function.

### Intended capability

This module represents the project's attempt to add a general-purpose research assistant alongside the vision system.

---

# 7. `Ai_agents/chatbot.py`

### Purpose

Provides conversational AI functionality.

There are currently two approaches in this file.

### Direct Groq API

The `chat_respond()` function uses:

```text
Groq
└── llama3-8b-8192
```

and provides a short assistant response.

### LangChain conversational processor

The file also contains:

```python
LanguageModelProcessor
```

which uses:

```text
ChatGroq
LangChain
ConversationBufferMemory
ChatPromptTemplate
```

The conversation maintains chat history through `ConversationBufferMemory`.

### Flow

```text
User speech/text
       ↓
chat_respond()
       ↓
Groq LLM
       ↓
Response
       ↓
Voice output
```

or:

```text
User input
    ↓
LangChain prompt
    ↓
Conversation memory
    ↓
ChatGroq
    ↓
Response
```

---

# 8. `Ai_agents/voiceAss.py`

### Purpose

Provides the primary voice-assistant interface.

### Speech-to-text

Uses:

```text
SpeechRecognition
```

and Google's speech-recognition service.

The microphone configuration includes:

* Energy threshold
* Pause threshold
* Timeout
* Phrase time limit

### Text-to-speech

Uses:

```text
pyttsx3
```

with a configured speech rate.

### Command processing

The assistant supports basic commands including:

```text
time
date
open google
exit
bye
```

For other commands, the input is passed to the chatbot module.

### Architecture

```text
Microphone
    ↓
SpeechRecognition
    ↓
Command classification
    │
    ├── Time
    ├── Date
    ├── Browser
    ├── Exit
    │
    └── General question
             ↓
         Groq chatbot
             ↓
          Response
             ↓
          pyttsx3
             ↓
          Speaker
```

---

# 9. `Ai_agents/database.py`

### Purpose

Provides simple persistent task storage using SQLite.

The database file is:

```text
tasks.db
```

### Schema

```sql
CREATE TABLE tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_name TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Main operations

```python
create_db()
add_task(task_name)
get_tasks()
update_task(task_id, status)
```

### Use case

The database can support future assistant functionality such as:

```text
"Remember this task"
"Show my tasks"
"Mark this task completed"
```

Currently, this is a standalone persistence component rather than an integrated task-management subsystem.

---

# 10. `Ai_agents/system_prompt.txt`

### Purpose

Defines the conversational behavior of the AI assistant.

The assistant is named:

```text
Lisa
```

and is instructed to:

* Assist blind users
* Answer questions
* Use short conversational responses
* Avoid generating code
* Keep responses below 20 words

This file acts as the behavioral configuration for the conversational AI.

---

# 11. `Ai_agents/vision_agent.py`

### Purpose

Experimental integration with **Landing AI's agentic object detection API**.

The module sends an image to the Landing AI endpoint with a prompt requesting detection of objects such as:

```text
roads
obstacles
vehicles
people
traffic signs
```

### Flow

```text
Image
  ↓
Landing AI API
  ↓
Agentic object detection
  ↓
JSON response
```

This module represents an alternative to the local YOLO-based detection pipeline.

---

# 12. `Ai_agents/vediosummarie.py`

### Purpose

Experimental image/video understanding using Google's Gemini models.

The module uploads a media file and sends it to Gemini for generative analysis. The current implementation uses the Gemini API and `gemini-2.0-flash-exp`.

### Intended flow

```text
Media
  ↓
Google Gemini
  ↓
Multimodal analysis
  ↓
Natural-language description
```

Despite the filename `vediosummarie.py`, the current code example primarily demonstrates multimodal image analysis.

---

# 13. `Ai_agents/test.py`

### Purpose

Testing/experimental module for the AI-agent components.

It is separate from the production-style vision modules and appears to be used for validating AI functionality.

---

# 14. `Ai_agents/test_db.py`

### Purpose

Testing module for the SQLite task database.

It is intended to validate the functions provided by:

```text
database.py
```

including database creation and task operations.

---

# 15. `Ai_agents/requirement.txt`

Contains dependencies associated with the AI-agent experiments.

The repository therefore has **two dependency files**:

```text
Requirements.txt
Ai_agents/requirement.txt
```

For a clean deployment, these should eventually be consolidated or separated into clearly defined environment profiles.

---

# 16. `Requirements.txt`

The root dependency file includes the main computer-vision, audio, and AI dependencies.

Current dependencies include:

```text
ultralytics
supervision
opencv-python==4.7.0.72
matplotlib
seaborn
numpy==1.26.4
gTTS==2.5.3
ipython==9.0.0
torchvision==2.6.0+cpu
torchaudio
speechrecognition
sounddevice
soundfile
```

---

# End-to-End Concept

The intended user experience can be represented as:

```text
                  USER
                   │
          ┌────────┴────────┐
          │                 │
       Camera            Voice
          │                 │
          ▼                 ▼
       OpenCV        Speech Recognition
          │                 │
          ▼                 ▼
       YOLOv10          Command / Query
          │                 │
          ▼                 ▼
   Object Detection    AI Chatbot
          │                 │
          ▼                 ▼
 Left / Center / Right     LLM
          │                 │
          └────────┬────────┘
                   ▼
             Text Response
                   │
                   ▼
             Text-to-Speech
                   │
                   ▼
                 USER
```

---

# Technology Stack

| Area                    | Technology              |
| ----------------------- | ----------------------- |
| Programming Language    | Python                  |
| Computer Vision         | OpenCV                  |
| Object Detection        | YOLOv10 / Ultralytics   |
| Detection Visualization | Supervision             |
| Speech Recognition      | SpeechRecognition       |
| Text-to-Speech          | pyttsx3 / gTTS          |
| Audio                   | sounddevice / soundfile |
| Conversational AI       | Groq                    |
| LLM Framework           | LangChain               |
| Agent Framework         | Phidata                 |
| Multimodal AI           | Google Gemini           |
| External Vision API     | Landing AI              |
| Search                  | DuckDuckGo              |
| Local Database          | SQLite                  |
| Image Processing        | OpenCV / PIL            |
| Visualization           | Matplotlib              |

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/darshan12345678910/Enhance_Vision_Aid.git

cd Enhance_Vision_Aid
```

## 2. Create a virtual environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r Requirements.txt
```

For AI-agent functionality, install the dependencies listed under:

```text
Ai_agents/requirement.txt
```

---

# Model Weights

The vision modules expect YOLOv10 model weights.

Examples currently referenced by the source:

```text
weights/yolov10s.pt
weights/yolov10m.pt
weights/yolov10b.pt
```

Place the required model files under:

```text
weights/
```

The exact model required depends on which module is being executed.

---

# Running the Vision System

## Basic YOLO detection

```bash
python detect.py
```

This opens the webcam and displays YOLO detections.

Press:

```text
q
```

to exit.

---

## Real-time object detection with audio

```bash
python object_detection.py
```

The module listens for voice commands and performs real-time detection.

Example:

```text
start detection
```

The detected objects are converted into positional descriptions and spoken to the user.

---

## Object position classification

```bash
python object_clasify.py
```

This runs YOLOv10 detection and classifies detected objects into:

```text
LEFT
CENTER
RIGHT
```

The resulting description is converted to speech.

---

# Running the Voice Assistant

The voice assistant can be started using:

```bash
python Ai_agents/voiceAss.py
```

The assistant starts by listening for commands.

Example interactions:

```text
"What is the time?"
"What is today's date?"
"Open Google"
"Tell me about artificial intelligence"
```

General questions are forwarded to the chatbot/LLM layer.

---

# Environment Variables

AI-agent modules require API credentials.

Use an environment file rather than hardcoding credentials.

Example:

```env
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
```

Then load them using:

```python
from dotenv import load_dotenv

load_dotenv()
```

---

# Important Security Notice

API credentials should **never be committed to GitHub**.

The current repository contains API credentials directly inside some experimental source code, including the Gemini and Landing AI integrations. These credentials should be considered compromised if they are real.

Recommended remediation:

1. Revoke exposed API keys.
2. Generate new keys.
3. Remove secrets from source files.
4. Store secrets in environment variables.
5. Add `.env` to `.gitignore`.
6. Remove previously committed secrets from Git history if necessary.

For example:

```python
api_key = os.getenv("LANDING_AI_API_KEY")
```

instead of:

```python
api_key = "..."
```

---

# Current Project Status

The repository is best characterized as a **research/prototype-stage assistive vision system**.

### Implemented / demonstrated

* YOLOv10 object detection
* Webcam-based detection
* Object spatial classification
* Left/center/right environmental descriptions
* Speech recognition
* Text-to-speech
* Conversational AI
* Groq integration
* Gemini experimentation
* Landing AI experimentation
* SQLite task persistence
* AI-agent experimentation

### Still requiring integration

The repository currently contains several independent experimental pipelines rather than one production-grade orchestrator.

For example:

```text
YOLO pipeline
       │
       ├── object_detection.py
       ├── object_clasify.py
       └── detect.py

AI pipeline
       │
       ├── chatbot.py
       ├── agent.py
       ├── vision_agent.py
       └── vediosummarie.py
```

These components are not yet exposed through one unified application architecture.

---

# Recommended Production Architecture

For future development, the system can be consolidated into:

```text
                    ┌──────────────────┐
                    │   Input Manager  │
                    │ Camera + Voice   │
                    └────────┬─────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
       ┌────────────────┐        ┌────────────────┐
       │ Vision Engine  │        │ Voice Engine   │
       │                │        │                │
       │ YOLOv10        │        │ STT            │
       │ Object Detect  │        │ Command Router │
       └───────┬────────┘        └───────┬────────┘
               │                         │
               ▼                         ▼
       ┌────────────────────────────────────────┐
       │          Context / Decision Engine     │
       │                                        │
       │ Object position                        │
       │ Confidence                             │
       │ User command                           │
       │ Scene context                          │
       └──────────────────┬─────────────────────┘
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
      ┌──────────────┐         ┌───────────────┐
      │ Local Logic  │         │ Cloud AI      │
      │ Fast / Safe  │         │ LLM / Vision  │
      └──────┬───────┘         └───────┬───────┘
             │                         │
             └────────────┬────────────┘
                          ▼
                  ┌───────────────┐
                  │ Response      │
                  │ Prioritizer   │
                  └───────┬───────┘
                          ▼
                  ┌───────────────┐
                  │ TTS / Audio   │
                  └───────────────┘
```

This would make it easier to add future capabilities such as:

* OCR
* Traffic-sign detection
* Obstacle prioritization
* Distance estimation
* Face recognition
* Navigation assistance
* Emergency alerts
* Scene description
* Personal-object recognition
* Offline inference

---

# Limitations

The current prototype has several limitations:

* Hardcoded Windows file paths exist in some modules.
* Different modules use different YOLOv10 model sizes.
* Multiple independent implementations perform similar detection tasks.
* Some AI integrations are experimental.
* Cloud services are required for some speech/AI functionality.
* API credentials need to be moved completely into environment configuration.
* No unified application entry point exists.
* No centralized configuration system exists.
* No formal automated test suite covers the complete system.
* Detection and speech processing are largely synchronous.
* Audio feedback can potentially become repetitive when objects remain continuously detected.
* Spatial classification is based primarily on 2D bounding-box position rather than actual depth/distance.

---

# Future Improvements

## Computer Vision

* Add object tracking to prevent repeated announcements.
* Add depth estimation.
* Estimate distance to obstacles.
* Prioritize dangerous objects.
* Add OCR.
* Add traffic-sign recognition.
* Add face recognition.
* Add scene understanding.

## Audio

* Add announcement prioritization.
* Add spatial audio.
* Add configurable speech rate.
* Avoid repeating unchanged detections.
* Add multilingual TTS.

## AI

* Create a unified vision-language agent.
* Use local models for offline operation.
* Implement context-aware prompting.
* Combine object detections with LLM reasoning.
* Maintain short-term scene memory.

## Architecture

* Create a single application entry point.
* Separate inference, input, output and orchestration layers.
* Add configuration management.
* Add structured logging.
* Add proper unit/integration tests.
* Add model management.
* Remove hardcoded paths.
* Containerize backend services where appropriate.

---

# Project Goals

The long-term objective of Enhance Vision Aid is to provide an accessible AI assistant that converts visual information into concise auditory guidance.

The central concept is:

```text
SEE → UNDERSTAND → PRIORITIZE → SPEAK
```

The camera provides visual information, computer vision detects relevant objects, AI components provide additional interpretation, and the voice layer communicates the result to the user.

---

# Disclaimer

This project is an experimental assistive-technology prototype.

Computer-vision predictions and AI-generated descriptions can be incorrect. The system should not currently be relied upon as the sole source of information for safety-critical navigation or obstacle avoidance.
