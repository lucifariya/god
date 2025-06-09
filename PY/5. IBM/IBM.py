#!/usr/bin/env python
# coding: utf-8

# ## Questions:
# 1. Define Cognitive Computing.
# 2. Services of IBM Watson.
# 3. Services of T. C. T. App.
# 4. Mashup means?
# 5. Speech Synthesis Means?
# 6. Functions of IBM Watson.
# 7. Applications of IBM Watson.

# ## 1:
# ### Cognitive Computing
# - Simulates human brain functions like pattern recognition and decision-making.
# - Learns as it processes more data.
# - Used in diverse real-world applications (e.g., healthcare, customer service, fraud detection).  

# ## 2:
# ### Watson Services (Lite Tier)  
# - Free tier available for experimentation ("no credit card required").
# - Services include:
#     * **Speech to Text –** Converts spoken audio to text.
#     * **Text to Speech –** Synthesizes speech from text (supports multiple languages).
#     * **Language Translator –** Translates text between languages and identifies languages.
#     * **Natural Language Understanding –** Analyzes sentiment, emotion, and keywords.
#     * **Visual Recognition –** Identifies objects, faces, and text in images/videos.
#     * **Watson Assistant –** Builds chatbots and virtual assistants.
#     * **Personality Insights –** Analyzes personality traits from text.
#     * **Tone Analyzer –** Detects emotional and social tones in text.

# ## 3:
# ### Traveler’s Companion Translation App (Case Study)
# - **Mashup** of three Watson services:
#     1. **Speech to Text** – Transcribes spoken English/Spanish.
#     2. **Language Translator –** Translates between English mand Spanish.
#     3. **Text to Speech –** Converts translated text back to speech.
# - Steps:
#     * Record a question (English) → Transcribe → Translate → Speak (Spanish).
#     * Record a response (Spanish) → Transcribe → Translate → Speak (English).
#     * Requires API keys for each service (stored in keys.py). 

# ## 4-5:
# ### Key Terms
# - **Mashup –** Combining multiple services into one application.
# - **SSML (Speech Synthesis Markup Language) –** Controls speech synthesis (e.g., pitch, pauses).
# - **Cognitive Computing –** AI systems that mimic human reasoning.

# ## 6:
# ### Key Functions
# #### speech_to_text(file_name, model_id)
# - **Process:**
#     1. Opens audio file ('rb' mode for binary read).
#     2. Calls Watson’s recognize method with:
#         * audio: File object.
#         * content_type: 'audio/wav'.
#         * model: Language model (e.g., en-US_BroadbandModel).
#     3. Parses JSON response to extract transcript. 
#     
# #### translate(text_to_translate, model)
# - **Process:**
#     1. Calls Watson’s translate method with:
#         * text: String to translate.
#         * model_id: Translation model (e.g., en-es).
#     2. Returns translated text from JSON (translations[0}['translation']).
#     
# #### text_to_speech(text_to_speak, voice_to_use, file_name)
# - **Process:**
#     1. Opens output file ('wb' mode for binary write).
#     2. Calls Watson’s synthesize method with:
#         * text: String to speak.
#         * accept: 'audio/wav'.
#         * voice: Voice model (e.g., es-US_SofiaVoice).
#     3. Writes audio bytes to file.
# 
# #### record_audio(file_name)
# - **Uses PyAudio:**
#     * Configures audio stream (44.1 kHz, 16-bit, stereo).
#     * Records 5 seconds of audio in chunks (1024 frames per chunk).
#     * Saves as WAV file using wave module.
# 
# #### play_audio(file_name)
# - **Uses PyDub:**
#     * Loads WAV file with AudioSegment.from_wav().
#     * Plays audio with pydub.playback.play().

# ## 7:
# ### Applications of IBM Cognitive Computing
# * Healthcare: Diagnosing diseases, personalized treatment, drug discovery.
# * Finance: Fraud detection, risk analysis, automated customer support.
# * Retail: AI-driven recommendations, personalized shopping experiences.
# * Manufacturing: Predictive maintenance, process optimization.
# * Cyber security: AI-powered threat detection and prevention.

# In[ ]:


# 
# 12: barchart wager aibani
# 13.8, 13.9, 13.10, 13.11
# 
# syllabus upto pg book page 624 (15.5)

