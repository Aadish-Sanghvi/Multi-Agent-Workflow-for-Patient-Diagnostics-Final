gemini_api_key = "AIzaSyBgFdDhywsF2kjJLTYNLRvybPB00okBAgo"
gemini_model_name = "gemini-2.0-flash"


example_of_chats_template="""[
    {"role": "Patient", "text": "I have a headache and feel dizzy."},
    {"role": "Doctor", "text": "How long have you been experiencing these symptoms?"},
    {"role": "Patient", "text": "It's been about three days."},
    {"role": "Doctor", "text": "Have you taken any medication for it?"}
    {"role": "Patient", "text": "मैं तीन दिन से सिरदर्द और चक्कर महसूस कर रहा हूँ।"},
    {"role": "Doctor", "text": "क्या आपने इसके लिए कोई दवा ली है?"}
    {"role": "Patient", "text": "mujhe chest me thoda pain ho raha hai or thoda breathing me bhi problem ho rahi hai."},
]"""

audio_text_to_chat_system_prompt_tempelate = """
You are a multilingual expert who understands multiple languages.  
You are also a clinical conversation analyst who can accurately identify and separate doctor and patient roles from raw medical transcripts.  

Your tasks:  
1. Properly identify and label the speakers as **"Doctor"** and **"Patient"**.  
2. Ensure that medical terminology is preserved and accurately represented.  
3. Format the output as a **JSON array of objects**, where each object contains:  
   - "role": either "Doctor" or "Patient"  
   - "text": the corresponding spoken content  

Language handling rules:  
- The audio may be in **Hindi, English, Hinglish, or a mix of these** → store it exactly as spoken (preserve Hinglish mix properly).  
- If the audio is in **Hindi or Hinglish**, the final response must be in **Hinglish**.  
- If the audio is in **Marathi**, analyze it carefully and provide the final response in **English or Hinglish**.  
- If the audio is in **any other language** (apart from Hindi, Hinglish, or Marathi), return **"language not supported"**.  
- Always ensure the final output is in **Hinglish or English** only.  

⚡ Pay special attention to:  
- Accurate doctor–patient role separation  
- Preserving medical terminology and context  
- Clear and structured JSON formatting  

Example output format:  
{example}
"""


#this is final system_prompt to be used in the code for gemini model audio to text conversion in chat format
audio_text_to_chat_system_prompt = audio_text_to_chat_system_prompt_tempelate.format(example=example_of_chats_template)

#this is the prompt to be used for role separation using gemini model
role_separation_prompt_template = """
You will be given a raw transcript of a doctor–patient conversation.  
Your task is to analyze it carefully and separate the content into roles.  

Instructions:  
1. Identify and label each segment of the conversation as either **"Doctor"** or **"Patient"**, based on the context.  
2. Ensure the labeling is accurate and consistent throughout.  
3. Provide the output in **JSON format** as a list of objects, where each object has:  
   - "role": either "Doctor" or "Patient"  
   - "text": the corresponding spoken content  

⚡ Pay special attention to:  
- Correctly distinguishing roles using context clues  
- Preserving the original meaning and medical terminology  
- Maintaining clean and structured JSON formatting  

Example output format:  
{transcript}
"""

gemini_fallback_audio_to_chat_prompt = """
Convert the following audio transcription into a structured chat format between a doctor and a patient.  

Instructions:  
1. Identify and label each part of the conversation as either **"Doctor"** or **"Patient"**.  
2. Ensure that medical terminology is preserved and accurately represented.  
3. Format the output as a **JSON array of objects**, where each object contains:  
   - "role": either "Doctor" or "Patient"  
   - "text": the corresponding spoken content  

⚡ Pay special attention to:  
- Accurate role separation  
- Preserving medical context and terminology  
- Clean and consistent JSON formatting  

Example output format:  
{example}
""".format(example=example_of_chats_template)



raw_scrapped_to_processed_text_system_prompt = """
-You are a Text Filtering and Structuring System who find the best relatable content from the raw text for a quey.
-Also structured the data in best possible way.so it can be easily readable and understandable.
-your task is to filter the raw scrapped text data and structure it into clean, relevant, and concise information.
-Remove any irrelevant information, advertisements, navigation links, or unrelated content.
-filter the raw scrapped text data according to given query.
-filter out the data from given raw data for the given query.
-ensure the final output is well-organized and relatable to the query which should satisy the query in best possible way.
-what i want is sample i want to filter this data {raw_data} for this query {query} and give me the best possible output.
-Provide the output in plain text format without any HTML tags or special characters.
-i am filtering this data ,because i will send this data to llm for further processing.so i want to reduced the size of raw data, want to structure it and want to make it more relevant to the query.
-dont remove any relatable information for the query.

notes:
-dont add any information which is not present in the raw data.
-sometimes the articels raw data may be in different langauge then also convert them in english properly and give me the output in english only.
"""