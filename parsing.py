# ==========================================
# Install dependencies (for local testing)
# ==========================================
# pip install google-generativeai

# ==========================================
# Imports
# ==========================================
import google.generativeai as genai
from config import gemini_api_key

# ==========================================
# Gemini Client Initialization (2.5 Flash)
# ==========================================
try:
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash') 
    print("✅ Gemini 2.5 Flash API initialized successfully!")
except Exception as e:
    print(f"⚠️ Failed to initialize Gemini: {str(e)}. Query parsing disabled.")
    gemini_model = None

# ==========================================
# Functions
# ==========================================
def transform_to_pubmed_query(final_query: str) -> str:
    """Transform final query to PubMed search query using Gemini 2.5 Flash."""
    if gemini_model is None:
        return "❌ Gemini model not available."

    prompt = f"""
    You are a medical query parser for PubMed. From the doctor-patient conversation: '{final_query}',
    perform the following:
    1. Identify the primary disease and 1-2 key symptoms or treatments.
    2. Map the primary disease and symptoms to MeSH terms where possible (e.g., 'diabetes' to 'Diabetes Mellitus' [MeSH Terms]).
    3. Use AND to combine the disease and symptoms, and OR for critical synonyms only if essential.
    4. Apply filters: articles from the last 5 years, clinical trials or reviews, and human studies.
    5. Output ONLY a concise PubMed query string, avoiding excessive terms or nested parentheses.
    
    Example Input: 'Patient with fever and cough for 2 weeks, suggestive of pneumonia'
    Example Output: '"Pneumonia" [MeSH Terms] AND ("Fever" [MeSH Terms] OR "Cough" [MeSH Terms]) AND ("2020/01/01"[PDAT] : "2025/12/31"[PDAT]) AND (clinical trial[PT] OR review[PT]) AND humans[Filter]'
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error generating PubMed query: {str(e)}"

def transform_to_who_query(final_query: str) -> str:
    """Transform final query to WHO search query using Gemini 2.5 Flash."""
    if gemini_model is None:
        return "❌ Gemini model not available."

    prompt = f"""
    You are a medical query parser for the WHO website (who.int) and data portal (https://data.who.int). From the doctor-patient conversation: '{final_query}',
    perform the following:
    1. Extract the primary disease and 1-2 key symptoms.
    2. Combine with terms like "guidelines" or "epidemiology" using AND/OR operators and quotes for phrases.
    3. Include 'site:who.int' to restrict to WHO website; mention 'data.who.int' if epidemiological data is relevant.
    4. Keep the query concise, avoiding broad terms like 'symptoms' or 'prevention' unless critical.
    5. Output ONLY a concise WHO query string.
    
    Example Input: 'Patient with fever and cough for 2 weeks, suggestive of pneumonia'
    Example Output: '"pneumonia" AND ("guidelines" OR "epidemiology") site:who.int'
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error generating WHO query: {str(e)}"

def parse_query_for_agents(final_query: str) -> dict:
    """Parse final query into PubMed and WHO queries for web scraping agent."""
    pubmed_query = transform_to_pubmed_query(final_query)
    who_query = transform_to_who_query(final_query)
    
    return {
        "pubmed_query": pubmed_query,
        "who_query": who_query
    }

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    final_query = '''
    "19-year-old female presents with fatigue, fever, and runny nose for several days; acute onset with high fever 39.8°C, tachycardia 101 bpm, blood pressure 100/67, reduced oxygen saturation 91% concerning for pneumonia. Physical exam may reveal crackles, bronchial breathing, or dullness to percussion. Provisional diagnosis: Pneumonia. Differential: bacterial pneumonia, viral pneumonia, atypical pneumonia, pulmonary embolism (if sudden onset), pneumothorax. Plan: Immediate antibiotic therapy, supplemental oxygen if SpO2 <92%, chest X-ray for confirmation, blood cultures, consider hospitalization given severity. Close monitoring of vital signs and oxygenation. Severity: severe presentation requiring hospitalization and medication.
    '''
    result = parse_query_for_agents(final_query)
    print("PubMed Query:", result["pubmed_query"])
    print("WHO Query:", result["who_query"])