from memory import BioClinicalMemoryAgent

# Initialize and load memory
agent = BioClinicalMemoryAgent()
agent.load_memory("sample_bioclinical_memory.json")

# Your test query (the conversation)
query = """
24-year-old female presents with sore throat, shortness of breath, and fatigue for several days; presenting with mild constitutional symptoms, vital signs stable with temperature 39.3\u00b0C, heart rate 80 bpm, blood pressure 166/57, oxygen saturation 95% within normal limits. Overall appears well. Provisional diagnosis: Healthy. Differential: viral prodrome, mild dehydration, stress-related symptoms, early viral illness, non-specific malaise. Plan: Reassurance and supportive care, adequate rest and hydration, symptom monitoring. Return for evaluation if symptoms progress or new concerning features develop. Severity: mild presentation requiring rest and fluids.
"""


result = agent.find_best_match(query)
print("Query:", result["query"])
print("Diagnosis:", result["diagnosis"])