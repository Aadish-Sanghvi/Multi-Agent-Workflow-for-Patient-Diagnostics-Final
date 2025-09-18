from memory import BioClinicalMemoryAgent

# Initialize and load memory
agent = BioClinicalMemoryAgent()
agent.load_memory("sample_bioclinical_memory.json")

# Your test query (the conversation)
query = """
i have hairfall from pubic area and my hair is thinning from the top of my head.
I have been using minoxidil 5% for 3 months but I don't see any improvement.
"""

result = agent.find_best_match(query)
print("Query:", result["query"])
print("Diagnosis:", result["diagnosis"])