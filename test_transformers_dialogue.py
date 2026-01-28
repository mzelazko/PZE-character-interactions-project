from transformers import pipeline
import re

# Load a model suitable for zero-shot classification or similar
# For dialogue attribution, we might need a more specific approach, 
# but let's try a general one first to see if it can identify speakers.
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = """
"I hope Mr. Bingley will like it," said Jane.
"He is a very fine young man," replied Elizabeth.
"""

# Extract quotes and surrounding context
quotes = re.findall(r'"([^"]*)"', text)
contexts = re.split(r'"[^"]*"', text)

candidate_labels = ["Jane", "Elizabeth", "Mr. Bingley"]

for i, quote in enumerate(quotes):
    context = contexts[i+1].strip() if i+1 < len(contexts) else ""
    print(f"Quote: {quote}")
    print(f"Context: {context}")
    
    # Try to attribute
    result = classifier(quote + " " + context, candidate_labels)
    print(f"Attributed to: {result['labels'][0]} (Score: {result['scores'][0]:.4f})")
    print("-" * 20)
