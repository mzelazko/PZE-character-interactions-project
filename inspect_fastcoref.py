import torch
from fastcoref import FCoref

# Test text with clear coreference
text = "Elizabeth Bennet was a bright young woman. She lived in Longbourn. Her family was well-known in the neighborhood."

# Initialize the model (using the default small model for efficiency)
try:
    # Force CPU to avoid potential GPU/memory issues
    model = FCoref(device='cpu')
    
    # Predict coreference clusters
    preds = model.predict(texts=[text])
    
    # Inspect the CorefResult object
    result = preds[0]
    print("Type of result:", type(result))
    print("\nAttributes and methods of CorefResult object:")
    print(dir(result))

except Exception as e:
    print(f"An error occurred during fastcoref inspection: {e}")

# Clean up to free memory
del model
torch.cuda.empty_cache()
