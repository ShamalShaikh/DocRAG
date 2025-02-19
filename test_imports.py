import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("Testing imports...")

try:
    from src.database import collection
    print("✓ Database import successful")
except Exception as e:
    print(f"✗ Database import failed: {str(e)}")

try:
    from src.generate_embeddings import model
    print("✓ Embedding model import successful")
except Exception as e:
    print(f"✗ Embedding model import failed: {str(e)}")

try:
    import ollama
    print("✓ Ollama import successful")
    models = ollama.list()
    print(f"Available models: {models}")
except Exception as e:
    print(f"✗ Ollama import/connection failed: {str(e)}")

print("\nTesting database connection...")
try:
    count = collection.count()
    print(f"✓ Database connection successful. Collection has {count} items")
except Exception as e:
    print(f"✗ Database connection failed: {str(e)}")

print("\nTesting embedding model...")
try:
    test_embedding = model.encode(["test sentence"])
    print(f"✓ Embedding model working. Output shape: {test_embedding.shape}")
except Exception as e:
    print(f"✗ Embedding model failed: {str(e)}")
