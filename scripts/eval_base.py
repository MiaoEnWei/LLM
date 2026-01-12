import pickle
import os

# This is the other file you mentioned
FILE_PATH = "/LLM/PrimeKG/pubmed_documents.pkl"

def inspect_file():
    print(f"Inspecting file: {FILE_PATH} ...")
    
    if not os.path.exists(FILE_PATH):
        print("ERROR: File not found!")
        return

    try:
        with open(FILE_PATH, 'rb') as f:
            data = pickle.load(f)
            
        print(f"Loaded successfully! Data type: {type(data)}")
        
        # If it's a list, print its length and the first item
        if isinstance(data, list):
            print(f"Total length: {len(data)}")
            if len(data) > 0:
                print("--- First item sample ---")
                print(data[0])
                if isinstance(data[0], dict):
                    print(f"Keys included: {data[0].keys()}")
        
        # If it's a dict, print its keys
        elif isinstance(data, dict):
            print(f"Top-level keys: {data.keys()}")
            # Take a look at the first entry
            first_key = list(data.keys())[0]
            print(f"--- Sample ({first_key}) ---")
            print(data[first_key])
            
    except Exception as e:
        print(f"Failed to read: {e}")

if __name__ == "__main__":
    inspect_file()
