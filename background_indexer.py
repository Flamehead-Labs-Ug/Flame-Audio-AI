import os
import sys
import json
import requests
import time

def main():
    # Get arguments from command line
    backend_url = sys.argv[1]
    auth_token = sys.argv[2]
    request_data_file = sys.argv[3]
    
    # Read request data from file
    with open(request_data_file, 'r') as f:
        request_data = json.load(f)
    
    # Wait a moment to ensure the UI has refreshed
    time.sleep(1)
    
    # Make the API request
    try:
        response = requests.post(
            f"{backend_url}/vectorstore/start_index",
            json=request_data,
            headers={
                "Authorization": f"Bearer {auth_token}"
            },
            timeout=60
        )
        print(f"Indexing API response: {response.status_code}")
    except Exception as e:
        print(f"Error in background indexing: {e}")
    
    # Clean up the temporary file
    try:
        os.remove(request_data_file)
    except:
        pass

if __name__ == "__main__":
    main()
