import requests

try:
    response = requests.get("https://www.google.com")
    print("Google Connection successful:", response.status_code)
    response = requests.get("https://huggingface.co")
    print("Huggingface Connection successful:", response.status_code)
except requests.ConnectionError as e:
    print("Connection failed:", e)