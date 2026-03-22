from gradio_client import Client
import time

# Replace <username> and <space_name> with your real values
client = Client("https://huggingface.co/spaces/ssaahhil0317E/spam-detector")

def check_message(text):
    result = client.predict(text, api_name="/predict")
    return result

if __name__ == "__main__":
    while True:
        msg = "Congratulations! You have won a free vacation. Click here to claim."
        print("Checking message:", msg)
        print("Result:", check_message(msg))
        time.sleep(60)  # waits 1 minute before next check
