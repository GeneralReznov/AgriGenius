import os
import google.generativeai as genai

# Set your API key here
os.environ['GOOGLE_API_KEY'] = "AIzaSyCZGGDVIyjebUyHX8m0xO6f1pBD6KKjErc"

# Configure the Gemini API with your API key
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Initialize the chat model
chat_model = genai.GenerativeModel('gemini-2.0-flash')

# Start a chat session
chat = chat_model.start_chat(history=[])

# Function to send messages and get responses
def get_response(user_input):
    response = chat.send_message(user_input)
    return response.text

# Example conversation
print("Chatbot: Hello! How can I assist you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Chatbot: Goodbye!")
        break
    response = get_response(user_input)
    print(f"Chatbot: {response}")