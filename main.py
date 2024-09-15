from jarvis import initialize_jarvis, chat
import os

# docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant

if __name__ == "__main__":
    # Custom configurations
    API_KEY = os.getenv('GEMINI_API')
    MODEL_NAME = "gemini-1.5-flash"

    # Initialize Jarvis assistant
    conversation = initialize_jarvis(API_KEY, MODEL_NAME)

    # Start the chat
    chat(conversation)
