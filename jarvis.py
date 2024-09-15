import logging
import os
import json
import hashlib
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables.history import RunnableWithMessageHistory
import google.generativeai as genai
import uuid

from vectordb import (
    initialize_qdrant,
    store_headers_in_qdrant,
    search_headers_in_qdrant,
    get_info_by_id,
    update_info_in_qdrant
)

genai.configure(api_key=os.getenv('GEMINI_API'))


model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize Qdrant and in-memory info storage
qdrant = initialize_qdrant()

def merge_info(existing_info, new_info):
    # Simple concatenation, you can implement more complex logic
    if existing_info == "mention":
        return new_info
    elif new_info == "mention":
        return existing_info
    else:
        return existing_info + " " + new_info

# Model and memory initialization
def initialize_jarvis(api_key, model_name):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.getLogger('absl').setLevel(logging.ERROR)

    llm = ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model=model_name,
        temperature=1,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    memory = ConversationBufferMemory(return_messages=True)

    def get_session_history():
        return memory.chat_memory

    conversation = RunnableWithMessageHistory(
        runnable=llm,
        get_session_history=get_session_history,
        memory=memory
    )
    
    return conversation

# Use SentenceTransformers for embedding generation
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('all-mpnet-base-v2') # Choose an appropriate model

def get_embedding(text):
    return embedding_model.encode(text).tolist() 

def chat(conversation):
    print("Start chatting with Jarvis (type 'exit' to stop):")
    while True:
        prompt = input("Barathan: ")
        if prompt.lower() == 'exit':
            print("Exiting chat.")
            break

        # Prepare headers prompt
        headers_prompt = f"""You are a highly intelligent assistant. Extract relevant topics from the following message and provide multiple headers for each topic.

For each topic:
- Provide 3-4 headers, including both general and specific terms.
- Headers should be 4 to 6 words.
- Try to create headers as generalized as possible while sticking close to the topic.
- Identify whether the message contains just a mention or provides detailed information.
- If there is information, provide a summary without leaving any details.

Examples:

1. Barathan's Message: "Hi Jarvis, Batuhan and I are going to continue our project on the t3 hackathon. Batuhan has been working on the frontend for the past two weeks."

Output:
[
    {{
        "headers": ["Barathan's project friend.", "Barathan's project partner", "Barathan's frontend developer friend", "Batuhan's project work"],
        "info": "Batuhan has been working on the frontend for the past two weeks."
    }},
    {{
        "headers": ["T3 Hackathon competition of Barathan", "Barathan and Batuhan's hackathon project", "Barathan's upcoming competition"],
        "info": "mention"
    }}
]

2. Barathan's Message: "Hey, I just had a meeting with Dr. Smith about our AI project. He suggested we use reinforcement learning for the next phase."

Output:
[
    {{
        "headers": ["Barathan's mentor named Dr. Smith", "Barathan's project advisor", "Barathan's meeting with professor", "Dr. Smith, coding project mentor"],
        "info": "mention"
    }},
    {{
        "headers": ["Barathan's AI project", "Barathan and Dr. Smith's research project", "Barathan's reinforcement learning project"],
        "info": "Dr. Smith suggested using reinforcement learning for the next phase."
    }}
]

Now process the following message from Barathan:
"{prompt}"
"""
        response = model.generate_content(headers_prompt)

        # Inside your chat function
        try:
            # Check if candidates are returned
            if not response.candidates:
                logging.error("No response candidates returned by the model.")
                # You can also check safety ratings here
                # continue

            # Access the response content correctly
            response_text = response.text.strip()

            print(response_text)

            # Remove code block delimiters if present
            if response_text.startswith("```"):
                response_text = response_text.split('\n', 1)[1]
            if response_text.endswith("```"):
                response_text = response_text.rsplit('\n', 1)[0]
            extracted_headers = json.loads(response_text)
        except Exception as e:
            logging.error("Error processing headers: %s", e)
            continue

        retrieved_info = []
        for item in extracted_headers:
            headers = item['headers']
            info = item['info']

            for header in headers:
                # Generate a valid UUID for each header
                header_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, header))

                # Generate embedding
                header_embedding = get_embedding(header)

                # Search for similar headers
                search_results = search_headers_in_qdrant(qdrant, header_embedding)

                if not search_results:
                    # No similar headers found
                    store_headers_in_qdrant(qdrant, header, header_id, header_embedding, info)
                else:
                    # Similar headers found
                    closest_result = search_results[0]
                    existing_id = closest_result.id
                    existing_info = closest_result.payload.get('info', '')
                    if info != existing_info:
                        # Merge the information
                        updated_info = merge_info(existing_info, info)
                        update_info_in_qdrant(qdrant, existing_id, updated_info)
                        retrieved_info.append(updated_info)
                    else:
                        retrieved_info.append(existing_info)

        # Generate assistant's response
        retrieved_info_text = "\n".join(retrieved_info)
        full_prompt = f"""You are Jarvis, an intelligent assistant designed to help Barathan.

You have access to the following information (If there is any):
{retrieved_info_text}

---

Using the information above, provide a helpful and context-aware response to Barathan:

Barathan: {prompt}

"""
        response = conversation.invoke({"input": full_prompt})

        content = response.content
        # Handle token usage as needed

        print("Jarvis:", content)