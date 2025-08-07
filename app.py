from flask import Flask, render_template, request, jsonify
import weaviate
import ollama
import warnings
import json
import re
from groq import Groq

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

app = Flask(__name__)

class_name = "StockPDF"

# Read folder names from Folder Names.txt
with open("Folder Names.txt", "r") as file:
    folder_names = [line.strip() for line in file]

intents = {
    "greeting": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"],
    "farewell": ["bye", "goodbye", "see you", "take care", "have a nice day", "until next time"],
    "assistance": ["help", "assist", "support", "can you help", "need assistance", "how do I"],
    "thanks": ["thank you", "thanks", "appreciate it", "grateful", "much appreciated"],
    "weather": ["what's the weather", "is it sunny", "will it rain", "temperature today"],
    "time": ["what time is it", "current time", "clock"],
    "joke": ["tell me a joke", "say something funny", "make me laugh"],
    "compliment": ["you're smart", "good job", "well done", "you're helpful"],
    "complaint": ["this isn't helpful", "you're not understanding", "that's wrong"],
    "smalltalk": ["how are you", "what's up", "how's it going"],
}

responses = {
    "greeting": "Hello! How can I assist you today?",
    "farewell": "Goodbye! Have a great day!",
    "assistance": "Sure, I am here to help. What do you need assistance with?",
    "thanks": "You're welcome! I'm glad I could help.",
    "weather": "I can't provide weather updates right now, but you can check a weather website or app.",
    "time": "I don't have the ability to tell the current time, but you can check a clock or your device.",
    "joke": "Why don't scientists trust atoms? Because they make up everything!",
    "compliment": "Thank you! I appreciate your kind words.",
    "complaint": "I'm sorry to hear that. Could you please provide more details so I can assist you better?",
    "smalltalk": "I'm just a bot, but I'm here to help! How can I assist you today?",
}

def identify_intent(user_input):
    user_input_lower = user_input.lower()
    for intent, keywords in intents.items():
        for keyword in keywords:
            if keyword in user_input_lower:
                return intent
    return None

def handle_user_input(user_input, use_context):
    intent = identify_intent(user_input)
    
    if intent:
        return responses[intent]
    else:
        return rag_with_llm_response(user_input, use_context)

def get_ollama_embedding(text):
    embedding = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return embedding['embedding']

def extract_source_folder(query):
    query_lower = query.lower()
    for folder_name in folder_names:
        if folder_name.lower() in query_lower:
            return folder_name
    return None

def rag_with_llm_response(user_input, use_context):
    if use_context:
        weaviate_client = weaviate.Client("http://localhost:8080")
        try:
            query = user_input
            source_folder = extract_source_folder(query)
            query_embedding = get_ollama_embedding(query)

            search_results = (weaviate_client.query
                              .get(class_name, ["content", "source"])
                              .with_near_vector({"vector": query_embedding}))
            
            if source_folder:
                search_results = search_results.with_where({
                    "path": ["source"],
                    "operator": "Equal",
                    "valueString": source_folder
                })
            
            search_results = search_results.with_limit(15).do()

            # Process the results to get 5 main chunks and their adjacent chunks
            processed_results = []
            results = search_results['data']['Get'][class_name]
            for i in range(0, min(15, len(results)), 3):
                chunk_group = results[i:i+3]
                processed_results.extend(chunk_group)

            context = "\n".join([result['content'] for result in processed_results[:15]])  # Use up to 15 chunks
            
            if source_folder:
                prompt = f"Based on the following context from the folder {source_folder}, \n\nContext: {context}\n\nAnswer: {user_input}"
            else:
                prompt = f"Based on the following context, \n\nContext: {context}\n\nAnswer: {user_input}"

            client = Groq(api_key='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')
            completion = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=8192,
                top_p=0.65,
                stream=False,  # Wait for the entire response
                stop=None,
            )
            
            return completion.choices[0].message.content

        finally:
            weaviate_client = None
    else:
        prompt = f"Answer the following query directly: {user_input}"
        client = Groq(api_key='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')
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=8192,
            top_p=0.65,
            stream=False,  # Wait for the entire response
            stop=None,
        )
        
        return completion.choices[0].message.content

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    use_context = request.json['useContext']
    intent = identify_intent(user_message)
    
    if intent:
        return jsonify({'response': responses[intent]})
    else:
        response = rag_with_llm_response(user_message, use_context)
        return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
