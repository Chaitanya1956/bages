from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load GPT-2 model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

@app.route('/')
def home():
    return 'Welcome to the GPT-2 Chatbot!'

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']

    # Tokenize input text
    input_ids = tokenizer.encode(user_message, return_tensors='pt')

    # Generate response
    response_ids = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
    bot_response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)