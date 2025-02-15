from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead
import os

app = Flask(__name__)

# Инициализация модели и токенизатора
tokenizer = AutoTokenizer.from_pretrained('tinkoff-ai/ruDialoGPT-medium')
model = AutoModelWithLMHead.from_pretrained('tinkoff-ai/ruDialoGPT-medium')

def generate_response(context):
    inputs = tokenizer(context, return_tensors='pt')
    generated_token_ids = model.generate(
        **inputs,
        top_k=10,
        top_p=0.95,
        num_beams=3,
        num_return_sequences=1,
        do_sample=True,
        no_repeat_ngram_size=2,
        temperature=1.2,
        repetition_penalty=1.2,
        length_penalty=1.0,
        eos_token_id=50257,
        max_new_tokens=40
    )
    response = tokenizer.decode(generated_token_ids[0])
    return response.split('@@ВТОРОЙ@@')[-1].strip()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.json
    message = data['message']
    dialog_history = data['history']
    
    # Формируем контекст для модели
    context = ''
    for i, msg in enumerate(dialog_history + [message]):
        speaker = '@@ПЕРВЫЙ@@' if i % 2 == 0 else '@@ВТОРОЙ@@'
        context += f' {speaker} {msg}'
    context += ' @@ВТОРОЙ@@'
    
    response = generate_response(context)
    return jsonify({'response': response})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
