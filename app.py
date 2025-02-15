import os
from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

app = Flask(__name__)

# Инициализация модели и токенизатора
tokenizer = AutoTokenizer.from_pretrained('tinkoff-ai/ruDialoGPT-medium')
model = AutoModelWithLMHead.from_pretrained('tinkoff-ai/ruDialoGPT-medium')

# ... остальной код остается без изменений ...

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 
