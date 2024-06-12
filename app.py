import os
from flask import Flask, request, jsonify, render_template
from transformers import MarianMTModel, MarianTokenizer

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

app = Flask(__name__)

def load_model(src_lang, tgt_lang):
    if src_lang == "en" and tgt_lang == "tr":
        model_name = 'Helsinki-NLP/opus-tatoeba-en-tr'
    elif src_lang == "tr" and tgt_lang == "en":
        model_name = 'Helsinki-NLP/opus-mt-tr-en'
    else:
        return None, None

    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate(text, src_lang, tgt_lang):
    tokenizer, model = load_model(src_lang, tgt_lang)
    if not tokenizer or not model:
        return "Model not available for the given language pair."

    src_text = [text]
    tgt_text = None  # This can be set if you have target text for translation, otherwise, keep it None
    inputs = tokenizer(src_text, text_pair=tgt_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    translated_tokens = model.generate(input_ids=input_ids, attention_mask=attention_mask)
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    return translated_text[0]


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate_text():
    try:
        data = request.json
        text = data.get('text', '')
        source_lang = data.get('source_lang', 'en')
        target_lang = data.get('target_lang', 'tr')
        src_lang = source_lang.split('-')[0]
        tgt_lang = target_lang.split('-')[0]
        translated_text = translate(text, src_lang, tgt_lang)
        return jsonify({'translated_text': translated_text})
    except Exception as e:
        print("Error occurred:", str(e))
    return jsonify({'error': 'An internal server error occurred.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
