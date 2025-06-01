from flask import Flask, render_template, request
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

app = Flask(__name__)

# Load pre-trained multilingual model (MBart50)
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# Language codes for MBart50
lang_code_map = {
    'English': 'en_XX',
    'Hindi': 'hi_IN',
    'Telugu': 'te_IN',
    'Tamil': 'ta_IN',
    'Bengali': 'bn_IN',
    'Kannada': 'kn_IN'
}

@app.route("/", methods=["GET", "POST"])
def index():
    translation = ""
    if request.method == "POST":
        src_lang = request.form["src_lang"]
        tgt_lang = request.form["tgt_lang"]
        text = request.form["text"]

        tokenizer.src_lang = lang_code_map[src_lang]
        encoded = tokenizer(text, return_tensors="pt")
        generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.lang_code_to_id[lang_code_map[tgt_lang]])
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    return render_template("index.html", translation=translation)

if __name__ == "__main__":
    app.run(debug=True)
