from flask import Flask, jsonify, request

from mosestokenizer import MosesSentenceSplitter, MosesTokenizer, MosesDetokenizer

from indicnlp.tokenize import sentence_tokenize, indic_tokenize, indic_detokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

import codecs
from subword_nmt.apply_bpe import BPE

import ctranslate2

from waitress import serve

from flask_cors import CORS

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['JSON_AS_ASCII'] = False
app.config['JSON_SORT_KEYS'] = False
CORS(app)

# En-Mr
## Split
splitsents_en = MosesSentenceSplitter('en')
## Tokenize
tokenize_en = MosesTokenizer('en')
detokenize_en = MosesDetokenizer('en')
## BPE
codes_en = codecs.open("en-mr/v2/bpe-codes/codes.en", encoding='utf-8')
bpe_en = BPE(codes_en)
## Translate
translator_enmr = ctranslate2.Translator("en-mr/v2/model_deploy/", inter_threads=4, intra_threads=1)

# Mr-En
# Normalize
factory=IndicNormalizerFactory()
normalizer_mr=factory.get_normalizer("mr")
## BPE
codes_mr = codecs.open("mr-en/v2/bpe-codes/codes.mr", encoding='utf-8')
bpe_mr = BPE(codes_mr)
## Translate
translator_mren = ctranslate2.Translator("mr-en/v2/model_deploy/", inter_threads=4, intra_threads=1)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        # Read input request
        input_json = request.get_json()
        
        input_config = input_json["config"]
        input_lang = input_config["language"]
        input_src, input_trg = input_lang["sourceLanguage"], input_lang["targetLanguage"]
        
        if input_src == input_trg:
            out = '\n'.join(inp["source"] for inp in input_source)
            print(out)
            response_body = {
                "config": input_config,
                "output": out
            }
            return jsonify(response_body), 201
        
        input_source = input_json["input"]
        output_target = []
        
        for input_text_json in input_source:
            input_text = input_text_json["source"].strip("\n")
            # Paragraphs
            input_paras = input_text.split("\n")
            input_list = [line for line in input_paras if len(line) > 1]

            output=[]
            # Iterate Over Paragraphs
            for paras in input_list:
                if input_src == 'en':

                    # Split Sentences
                    inp_lines = splitsents_en([paras.strip('\n')])

                    # Lowercase
                    inp_lines = [line.lower() for line in inp_lines]
                    inp_lines = [line+'.' if line[-1] != '.' else line for line in inp_lines ]

                    # Tokenize
                    inp_lines = [' '.join(tokenize_en(line)) for line in inp_lines]

                    # Apply BPE
                    inp_lines = [bpe_en.process_line(line).split(" ") for line in inp_lines]

                    # Translate
                    out_lines = translator_enmr.translate_batch(inp_lines, beam_size=5, max_batch_size=16)

                    # Remove BPE
                    out_lines = [(' '.join(line.hypotheses[0]) + " ").replace("@@ ", "") for line in out_lines]

                    # Post Processing
                    out_lines = [line.replace('"', '').replace("u200d", "").strip() for line in out_lines]

                    # Detokenize
                    out_lines = [indic_detokenize.trivial_detokenize(line) for line in out_lines]
                    

                elif input_src == 'mr':

                    # Split Sentences
                    inp_lines = sentence_tokenize.sentence_split(paras, "mr")

                    # Normalize
                    inp_lines = [normalizer_mr.normalize(line) for line in inp_lines]

                    # Tokenize
                    inp_lines = [' '.join(indic_tokenize.trivial_tokenize(line)) for line in inp_lines]

                    # Apply BPE
                    inp_lines = [bpe_mr.process_line(line).split(" ") for line in inp_lines]

                    # Translate
                    out_lines = translator_mren.translate_batch(inp_lines, beam_size=5, max_batch_size=16)

                    # Remove BPE
                    out_lines = [(' '.join(line.hypotheses[0]) + " ").replace("@@ ", "").replace("u200d",'').replace('"', '').strip() for line in out_lines]

                    # Detokenize
                    out_lines = [detokenize_en(line.split(" ")) for line in out_lines]

                output.append(out_lines)

            output_text = '\n'.join([' '.join(lines) for lines in output])

            output_target_json = {}
            output_target_json["source"] = input_text
            output_target_json["target"] = output_text

            output_target.append(output_target_json)

        response_body = {
            "config": input_config,
            "output": output_target
            }
        return jsonify(response_body), 201
    else:
        response_body = {
                "config": input_config,
                "output": " "
                }
        return jsonify(response_body), 400

if __name__ == '__main__':
    # app.run(host="0.0.0.0", port=8080, debug=True) # Development mode
    serve(app, host='0.0.0.0', port=8080)   # Production mode
