import sys
import os
import pdb

from flask import Flask, request, render_template
from gec import GEC
from controller import Controller

# ------------ Web App ------------ #
os.environ['CUDA_VISIBLE_DEVICES'] = '' # disable GPU
path = '/home/alta/BLTSpeaking/ged-pm574/nmt-exp/lib/models/correction/scheduled'
gec = GEC(path)
controller = Controller()

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('page.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['textbox']
    sentences = controller.split(text)
    output_lines = []
    for sent in sentences:
        words = gec.translate(sent)
        output_lines.append(' '.join(words))
    output_text = '<xmp>'
    counter = 0
    for a,b in zip(sentences, output_lines):
        counter += 1
        output_text += '------------ sentence #' + str(counter) + ' ------------\n\n'
        output_text += 'Input:  {}\n\n'.format(a.strip('.'))
        output_text += 'Output: {}\n\n'.format(b.strip('.'))
    output_text += '-------------------------------------\n\n'
    output_text += '</xmp>'
    return output_text


if __name__ == '__main__':
    app.run()
