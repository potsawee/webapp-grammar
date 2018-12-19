import sys
import os
import pdb
sys.path.insert(0,'/home/alta/BLTSpeaking/ged-pm574/local/seq2seq/')

import tensorflow as tf
from flask import Flask, request, render_template
from model import EncoderDecoder
from helper import load_vocab, read_config

model_path = '/home/alta/BLTSpeaking/ged-pm574/nmt-exp/lib/models/clc-gec-exp1'
config_path = model_path + '/config.txt'
config = read_config(config_path)
config['model_path'] = model_path

# ----------- NMT model ----------- #
class Translator(object):
    def __init__(self, config):

        self.config = config
        self.model_number = self.config['num_epochs'] - 1
        self.save_path = self.config['model_path'] + '/model-' + str(self.model_number)

        self.vocab_paths = {'vocab_src': self.config['vocab_src'], 'vocab_tgt': self.config['vocab_tgt']}
        self.src_word2id, self.tgt_word2id = load_vocab(self.vocab_paths)
        self.tgt_id2word = list(self.tgt_word2id.keys())


        self.params = {'vocab_src_size': len(self.src_word2id),
                'vocab_tgt_size': len(self.tgt_word2id),
                'go_id':  self.tgt_word2id['<go>'],
                'eos_id':  self.tgt_word2id['</s>']}

        # build the model
        self.model = EncoderDecoder(config, self.params)
        self.model.build_network()

        # session
        self.sess = tf.Session(config=tf.ConfigProto())

        saver = tf.train.Saver()
        saver.restore(self.sess, self.save_path)

    def __del__(self):
        self.sess.close()

    def translate(self, sentence):
        sent_ids = []
        for word in sentence.split():
            if word in self.src_word2id:
                sent_ids.append(self.src_word2id[word])
            else:
                sent_ids.append(self.src_word2id['<unk>'])
        sent_len = len(sent_ids)

        feed_dict = {self.model.src_word_ids: [sent_ids],
            self.model.src_sentence_lengths: [sent_len],
            self.model.dropout: 0.0}

        [translations] = self.sess.run([self.model.translations], feed_dict=feed_dict)
        words = []
        for id in translations[0]:
            if id == self.params['eos_id']:
                break
            words.append(self.tgt_id2word[id])

        return 'output: {}'.format(' '.join(words))


# ------------ Web App ------------ #
os.environ['CUDA_VISIBLE_DEVICES'] = '' # disable GPU
translator = Translator(config)

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('page.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    x = translator.translate(text)
    return x
    # processed_text = text.upper()
    # return processed_text

if __name__ == '__main__':
    app.run()
