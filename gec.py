import sys
seq2seq_repo = '/home/alta/BLTSpeaking/ged-pm574/local/seq2seq/'
sys.path.insert(0, seq2seq_repo)

import tensorflow as tf
from spellchecker import SpellChecker
from model import EncoderDecoder
from helper import load_vocab, read_config

# ----------- NMT model ----------- #
class GEC(object):
    def __init__(self):
        pass

    def __init__(self, path):
        self.build_model(path)

    def __del__(self):
        self.sess.close()

    def build_model(self, path):
        print('building model...')
        # path e.g. '/home/alta/BLTSpeaking/ged-pm574/nmt-exp/lib/models/correction/scheduled'
        self.config_path = path + '/config.txt'
        self.config = read_config(self.config_path)
        self.config['model_path'] = path

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
        self.model = EncoderDecoder(self.config, self.params)
        self.model.build_network()

        # session
        self.sess = tf.Session(config=tf.ConfigProto())

        saver = tf.train.Saver()
        saver.restore(self.sess, self.save_path)

    def translate(self, sentence):
        sent_ids = []
        spell = SpellChecker()

        for word in sentence.split():
            if word in self.src_word2id:
                sent_ids.append(self.src_word2id[word])
            else:
                x = spell.correction(word)
                if x in self.src_word2id:
                    print("Spellcheck: {} => {}".format(word, x))
                    sent_ids.append(self.src_word2id[x])
                else:
                    sent_ids.append(self.src_word2id['<unk>'])


        # add dot at the end
        sent_ids.append(self.src_word2id['.'])

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

        # return 'output: {}'.format(' '.join(words))
        return words
