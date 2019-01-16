class Controller(object):
    def __init__(self):
        self.sentences = []

    def split(self, text):
        sents = text.replace('.', '\n').split('\n')
        sentences = []
        for sent in sents:
            sent = sent.strip()
            if sent == '':
                continue
            sent = sent.lower()
            sentences.append(sent)
        return sentences
