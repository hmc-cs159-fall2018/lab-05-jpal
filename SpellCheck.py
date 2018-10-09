import spacy
from EditDistance import EditDistanceFinder
from LanguageModel import LanguageModel

class SpellChecker(object):

    def __init__(self, channel_model=None, language_model=None, max_distance):
        self.channel_model = channel_model
        self.language_model = language_model
        self.max_distance = max_distance
        self.nlp = spacy.load("en", pipeline=["tagger", "parser"])

    def load_channel_model(self, fp):
        self.channel_model = EditDistanceFinder()
        self.channel_model.load(fp)

    def load_language_model(self, fp):
        self.language_model = LanguageModel()
        self.language_model.load(fp)

    def bigram_score(self, prev_word, focus_word, next_word):
        score = lambda x, y: self.language_model.bigram_prob(x,y)
        return (score(prev_word, focus_word) + score(focus_word, next_word))/(2.0)

    def inserts(self, word):
        l = []
        for i in range(len(word)):
            for char in string.ascii_lowercase:
                l.append(word[:i] + char + word[i:])
        return [x for x in l if x in self.language_model]

    def deletes(self, word):
        l = []
        for i in range(len(word)):
            l.append(word[:i] + word[i+1:])
        return [x for x in l if x in self.language_model]

    def substitutions(self, word):
        l = []
        for i in range(len(word)):
            for char in string.ascii_lowercase:
                l.append(word[:i] + char + word[i+1:])
        return [x for x in l if x in self.language_model]

    def unigram_score(self, word):
        return self.language_model.unigram_prob(w)

    def cm_score(self, error_word, corrected_word):
        return self.channel_model.align(error_word, corrected_word)[0]

    def generate_candidates(self, word):
        source = [word]
        for i in range(self.max_distance):
            nested = list(map(self._one_step, source))
            flat = [l for sublist in nested for l in sublist]
            source = flat
        return flat

    def _one_step(self, word):
        return self.inserts(word) + self.deletes(word) + self.substitutions(word)
