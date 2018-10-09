import spacy
from EditDistance import EditDistanceFinder
from LanguageModel import LanguageModel
import string


class SpellChecker(object):

    def __init__(self, max_distance, channel_model=None, language_model=None):
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

    def unigram_score(self, word):
        return self.language_model.unigram_prob(word)

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

    def cm_score(self, error_word, corrected_word):
        return self.channel_model.align(error_word, corrected_word)[0]

    def generate_candidates(self, word):
        source = [word]
        for i in range(self.max_distance):
            nested = list(map(self._one_step, source))
            flat = [l for sublist in nested for l in sublist]
            source = list(set(flat))
        return source

    def check_sentence(self, sentence, fallback=False):
        l = []
        for i in range(len(sentence)):
            word = sentence[i]
            if word in self.language_model:
                l.append([word])
            else:
                choices = self.generate_candidates(word)
                if len(choices) == 0:
                    if fallback:
                        l.append([word])
                else:
                    if i<1:
                        prev_word = '%'
                    else:
                        prev_word = sentence[i-1]

                    if i+1 == len(sentence):
                        next_word = '%'
                    else:
                        next_word = sentence[i+1]

                    #rank = lambda x: self.cm_score(x, word)
                    #rank = lambda x: self.bigram_score(prev_word, x, next_word)
                    rank = lambda x: self._combine_scores(self.cm_score(x, word), self.bigram_score(prev_word, x, next_word))
                    ranked = sorted(choices, key = rank, reverse=False)
                    l.append(list(ranked))


        return l

    def _combine_scores(self, cm_score, bigram_score):
        return 10*cm_score + bigram_score


    def _one_step(self, word):
        return self.inserts(word) + self.deletes(word) + self.substitutions(word)

    def autocorrect_sentence(self, sentence):
        options = self.check_sentence(sentence, fallback=True)
        print(options)
        return [x[0] for x in options]

    def suggest_sentence(self, sentence, max_suggestions):
        options = self.check_sentence(sentence)
        get = lambda x: x[0] if len(x) == 0 else x[:max_suggestions]
        return [get(x) for x in options]


s = SpellChecker(5)

with open('lm.pkl', 'rb') as fp:
    s.load_language_model(fp)

with open('ed.pkl','rb') as fp:
    s.load_channel_model(fp)

print(s.suggest_sentence(['it', 'was', 'the', 'best', 'of', 'times', 'it', 'was', 'the', 'borst', 'of', 'times'], 4))
