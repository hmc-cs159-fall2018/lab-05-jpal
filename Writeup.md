1. In Writeup.md, explain how Laplace smoothing works in general and how it is implemented in the EditDistance.py file. Why is Laplace smoothing needed in order to make the prob method work? In other words, the prob method wouldn’t work properly without smoothing – why?

Laplace smoothing is a form of smoothing which simply adds a constant (usually 1) to the values of all possible combinations of things we've seen (we say Laplace in the case of bi-gram probabilities) before these counts are turned into probabilities.

in train_costs, we add 0.1 to the counts of all possible alignments
```
for a in alphabet:
    for b in alphabet:
        counts[a][b] += .1
```

If we're going to take the log of a probability, it better not be zero. We use smoothing to avoid this possibility.

2. Describe the command-line interface for EditDistance.py. What command should you run to generate a model from /data/spelling/wikipedia_misspellings.txt and save it to ed.pkl?

Our command line interface requires a source arg to pull data from, and a store arg to save the trained model to. Only the source arg is marked as required, but without a source the program errors out.

Here is this example command:
`python EditDistance.py --source /data/spelling/wikipedia_misspellings.txt  --store ed.pkl`



3.What n-gram orders are supported by the given `LanguageModel` class?

only unigrams and bigrams (there exists a `self.unigrams` and a `self.bigrams` and a `unigram_prob` `bigram_prob`)

4. How does the given `LanguageModel` class deal with the problem of 0-counts?

The LanguageModel adds a value `alpha` to every count before calculating probability. it also takes into account this additional alpha of counts added to properly convert to real probabilities.

```
numerator = self.unigrams[word] + self.alpha
denominator = sum(self.unigrams.values()) + (self.alpha * self.V)
return log(numerator/denominator)
```

5. What behavior does the `__contains__()` method of the `LanguageModel` class provide?

The function itself checks if a given word is in the vocabulary. The naming conventions lets us do fun things.

`word in self.vocabulary` and `word in self` now behave the same, because calling `in self` is asking if `self.__contains__` the value.


6. Spacy uses a lot of memory if it tries to load a very large document. To avoid that problem, `LanguageModel` limits the amount of text that’s processed at once with the `get_chunks` method. Explain how that method works.

`get_chunks` uses the python `yield` to return the first (then second... then nth) "chunk" of the files it was given. the `yield` + `for chunk in` combination lets us generate the next value of the iterator when needed for all the chunks in all the files.


7. Describe the command-line interface for `LanguageModel.py`. What command should you run to generate a model from `/data/gutenberg/*.txt` and save it to `lm.pkl` if you want an alpha value of 0.1 and a vocabulary size of 40000?

our command:

`python LanguageModel.py -s lm.pkl -a 0.1 -v 40000 /data/gutenberg/*.txt`

however some of these args use default values equal to what we want so this should be equivalent:

`python LanguageModel.py -s lm.pkl /data/gutenberg/*.txt`
