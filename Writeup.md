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

`get_chunks` uses the python `yield` to return the first (then second... then nth) "chunk" of the files it was given. the `yield` + `for chunk in` combination lets us generate the next value of the iterator when needed for all the chunks in all the files. practically it lets us go through the whole set of files while only holding one chunk_size worth of the files at a time.


7. Describe the command-line interface for `LanguageModel.py`. What command should you run to generate a model from `/data/gutenberg/*.txt` and save it to `lm.pkl` if you want an alpha value of 0.1 and a vocabulary size of 40000?

our command:

`python LanguageModel.py -s lm.pkl -a 0.1 -v 40000 /data/gutenberg/*.txt`

however some of these args use default values equal to what we want so this should be equivalent:

`python LanguageModel.py -s lm.pkl /data/gutenberg/*.txt`


6. How often did your spell checker do a better job of correcting than ispell? Conversely, how often did ispell do a better job than your spell checker?

I had difficulty using diff because the output of auto spell was not the same as that of the ispell file. I could have fixes this but I ran out of time.


7. Can you characterize the type of errors your spell checker tended to best at, and the type of errors ispell tended to do best at?

8. Comment on anything else you notice that is interesting about spell checking – either for your model or for ispell.

I lowercased at the wrong time and my spellcheck is really bad at names now

Tokenization ruins a common mistake I see of adding the space too early (int he) vs (in the)


# Transpositions

9. Describe your approach

I'm going to add Transpositions to the generated list of words, then add this metric to the edit distance finder. Either part without the other would lead to different results
For a very simple way of calculating transpose cost, I'm averaging taking the two substitution costs.

10. Give examples of how your approach works, including specific sentences where your new model gives a different (hopefully better!) result than the baseline model.

Old behavior:

`s.suggest_text('this is the bset',4) => [['this'], ['is'], ['the'], ['sea', 'lord', 'son', 'be']]`

New behavior:

`s.suggest_text('this is the bset',4) => [['this'], ['is'], ['the'], ['sea', 'best', 'lord', 'son']]`

It still REALLY likes the sea (i think old books will give you that bias) but now it suggests 'best' at least at number 2

11. Discuss any challenges you ran into, design decisions you made, etc.

The actual "probability" of swapping was never learned (I didn't teach it to do that) so my probabilities are very much off. It works! but not as well as a real model would.
