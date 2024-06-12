# %%
%pip install gensim
# %pip uninstall scipy
%pip install scipy==1.10.1
# %%
from gensim.models import Word2Vec
# %%
# Define the corpus
corpus = [
    "the cat sat on the mat",
    "the dog played in the yard",
    "birds chirped in the trees"
]

# Tokenize the corpus
tokenized_corpus = [sentence.split() for sentence in corpus]

# Train the Word2Vec model
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

# Get word vectors
word_vectors = model.wv

# Print the word vectors
for word in word_vectors.index_to_key:
    print(f"Word: {word}, Vector: {word_vectors[word]}")

# %%
