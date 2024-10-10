import numpy as np

def random_one_hot(p):
    # Crée un vecteur de zéros de taille p
    one_hot_vector = np.zeros((p,1))
    # Choisit aléatoirement un index entre 0 et p-1
    random_index = np.random.randint(0, p)
    # Place un 1 à l'index choisi
    one_hot_vector[random_index] = 1
    return one_hot_vector

##################################
# hyperparameters for our GPT

# vocab size is 2, so we only have two possible tokens: 0,1
vocab_size = 2
# context length is 3, so we take 3 bits to predict the next bit probability
context_length = 3

def all_possible(n, k):
    # return all possible lists of k elements, each in range of [0,n)
    if k == 0:
        yield []
    else:
        for i in range(n):
            for c in all_possible(n, k - 1):
                yield [i] + c

def all_possible_small(n, k):
    # return all possible lists of k elements, each in range of [0,n)
    l = []
    for c in range(1,context_length):
      l+= all_possible(vocab_size, c)
    return l