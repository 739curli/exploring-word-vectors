
import numpy as np
import pickle
# Reload removal list
remove = pickle.load(open('removal_list','rb'))
removal_list = [i for i,v in enumerate(remove) if v==0]
del remove
# Load and remove extraneous word vectors
word_vectors = np.load('word_vectors.npy', mmap_mode='r')
word_vectors = np.delete(word_vectors,removal_list,axis=0)
# Split word vector list
middle = int(len(word_vectors)/2)
first_set_word_vectors = word_vectors[:middle]
pickle.dump(first_set_word_vectors,open('first_set_word_vectors','wb'))
del first_set_word_vectors
second_set_word_vectors = word_vectors[middle:]
pickle.dump(second_set_word_vectors,open('second_set_word_vectors','wb'))
del second_set_word_vectors
del word_vectors
