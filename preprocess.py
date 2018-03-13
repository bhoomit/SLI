import os
from scripts import spectrograms

root = '.'
# file = open(os.path.join(root, 'data/trainingData.csv'), 'r')
# spectrograms.generate(enumerate(file.readlines()[1:]))

spectrograms.generate('~/bhoomit/SLI/data/')
