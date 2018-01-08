import sys
import re
import math
import csv
import json
import nltk
from abc import ABCMeta, abstractmethod
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from collections import Counter

'''
	Base class for the text classifier
'''
class Classifier(metaclass=ABCMeta):
	'''
		Initialize the classifier with training data
	'''
	def __init__(self, filename) :
		# Read the training file
		try :
			iFile = open(filename, 'r')
		except:
			print('Cannot read training dataset ' + filename + '. Make sure the file is readable', file=sys.stderr)
			sys.exit(1)

		# Initialize data structures
		self.words = set()				# Words Vocabulary
		self.bagWords = dict()			# Bag of words
		self.countClasses = dict()		# Occurrence of each class
		self.totalClasses = 0			# Total number of classes
		self.countWords = dict()		# Count of each word in a class
		self.totalWords = dict()		# Count of total words in a class
		self.threshold = 2				# Word frequency threshold
		
		# Train the model
		self.train(iFile)

		iFile.close()

	'''
		Preprocess tokens for adding feature to the classifier
	'''
	@abstractmethod
	def preprocess(self, words) :
		pass

	'''
		Train the classifier with training dataset
	'''
	def train(self, iFile) :
		for line in iFile :
			# Remove all special characters except apostrophe
			line = re.sub(r"[^a-z0-9']", " ", line.strip())

			# If line is empty
			if not line :
				continue

			# Split words
			words = line.split()

			# Initialize class
			_class = words[0]
			# Remove class from word list
			words = words[1:]
			# Preprocess as per the enhancement requirement
			words = self.preprocess(words)

			# Initialize class
			if _class not in self.countClasses :
				self.countClasses[_class] = 0
				self.countWords[_class] = dict()
				self.totalWords[_class] = 0

			self.countClasses[_class] += 1
			self.totalClasses += 1
			self.words.update(words)
			self.totalWords[_class] += len(words)
			for word in words:
				# Add word to the bag of words
				if word not in self.bagWords :
					self.bagWords[word] = 1
				else :
					self.bagWords[word] += 1

				# Add word frequency
				if word not in self.countWords[_class] :
					self.countWords[_class][word] = 1
				else :
					self.countWords[_class][word] += 1

		# Identify words with frequency lower than threshold
		wordsToRemove = list()
		for word in self.bagWords :
			if self.bagWords[word] < self.threshold :
				wordsToRemove.append(word)
				# Remove word from vocabulary
				self.words.remove(word)

		# Remove the count of words to be removed
		for _class in self.countWords :
			if word in self.countWords[_class] :
				self.totalWords[_class] -= self.countWords[_class][word]
				self.countWords[_class].pop(word)

	'''
		Perform classification on the given document
	'''
	def classify(self, document) :
		# Preprocess the given words as per the enhancement
		document = self.preprocess(document)

		# Calculate probability for each class
		probability = -math.inf
		computedClass = None
		for _class in self.countClasses :
			# Calculate class probability
			classProbability = math.log(self.countClasses[_class] / self.totalClasses)

			# Calculate probability of words
			for word in document :
				if word in self.words :
					wordCount = self.countWords[_class][word] if word in self.countWords[_class] else 0
					classProbability += math.log(wordCount + 1) - math.log(self.totalWords[_class] + len(self.words))

			# If the probabilty of current class is greater than previous class
			if classProbability > probability :
				probability = classProbability
				computedClass = _class

		return computedClass

	'''
		Return 20 representative words of each class from training data
	'''
	def representative_words(self) :
		rwords = dict()
		# Identify 20 most common words
		for _class in self.countWords: 
			rwords[_class] = dict(Counter(self.countWords[_class]).most_common(20))
		return rwords

'''
	Normal Naive Bayes text classifier with no feature enhancement
'''
class NBClassifier(Classifier):
	def __init__(self, filename) :
		self.name = 'Multinomial Naive Bayes Classifier'
		super().__init__(filename)

	'''
		Preprocess words. In Naive Bayes, no additional features are added
	'''
	def preprocess(self, words) :
		return words

'''
	Naive Bayes text classifier with stop words removed
'''
class NBClassifierSW(Classifier) :
	def __init__(self, filename) :
		self.name = 'Multinomial Naive Bayes w/o stop-words'
		self.stopwords = stopwords.words('english')
		super().__init__(filename)

	'''
		Remove stop words from the set of words
	'''
	def preprocess(self, words) :
		return [word for word in words if word not in self.stopwords]

'''
	Naive Bayes text classifier with Stemmer
'''
class StemNBClassifier(Classifier) :
	def __init__(self, filename) :
		self.name = 'Multinomial Naive Bayes with stems'
		self.stemmer = SnowballStemmer('english', ignore_stopwords=True)
		super().__init__(filename)

	'''
		Stem each word with Snowball stemmer
	'''
	def preprocess(self, words) :
		return [self.stemmer.stem(word) for word in words]

'''
	Naive Bayes text classifier with Parts of Speech Tagging
'''
class POSNBClassifier(Classifier) :
	def __init__(self, filename) :
		self.name = 'Multinomial Naive Bayes with POS'
		super().__init__(filename)

	'''
		Add parts of speech to every word
	'''
	def preprocess(self, words) :
		a = nltk.pos_tag(words)
		return nltk.pos_tag(words)

	'''
		Return 20 representative words of each class from training data
		Remove parts of speech from the word
	'''
	def representative_words(self) :
		tempRWords = super().representative_words()
		rwords = dict()
		for _class in tempRWords :
			rwords[_class] = dict()
			for word in tempRWords[_class] :
				rwords[_class][word[0]] = tempRWords[_class][word]
		return rwords


'''
	Execute classification on the given file
'''
def execute_classification(classifier, filename) :
	# Read file
	try :
		iFile = open(filename, 'r')
	except :
		print('Could not open file ' + filename + '. Make sure it exists and you have read permissions', file=sys.stderr)
		sys.exit(1)

	# Initialize the counts
	totalCount = 0
	correctCount = 0

	for line in iFile :
		# Strip line and remove punctuations except apostrophe
		line = re.sub(r"[^a-z0-9']", " ", line.strip())
		if not line :
			continue

		# Split line into words
		words = line.split()

		# Extract given class for the document
		givenClass = words[0]

		# Compute class of remaining words
		computedClass = classifier.classify(words[1:])

		# Set counts for accuracy calculation
		if computedClass == givenClass :
			correctCount += 1
		totalCount += 1

	# Calculate accuracy
	accuracy = correctCount * 100 / totalCount

	# Display output
	print('+----------------------------------------------------+')
	print('| %-50s |' % (classifier.name + ' (' + filename + ')'))
	print('+----------------------------------------------------+')
	print('| No. Documents: %4d     Correctly Classified: %4d |' % (totalCount, correctCount))
	print('| Accuracy: %39.2f%% |' % accuracy)
	print('+----------------------------------------------------+')

	iFile.close()

'''
	Generate JSON file for 20 most representative words
	as per the current classifier
'''
def write_representative_words(classifier, filename) :
	# Open file for writing
	try :
		oFile = open(filename, 'w')
	except :
		print('Could not open file ' + filename + ' for writing. Make sure you have proper permissions', file=sys.stderr)
		sys.exit(1)

	# Get representative words from the classifier
	rwords = classifier.representative_words()

	# Write JSON file
	oFile.write(json.dumps(rwords))
	oFile.close()

'''
	Solution 1: Naive Bayes Classification
'''
# Initialize and train classifier
classifier = NBClassifier('data/train')
# Classify dev set
execute_classification(classifier, 'data/dev')
# Classify test set
execute_classification(classifier, 'data/test')
# Write representative words
write_representative_words(classifier, 'mnbc.json')

'''
	Solution 2.a: Naive Bayes Classifier without Stop Words
'''
# Initialize and train classifier
classifier = NBClassifierSW('data/train')
# Classify dev set
execute_classification(classifier, 'data/dev')
# Classify test set
execute_classification(classifier, 'data/test')
# Write representative words
write_representative_words(classifier, 'mnbc_wo_stop.json')

'''
	Solution 2.b: Naive Bayes Classifier with Stems
'''
# Initialize and train classifier
classifier = StemNBClassifier('data/train')
# Classify dev set
execute_classification(classifier, 'data/dev')
# Classify test set
execute_classification(classifier, 'data/test')

'''
	Solution 2.b: Naive Bayes Classifier with POS Tagging
'''
# Initialize and train classifier
classifier = POSNBClassifier('data/train')
# Classify dev set
execute_classification(classifier, 'data/dev')
# Classify test set
execute_classification(classifier, 'data/test')