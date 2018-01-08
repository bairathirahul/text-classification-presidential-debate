# Assignment 4 - Text Classification

The text classification is implemented by a Python program rbairath.py. The program is executable using Python 3 only. The program also needs "nltk" package for successful execution.

To execute, on the command prompt / terminal, write:

#### Windows:
`py rbairath.py`

#### Mac:
`python3 rbairath.py`

The program executes text classification using 4 classification models, which are as follows:
* Naive Bayes
* Naive Bayes without Stopwords
* Naive Bayes with Stems
* Naive Bayes with Parts of Speech tagging

The output of the program is printed on the console and contains accuracy of these models for both dev and test data set.

## Solution 3:
The program also generates two JSON file "mnbc_wo_stopwords.json" and "mnbc.json", which contains representative words and their frequency with respect to every speaker in the JSON format. This file is imported by R markdown script rbairath.RMD to generate word-cloud representation of 20 most represntative words of each speaker, as required by the Problem 3.

The R markdown script dependsd on "rjson" and "wordcloud" package, which must be installed before its execution. 

The output of R markdown is contained in "rbairath.html" file. This HTML file lists word cloud for each of the 13 speakers (both excluding and including stop-words).