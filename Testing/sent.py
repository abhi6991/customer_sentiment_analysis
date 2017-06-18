import nltk
import random
from nltk.corpus import state_union
from nltk.corpus import stopwords
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
import pickle
from nltk.classify import ClassifierI
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from scipy.stats import mode



class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
        

my_stopwords=['.',',','!','the']
negative_words=['no','not','nothing','any']
stop_words = set(stopwords.words('english'))

# print(stop_words)
for w in my_stopwords:
	stop_words.add(w)

pos_file=open("positive.txt","r").read()
neg_file=open("negative.txt","r").read()

pos_f=open("pos.txt","w")
neg_f=open("neg.txt","w")

documents=[]
flag=0
for line in pos_file.split('\n'):
	documents.append((line.lower(),"pos"))
	temp_list=word_tokenize(str(line))
	for w in temp_list:
		if(w in stop_words):
			continue
		w=w.lower()
		if flag==1:
			# print('w after neg='+w+'\n')
			w='not'+w
			flag=0
		if w in negative_words:
			flag=1		
		
		pos_f.write(str(w)+' ')
	pos_f.write('\n')	

flag=0
for line in neg_file.split('\n'):
	documents.append((line.lower(),"neg"))
	temp_list=word_tokenize(str(line))
	for w in temp_list:
		if(w in stop_words):
			continue
		w=w.lower()
		if flag==1:
			# print('w after neg='+w+'\n')
			w='not'+w
			flag=0

		if w in negative_words:
			# print('neg word is:'+w)
			flag=1
		neg_f.write(w+' ')
	neg_f.write('\n')	

pos_f.close()
neg_f.close()
#print(documents)	

pos_f=open("pos.txt","r").read()
neg_f=open("neg.txt","r").read()


all_words=[]

all_words_pos=word_tokenize(pos_file)
all_words_neg=word_tokenize(neg_file)

for w in all_words_pos:
	all_words.append(w.lower())

for w in all_words_neg:
	all_words.append(w.lower())	

all_words=nltk.FreqDist(all_words)
# print(all_words)	

train_text=state_union.raw("2005-GWBush.txt")
#print(train_text)

custom_sent_tokenizer=PunktSentenceTokenizer(train_text)

tokenized=custom_sent_tokenizer.tokenize(pos_f)

# print(tokenized)

tag_list=['JJ']
list_adjetives=[]
pos_features=open("pos_features.txt","w")
neg_features=open("neg_features.txt","w")

def process_content():
	try:
		for i in tokenized:
			words = nltk.word_tokenize(i)
			tagged = nltk.pos_tag(words)
			for tag in tagged:
				if tag[1] in tag_list:
					list_adjetives.append(tag[0])
					
	except Exception as e:
		print(str(e))
process_content()

set_adjective=set(list_adjetives)

for w in set_adjective:
	pos_features.write(w+'\n')

list_adjetives = nltk.FreqDist(list_adjetives)

word_features = list(list_adjetives.keys())[:250]
# print(word_features)

list_adjetives=[]

tokenized=custom_sent_tokenizer.tokenize(neg_f)

def process_content1():
	try:
		for i in tokenized:
			words = nltk.word_tokenize(i)
			tagged = nltk.pos_tag(words)
			for tag in tagged:
				if tag[1] in tag_list:
					list_adjetives.append(tag[0])
					
	except Exception as e:
		print(str(e))
process_content1()


set_adjective=set(list_adjetives)

# print(set_adjective)
for w in set_adjective:
	neg_features.write(w+'\n')

# print('\n\n')
list_adjetives = nltk.FreqDist(list_adjetives)

word_features2 = list(list_adjetives.keys())[:150]
# print(word_features)
for w in word_features2:
	word_features.append(w)

##Done with building the feature set

def find_features(document):
    words = word_tokenize(document)
    # print(words)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    # print("features::",features)    

    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

# print(featuresets)

random.shuffle(featuresets)

# print(featuresets)
print(len(featuresets))

training_set = featuresets[:300]

# set that we'll test against.
testing_set = featuresets[301:350]

classifier = nltk.NaiveBayesClassifier.train(training_set)
# print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(10)

save_classifier = open("pickle_algos/naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()


MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

save_classifier = open("pickle_algos/MNBclassifier.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

save_classifier = open("pickle_algos/Bernoulliclassifier.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

save_classifier = open("pickle_algos/logisticRegression.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

save_classifier = open("pickle_algos/SDGCclassifier.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

save_classifier = open("pickle_algos/SVCclassifier.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

save_classifier = open("pickle_algos/LinearSVCclassifier.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()



voted_classifier = VoteClassifier(
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)