# for using NLTKClassifier we need to have NLTK, NumPy and SciPy modules

import glob
import operator
from textblob import TextBlob
from textblob.classifiers import \
    NaiveBayesClassifier, DecisionTreeClassifier, MaxEntClassifier, NLTKClassifier

junk = ['a', 'an', 'the', 'of', 'br', 'by', 'as',
        'to', 'in', 'into', 'for', 'at', 'with', 'on', 'from',
        "'s", "'ve", 'has', 'have',
        'hi', 'who', '&', 'and', 'or',
        'my', 'me', 'he', 'she', 'him', 'her', 'his', 'its',
        'it', 'this', 'these', 'those', 'there', 'that', 'they',
        'i','be', 'am', 'is', 'are', 'was', 'were', 'you']

good = ['good', 'wonderful', 'best', 'better', 'love', 'like', 'positive', 'fine',
        'excellent', 'superb', 'great', 'awesome']

bad = ['bad', 'awful', 'worse', 'worst', 'hate', 'negative', 'poor', 'sorry', 'badly',
       'terrible', 'shit', 'sucks']

totalDictPos = {}
totalDictNeg = {}

def mostCommon(opinion):
    words = opinion.split(" ")
    tempDict = {}
    for word in words:
        if word in tempDict:
            tempDict[word] += 1
        else:
            tempDict[word] = 1
    tempDictSorted = sorted(tempDict.items(), key=operator.itemgetter(1))
    return tempDictSorted[-3:]

def preProcessingPos(l):
    blob = TextBlob(l[0])
    blob = blob.lower()
    words = blob.words              # Tokenizing
    newOpinion = ''
    for word in words:

        if word in good:
            newOpinion += 'good' + ' '
        elif word in bad:
            newOpinion += 'bad' + ' '
        elif word not in junk:        # Stemming
            if word[-1] == 's':
                word = word[0:-1]
            if word[-3:] == 'ing':
                word = word[0:-3]
            if word[-2:] == 'ly':
                word = word[0:-2]
            if word == "n't":
                word = 'not'
            if word[-2:] == 'ed':
                word = word[0:-2]
            newOpinion += word + ' '

    threeWords = mostCommon(newOpinion)
    for t in threeWords:
        if t[0] in totalDictPos:
            totalDictPos[t[0]] += t[1]
        else:
            totalDictPos[t[0]] = t[1]

    return newOpinion

def preProcessingNeg(l):
    blob = TextBlob(l[0])
    blob = blob.lower()
    words = blob.words              # Tokenizing
    newOpinion = ''
    for word in words:

        if word in good:
            newOpinion += 'good' + ' '
        elif word in bad:
            newOpinion += 'bad' + ' '
        elif word not in junk:        # Stemming
            if word[-1] == 's':
                word = word[0:-1]
            if word[-3:] == 'ing':
                word = word[0:-3]
            if word[-2:] == 'ly':
                word = word[0:-2]
            if word == "n't":
                word = 'not'
            if word[-2:] == 'ed':
                word = word[0:-2]
            newOpinion += word + ' '

    threeWords = mostCommon(newOpinion)
    for t in threeWords:
        if t[0] in totalDictNeg:
            totalDictNeg[t[0]] += t[1]
        else:
            totalDictNeg[t[0]] = t[1]

    return newOpinion

# --------------------Train--------------------
train = []
path = './IMDBdataset/train/neg/*.txt'
for filename in glob.glob(path):
    with open(filename, 'r') as f:
        opinion = f.readlines()
        newOpinion = preProcessingNeg(opinion)
        train.append((newOpinion, 'neg'))
path = './IMDBdataset/train/pos/*.txt'
for filename in glob.glob(path):
    with open(filename, 'r') as f:
        opinion = f.readlines()
        newOpinion = preProcessingPos(opinion)
        train.append((newOpinion, 'pos'))
print("Train set is ready!")

# ---------------------Test---------------------
test = []
path = './IMDBdataset/test/neg/*.txt'
for filename in glob.glob(path):
    with open(filename, 'r') as f:
        opinion = f.readlines()
        newOpinion = preProcessingNeg(opinion)
        test.append((newOpinion, 'neg'))
path = './IMDBdataset/test/pos/*.txt'
for filename in glob.glob(path):
    with open(filename, 'r') as f:
        opinion = f.readlines()
        newOpinion = preProcessingPos(opinion)
        test.append((newOpinion, 'pos'))
print("Test set is ready!\n")

print("-----<< Classifiers >>-----")
print("1.Naive Bayes    2.Decision Tree")
print("3.MaxEnt         3.NLTK\n")
choice = input("Select one classifier number: ")

# for testing with different dataset sizes
# size = input("n: ")
# trains = []
# for i in range(int(size)):
#     trains.append(train[i])
# for i in range(250, int(size)+250):
#     trains.append(train[i])

trains = train

if choice == "1":
    print("\n" + "#NaiveBayesClassifier")
    cl1 = NaiveBayesClassifier(trains)
    print("Classifier: Naive Bayes -- Accuracy: ", cl1.accuracy(test), "\n")

elif choice == "2":
    print("\n" + "#DecisionTreeClassifier")
    cl2 = DecisionTreeClassifier(trains)
    print("Classifier: Decision Tree -- Accuracy: ", cl2.accuracy(test), "\n")

elif choice == "3":
    print("\n" + "#MaxEntClassifier")
    cl3 = MaxEntClassifier(trains)
    print("Classifier: Maximum Entropy -- Accuracy: ", cl3.accuracy(test), "\n")

elif choice == "4":
    print("\n" + "#NLTKClassifier")
    cl4 = NLTKClassifier(trains)
    print("Classifier: NLTK -- Accuracy: ", cl4.accuracy(test), "\n")

else:
    print("Bad input!")

# most repeated words (most important properties)
totalDictPosSorted = sorted(totalDictPos.items(), key=operator.itemgetter(1))
totalDictNegSorted = sorted(totalDictNeg.items(), key=operator.itemgetter(1))