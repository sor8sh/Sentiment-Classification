# for using NLTKClassifier we need to have NLTK, NumPy and SciPy modules

import glob
from textblob.classifiers import \
    NaiveBayesClassifier, DecisionTreeClassifier, MaxEntClassifier, NLTKClassifier

# --------------------Train--------------------
train = []
path = './IMDBdataset/train/neg/*.txt'
for filename in glob.glob(path):
    with open(filename, 'r') as f:
        opinion = f.readlines()
        train.append((opinion[0], 'neg'))
path = './IMDBdataset/train/pos/*.txt'
for filename in glob.glob(path):
    with open(filename, 'r') as f:
        opinion = f.readlines()
        train.append((opinion[0], 'pos'))
print("Train set is ready!")

# ---------------------Test---------------------
test = []
path = './IMDBdataset/test/neg/*.txt'
for filename in glob.glob(path):
    with open(filename, 'r') as f:
        opinion = f.readlines()
        test.append((opinion[0], 'neg'))
path = './IMDBdataset/test/pos/*.txt'
for filename in glob.glob(path):
    with open(filename, 'r') as f:
        opinion = f.readlines()
        test.append((opinion[0], 'pos'))
print("Test set is ready!\n")

print("-----<< Classifiers >>-----")
print("1.Naive Bayes    2.Decision Tree")
print("3.MaxEnt         3.NLTK\n")
choice = input("Select one classifier number: ")

if choice == "1":
    print("\n" + "#NaiveBayesClassifier")
    cl1 = NaiveBayesClassifier(train)
    print("Classifier: Naive Bayes -- Accuracy: ", cl1.accuracy(test), "\n")

elif choice == "2":
    print("\n" + "#DecisionTreeClassifier")
    cl2 = DecisionTreeClassifier(train)
    print("Classifier: Decision Tree -- Accuracy: ", cl2.accuracy(test), "\n")

elif choice == "3":
    print("\n" + "#MaxEntClassifier")
    cl3 = MaxEntClassifier(train)
    print("Classifier: Maximum Entropy -- Accuracy: ", cl3.accuracy(test), "\n")

elif choice == "4":
    print("\n" + "#NLTKClassifier")
    cl4 = NLTKClassifier(train)
    print("Classifier: NLTK -- Accuracy: ", cl4.accuracy(test), "\n")

else:
    print("Bad input!")

# NB --> 0.8
# DT --> 0.634
