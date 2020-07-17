from nltk import word_tokenize, pos_tag
import nltk
sentences = nltk.corpus.treebank.tagged_sents()
 
#print (sentences[0])
print ("Tagged sentences: ", len(sentences))
print ("Tagged words:", len(nltk.corpus.treebank.tagged_words()))
def features(t_sentence, index):
    """ t_sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': t_sentence[index],
        'is_first': index == 0,
        'is_last': index == len(t_sentence) - 1,
        'is_capitalized': t_sentence[index][0].upper() == t_sentence[index][0],
        'is_all_caps': t_sentence[index].upper() == t_sentence[index],
        'is_all_lower': t_sentence[index].lower() == t_sentence[index],
        'prefix-1': t_sentence[index][0],
        'prefix-2': t_sentence[index][:2],
        'prefix-3': t_sentence[index][:3],
        'suffix-1': t_sentence[index][-1],
        'suffix-2': t_sentence[index][-2:],
        'suffix-3': t_sentence[index][-3:],
        'prev_word': '' if index == 0 else t_sentence[index - 1],
        'next_word': '' if index == len(t_sentence) - 1 else t_sentence[index + 1],
        'has_hyphen': '-' in t_sentence[index],
        'is_numeric': t_sentence[index].isdigit(),
        'capitals_inside': t_sentence[index][1:].lower() != t_sentence[index][1:]
    }
 
import pprint 
#pprint.pprint(features(['This', 'is', 'a','trial', 'sentence'], 2))

def untag(sentences):
    return [w for w, t in sentences]
cutoff = int(.75 * len(sentences))
training_sentences = sentences[:cutoff]
test_sentences = sentences[cutoff:]
 
print ("training sentences:",len(training_sentences) )  # 2935
print ("test sentences:",len(test_sentences)  )       # 979
 
def transform_to_dataset(sentences):
    X, y = [], []
 
    for tagged in sentences:
        for index in range(len(tagged)):
            X.append(features(untag(tagged), index))
            y.append(tagged[index][1])
 
    return X, y
 
X, y = transform_to_dataset(training_sentences)
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
 
clf = Pipeline([
    ('vectorizer', DictVectorizer(sparse=False)),
    ('classifier', DecisionTreeClassifier(criterion='entropy'))
])
 
clf.fit(X[:10000], y[:10000])
 
print ('Training model completed')
 
X_test, y_test = transform_to_dataset(test_sentences)
 
print ("Accuracy of model:", clf.score(X_test, y_test))
 
def pos_tag(sentence):
    tags = clf.predict([features(sentence, index) for index in range(len(sentence))])
    d=dict(zip(sentence, tags))
    noun=[]
    pronoun=[]
    adjective=[]
    verb=[]
    adverb=[]
    preposition=[]
    conjuction=[]
    interjection=[]
    for i,j in d.items():
        if j=="NN" or j=="NNS" or j=="NNP" or j=="NNPS":
            noun.append(i)
        elif j=="PRP" or j=="PRP$" or j=="WP":
            pronoun.append(i)
        elif j=="JJ" or j=="JR" or j=="JS":
            adjective.append(i)
        elif j=="VB" or j=="VBG" or j=="VBD" or j=="VBN" or j=="VBP" or j=="VBZ":
            verb.append(i)
        elif j=="RB" or j=="RBS" or j=="RBR" or  j=="WRB":
            adverb.append(i)
        elif j=="IN":
            preposition.append(i)
        elif j=="CC":
            conjuction.append(i)
        elif j=="UH":
            interjection.append(i)
    print("NOUN:",noun)
    print("NOUN_COUNT",len(noun))
    print("PRONOUN:",pronoun)
    print("PRONOUN_COUNT",len(pronoun))
    print("ADJECTIVE",adjective)
    print("ADJECTIVE_count",len(adjective))
    print("VERB",verb)
    print("VERB_COUNT",len(verb))
    print("ADVERB",adverb)
    print("ADVERB_COUNT",len(adverb))
    print("PREPOSTION",preposition)
    print("PREPOSTION_count",len(preposition))
    print("CONJUCTION",conjuction)
    print("CONJUCTION_COUNT",len(conjuction))
    print("INTERJECTION",interjection)
    print("INTERJECTION_COUNT",len(interjection))
    return "TOTAL WORD COUNT :",len(d)
 
print(pos_tag(word_tokenize('The newly elected members of Rajya Sabha would be administered oath or affirmation on July 22, said persons aware of the details.M. Venkaiah Naidu, chairman, RS, has decided to proceed with the oath-taking ceremony keeping in view the resumption of meetings by the department-related parliamentary standing committees of both the RS and Lok Sabha and the interest expressed by the new members to participate in such meetings, Naidu’s office said in a statement.  Member of Parliament can participate in the meetings and other house proceedings only after being administered the oath or affirmation; even though they are eligible to draw salaries and other benefits.')))
