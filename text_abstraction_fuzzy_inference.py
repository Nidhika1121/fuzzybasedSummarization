

from __future__ import unicode_literals
import spacy,en_core_web_sm
from collections import Counter
import spacy
nlp = spacy.load('en_core_web_sm')
from spacy.matcher import Matcher
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl




matcher = Matcher(nlp.vocab)

def getMainNounChuck(inputSentence):
  lenChunk = 0
  prevLen = -1
  mainChunk = ""
  for chunk in inputSentence.noun_chunks:
       lenChunk =  len(chunk)
       print (chunk) 
       print(lenChunk)
       if prevLen < lenChunk:
         mainChunk = chunk
         prevLen = lenChunk

  print("Main chunk is:  ", mainChunk)
  return mainChunk


def getSentimentScoreOfChunk(doc):
    matches = matcher(doc)
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]
        span = doc[start:end]
    print("Sentiment", doc.sentiment * 10)
    return doc.sentiment * 10


def getSimilarity(doc1, doc2):
  return doc1.similarity(doc2)



def getPOSCOUNT(inputText, posTag):
  nlp = en_core_web_sm.load()

  count = 0 
  for token in nlp(inputText):
    count +=1 

  dictT= (Counter(([token.pos_ for token in nlp(inputText)])))
  print(dictT)
  return dictT[posTag]/(count+1) * 100


title = ('Financial London situations.') 
text = ('There is a developer beautiful and great conference happening on 21 July 2019 in London.')
doc = nlp(text)
print(getMainNounChuck(doc))
print("the similarity between sentences is", getSimilarity(nlp(title),getMainNounChuck(doc)))
getSentimentScoreOfChunk(doc)
print("Noun count" , getPOSCOUNT(text,"NOUN")) 
print("Verb count", getPOSCOUNT(text,"VERB"))
print("Adj count", getPOSCOUNT(text,"ADJ"))


#simWithNounChunk =  getSimilarity(doc,doc)
#sentimentScore = getSentimentScoreOfChunk(doc)
#nounCount=  getPOSCOUNT(text,"NOUN")
#verbCount =  getPOSCOUNT(text,"VERB")
#adjCount =  getPOSCOUNT(text,"ADJ")



###########################################
## fuzzy inference engine
###########################################




# Antecedent/Consequents functions
similarity_title = ctrl.Antecedent(np.arange(0, 1.25, .1), 'similarity_title')
sentiment_score = ctrl.Antecedent(np.arange(0,  1.25, .1), 'sentiment_score')
nounCount = ctrl.Antecedent(np.arange(0, 110, 10), 'nounCount')
verbCount  = ctrl.Antecedent(np.arange(0, 110, 10), 'verbCount')
adjCount  =  ctrl.Antecedent(np.arange(0, 110, 10), 'adjCount')
rank = ctrl.Consequent(np.arange(0, 24, 1), 'rank')


similarity_title['low'] = fuzz.trimf(similarity_title.universe, [0, 0.3, 0.5])
similarity_title['average'] = fuzz.trimf(similarity_title.universe, [0.3, 0.7, 1])
similarity_title['high'] = fuzz.trimf(similarity_title.universe, [0.6, 0.8, 21])

sentiment_score['low'] = fuzz.trimf(sentiment_score.universe, [0, 0.3, 0.5])
sentiment_score['average'] = fuzz.trimf(sentiment_score.universe, [0.3, 0.7, 1])
sentiment_score['high'] = fuzz.trimf(sentiment_score.universe, [0.6, 0.8, 21])


  
nounCount['low'] = fuzz.trimf(nounCount.universe, [0, 30, 50])
nounCount['average'] = fuzz.trimf(nounCount.universe, [30, 70, 100])
nounCount['high'] = fuzz.trimf(nounCount.universe, [60, 80, 102])

verbCount['low'] = fuzz.trimf(verbCount.universe, [0, 30, 50])
verbCount['average'] = fuzz.trimf(verbCount.universe, [30, 70, 100])
verbCount['high'] = fuzz.trimf(verbCount.universe, [60, 80, 102])


adjCount['low'] = fuzz.trimf(adjCount.universe, [0, 30, 50])
adjCount['average'] = fuzz.trimf(adjCount.universe, [30, 70, 100])
adjCount['high'] = fuzz.trimf(adjCount.universe, [60, 80, 102])


# membership functions rank
rank['low'] = fuzz.trimf(rank.universe, [0, 0, 10])
rank['average'] = fuzz.trimf(rank.universe, [5, 12, 17])
rank['high'] = fuzz.trimf(rank.universe, [10, 17, 21])


sentiment_score.view()
nounCount.view()
verbCount.view()
adjCount.view()
rank.view()


rule1 = ctrl.Rule(similarity_title['low'] | sentiment_score['low'], rank['low'])
rule2 = ctrl.Rule(sentiment_score['average'], rank['average'])
rule3 = ctrl.Rule(sentiment_score['average'] | quality['average'], rank['average'])
rule4 = ctrl.Rule(nounCount['low'] | verbCount['low'], rank['low'])
rule5 = ctrl.Rule(similarity_title['low']  | nounCount["high"] | verbCount["high"] | adjCount["high"] , rank['high'])
rule6 = ctrl.Rule( nounCount["high"] & sentiment_score['high'] & verbCount["high"] & adjCount["high"] , rank['high'])
rule7 = ctrl.Rule(similarity_title['high']   & sentiment_score['high'] & verbCount["high"] & adjCount["high"] , rank['high'])
rule8 = ctrl.Rule(similarity_title['high']  & nounCount["high"]  & verbCount["high"] & adjCount["high"] , rank['high'])
rule9 = ctrl.Rule(similarity_title['high']  & nounCount["high"] & sentiment_score['high']  & adjCount["high"] , rank['high'])
rule10 = ctrl.Rule(similarity_title['high']  & nounCount["high"] & sentiment_score['high'] & verbCount["high"]  , rank['high'])



rankFIS = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10])
rankFIS = ctrl.ControlSystemSimulation(rankFIS)
 
 
rankFIS.input['similarity_title'] =  getSimilarity(doc,doc)
rankFIS.input['sentiment_score'] = getSentimentScoreOfChunk(doc)
rankFIS.input['nounCount']  =  getPOSCOUNT(text,"NOUN")
rankFIS.input['verbCount']  =  getPOSCOUNT(text,"VERB")
rankFIS.input['adjCount']  =  getPOSCOUNT(text,"ADJ")

rankFIS.compute()
print ("the answer is") 
print (rankFIS.output['rank'])
rank.view(sim=rankFIS)
