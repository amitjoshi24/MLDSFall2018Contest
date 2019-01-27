import tensorflow as tf
print(tf.__version__)
import numpy as np
#from ApplicationEntry import ApplicationEntry
#from TensorflowApplicationEntry import TensorflowApplicationEntry
import traceback
import os, shutil
import csv
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

csv.field_size_limit(sys.maxsize)
class ExtraFeatureExtractor():
	def __init__(self, csvFileName, csvFileName2):
		self.csvFileName = csvFileName;
		self.csvFileName2 = csvFileName2;

	def extractData(self, labelCol, csvFileName = None, outputFileName = None, outputFileName2 = None):
		labels = list()
		features = list()
		rawTexts = list()
		testFeatures = list()


		with open(self.csvFileName) as csvfile:
			reader = csv.reader(csvfile)
			first = True
			for row in reader:
				if(first):
					first = False
					continue
				if len(row) <= 1:
					continue
				#features.append(row[0:labelCol-1])
				#features.append(row[0:labelCol-1])

				newFeature = row[1:labelCol-1]
				
				rawTexts.append(row[labelCol-1]) #happens to be the raw text column

				features.append(newFeature)

				if(row[labelCol] == "prose"):
					labels.append([0])
				else:
					labels.append([1])

		trainLength = len(labels)
		

		newFeatures = list()
		for feature in features:
			feature = [float(numeric_string) for numeric_string in feature]
			newFeatures.append(feature)

		features = newFeatures

		

		with open(self.csvFileName2) as csvfile:
			reader = csv.reader(csvfile)
			first = True
			for row in reader:
				if(first):
					first = False
					continue
				if len(row) <= 1:
					continue
				#features.append(row[0:labelCol-1])
				#features.append(row[0:labelCol-1])

				newFeature = row[1:labelCol-1]
				
				rawTexts.append(row[labelCol-1]) #happens to be the raw text column

				testFeatures.append(newFeature)	

		newFeatures = list()
		for feature in testFeatures:
			feature = [float(numeric_string) for numeric_string in feature]
			newFeatures.append(feature)

		testFeatures = newFeatures

		print (len(features))
		print (len(testFeatures))
		print (len(rawTexts))
		print ("just printed length of both features and testFeatures")
		#now, it is time to add on to features

		tfidf = TfidfVectorizer(min_df = 0.15, max_df = 0.85, ngram_range = (2, 3))

		extraFeatures = tfidf.fit_transform(rawTexts)

		densed = extraFeatures.todense()

		counter = 0
		for newFeature in densed:
			toAdd = np.array(newFeature)[0].tolist()
			if(counter < trainLength):
				print ("adding to train: " + str(counter))
				features[counter].extend(toAdd)
			else:
				print ("adding to test: " + str(counter))
				testFeatures[counter - trainLength].extend(toAdd)
			counter += 1
		print (len(features[0]))


		labels = np.asarray(labels)

		features = np.asarray(features)

		testFeatures = np.asarray(testFeatures)
		print ("--------------------Features retrieved-----------------------")
		print(features)
		print ("--------------------Features Printed-----------------------")

		f = open(outputFileName, 'w')
		f.write("ID,Personal Pronouns,Demonstrative Pronouns,Quidam,Reflexive Pronouns,Iste,Alius,Ipse,Idem,Priusquam,Antequam,Quominus,Dum,Quin,Ut,Conditionals,Prepositions,Interrogative Sentences,Superlatives,Atque + consonant,Relative Clauses,Mean Length Relative Clauses,Gerunds and Gerundives,Cum,Conjunctions,Vocatives,Mean Sentence Length,text,class\n")
		for i in range(len(features)):
			#print (i)
			f.write(str(i) + ",")
			for j in range(0, len(features[i])):
				f.write(str(features[i][j]) + ",")
			f.write("fakeText,")
			if(labels[i][0] == 1):
				f.write("poetry")
			else:
				f.write("prose")
			f.write("\n")

		f = open(outputFileName2, 'w')
		f.write("ID,Personal Pronouns,Demonstrative Pronouns,Quidam,Reflexive Pronouns,Iste,Alius,Ipse,Idem,Priusquam,Antequam,Quominus,Dum,Quin,Ut,Conditionals,Prepositions,Interrogative Sentences,Superlatives,Atque + consonant,Relative Clauses,Mean Length Relative Clauses,Gerunds and Gerundives,Cum,Conjunctions,Vocatives,Mean Sentence Length,text\n")
		for i in range(len(testFeatures)):
			#print (i)
			f.write(str(i) + ",")
			for j in range(0, len(testFeatures[i])):
				f.write(str(testFeatures[i][j]) + ",")
			f.write("fakeText")
			f.write("\n")


extraFeatureExtractor = ExtraFeatureExtractor("training_dataset.csv", "test_dataset.csv");
extraFeatureExtractor.extractData(labelCol = 28, outputFileName = "extra_training_dataset.csv", outputFileName2 = "extra_test_dataset.csv")

