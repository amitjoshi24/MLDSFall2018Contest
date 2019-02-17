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
import re
import string

csv.field_size_limit(sys.maxsize)
class NGramsFeatureExtractor():
	def __init__(self, csvFileName, csvFileName2):
		self.csvFileName = csvFileName;
		self.csvFileName2 = csvFileName2;

	def extractData(self, labelCol, csvFileName = None, outputFileName = None, outputFileName2 = None):
		labels = list()
		features = list()
		rawTexts = list()
		testRawTexts = list()
		testFeatures = list()

		printingRawTexts = list()
		testPrintingRawTexts = list()


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
				
				text = row[labelCol-1]
				text.replace('"', "")

				text = ''.join([i if ord(i) < 128 else '' for i in text])
				text.replace('"', '');
				text.replace('"', '');
				text.replace('\"', '');
				text.replace('\"', '');
				text = ''.join([i if i is not "\"" else '' for i in text])

				printingRawTexts.append(text)

				finalText = re.sub('\<.+\>', '', text)
				finalText.replace("\n", "")
				finalText.replace("\r", "")
				finalText.replace("\t", "")
				finalText = finalText.lower()
				
				rawTexts.append(finalText.translate(None, string.punctuation)) #happens to be the raw text column

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
				
				text = row[labelCol-1]
				text.replace('"', "")
				
				text = ''.join([i if ord(i) < 128 else '' for i in text])
				text.replace('"', '');
				text.replace('"', '');
				text.replace('\"', '');
				text.replace('\"', '');
				text = ''.join([i if i is not "\"" else '' for i in text])
				testPrintingRawTexts.append(text)

				finalText = re.sub('\<.+\>', '', text)
				finalText.replace("\n", "")
				finalText.replace("\r", "")
				finalText.replace("\t", "")
				finalText = finalText.lower()

				testRawTexts.append(finalText.translate(None, string.punctuation)) #happens to be the raw text column

				testFeatures.append(newFeature)	

		newFeatures = list()
		for feature in testFeatures:
			feature = [float(numeric_string) for numeric_string in feature]
			newFeatures.append(feature)

		testFeatures = newFeatures

		print (len(features))
		print (len(testFeatures))
		print (len(rawTexts))
		print (len(testRawTexts))
		print ("just printed length of both features and testFeatures")
		#now, it is time to add on to features

		tfidf = TfidfVectorizer(min_df = 0.1, max_df = 0.9, ngram_range = (2, 4), sublinear_tf = False)

		extraFeatures = tfidf.fit_transform(rawTexts)

		print (tfidf.get_feature_names())

		ngramFeatureNames = list()
		newFeatureNames = tfidf.get_feature_names()
		for i in range(len(newFeatureNames)):
			ngramFeatureNames.append("frequency: " + str(newFeatureNames[i]))
			#print (str(i) + ": " + ngramFeatureNames[i])

		densed = extraFeatures.todense()

		counter = 0
		for newFeature in densed:
			toAdd = np.array(newFeature)[0].tolist()
			if(counter < trainLength):
				print ("adding to train: " + str(counter))
				features[counter].extend(toAdd)
			else:
				print ("HORRIBLE ERROR")
			counter += 1

		extraTestFeatures = tfidf.transform(testRawTexts)

		testDensed = extraTestFeatures.todense()

		counter = 0
		for newFeature in testDensed:
			toAdd = np.array(newFeature)[0].tolist()
			print ("adding to test:")
			testFeatures[counter].extend(toAdd)
			counter += 1

		print (len(features[0]))


		labels = np.asarray(labels)

		features = np.asarray(features)

		testFeatures = np.asarray(testFeatures)
		print ("--------------------Features retrieved-----------------------")
		print(features)
		print ("--------------------Features Printed-----------------------")

		origFeaturesHeader = "ID,Personal Pronouns,Demonstrative Pronouns,Quidam,Reflexive Pronouns,Iste,Alius,Ipse,Idem,Priusquam,Antequam,Quominus,Dum,Quin,Ut,Conditionals,Prepositions,Interrogative Sentences,Superlatives,Atque + consonant,Relative Clauses,Mean Length Relative Clauses,Gerunds and Gerundives,Cum,Conjunctions,Vocatives,Mean Sentence Length,"
		totalFeaturesHeader = origFeaturesHeader
		for i in range(len(ngramFeatureNames)):
			if i < len(ngramFeatureNames) - 1:
				totalFeaturesHeader += ngramFeatureNames[i] + ","
			else:
				totalFeaturesHeader += ngramFeatureNames[i] + ","

		f = open(outputFileName, 'w')
		f.write(totalFeaturesHeader + "text,class\n")
		for i in range(len(features)):
			#print (i)
			f.write(str(i) + ",")
			for j in range(0, len(features[i])):
				f.write(str(features[i][j]) + ",")
			#f.write("fakeText,")
			#f.write(rawTexts[i] + ",")
			f.write("\"" + printingRawTexts[i] + "\"" + ",")
			if(labels[i][0] == 1):
				f.write("poetry")
			else:
				f.write("prose")
			f.write("\n")

		f = open(outputFileName2, 'w')
		f.write(totalFeaturesHeader + "text\n")
		for i in range(len(testFeatures)):
			#print (i)
			f.write(str(i) + ",")
			for j in range(0, len(testFeatures[i])):
				f.write(str(testFeatures[i][j]) + ",")
			#f.write("fakeText")
			#f.write(testRawTexts[i])
			f.write("\"" + testPrintingRawTexts[i] + "\"")
			f.write("\n")


extraFeatureExtractor = NGramsFeatureExtractor("training_dataset.csv", "test_dataset.csv");
extraFeatureExtractor.extractData(labelCol = 28, outputFileName = "extra_training_dataset.csv", outputFileName2 = "extra_test_dataset.csv")
