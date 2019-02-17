from gensim.models import Word2Vec

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
#from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string

csv.field_size_limit(sys.maxsize)

class Word2VecFeatureExtractor():
	def __init__(self, csvFileName, csvFileName2):
		self.csvFileName = csvFileName;
		self.csvFileName2 = csvFileName2;

	def extractData(self, labelCol, deeperDistinctionsTrainFileName, deeperDistinctionsTestFileName, csvFileName = None, outputFileName = None, outputFileName2 = None):
		labels = list()
		features = list()
		rawTexts = list()
		testRawTexts = list()
		testFeatures = list()

		origTrainFeaturesHeader = ""
		with open(self.csvFileName) as csvfile:
			reader = csv.reader(csvfile)
			first = True
			for row in reader:
				if(first):
					first = False
					origTrainFeaturesHeader = ",".join(row[0:len(row)-2])
					continue
				if len(row) <= 1:
					continue
				#features.append(row[0:labelCol-1])
				#features.append(row[0:labelCol-1])

				newFeature = row[1:labelCol-1]
				
				text = row[labelCol-1]

				finalText = re.sub('\<.+\>', '', text)
				
				finalText.replace("\n", "")
				finalText.replace("\r", "")
				finalText.replace("\t", "")
				#finalText = finalText.lower()
				#print ("after: " + finalText)
				#exit()
				rawTexts.append(finalText.translate(None, string.punctuation).split(" ")) #happens to be the raw text column

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

		
		origTestFeaturesHeader = ""
		with open(self.csvFileName2) as csvfile:
			reader = csv.reader(csvfile)
			first = True
			for row in reader:
				if(first):
					first = False
					origTestFeaturesHeader = ",".join(row[0:len(row)-1])
					continue
				if len(row) <= 1:
					continue
				#features.append(row[0:labelCol-1])
				#features.append(row[0:labelCol-1])

				newFeature = row[1:labelCol-1]
				
				text = row[labelCol-1]

				finalText = re.sub('\<.+\>', '', text)
				finalText.replace("\n", "")
				finalText.replace("\r", "")
				finalText.replace("\t", "")

				#finalText = finalText.lower()
				#print ("after2: " + finalText)
				testRawTexts.append(finalText.translate(None, string.punctuation).split(" ")) #happens to be the raw text column

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

		vectorSize = 203

		model = Word2Vec(rawTexts, size=vectorSize, window=5, min_count=2, sg = 1, workers=1)
		model.train(rawTexts, total_examples=len(rawTexts), epochs=15)
		origTrainFeaturesHeader += ","
		origTestFeaturesHeader += ","
		for i in range(vectorSize):
			one_hot = [0] * vectorSize
			one_hot[i] = 1
			one_hot_vector = np.asarray(one_hot)

			similarWord = str(model.similar_by_vector(one_hot_vector, topn = 1)[0][0]).lower()
			similarWord = similarWord.translate(None, string.punctuation)
			similarWord = similarWord.replace("\n", "")
			similarWord = similarWord.replace("\r", "")
			print(str(i) + " " + str(similarWord))
			#print (similarWord[0])
			#print (similarWord[0][0])
			#print ('--------')

			origTrainFeaturesHeader += "\"similarWord: " + similarWord + "\","
			origTestFeaturesHeader += "\"similarWord: " + similarWord + "\","

		happinessArray = list()
		sadnessArray = list()
		godArray = list()

		happyVector = model['felix']
		sadVector = model['tristis']
		godVector = model['deus']
		#print ("happy: " + str(model['felix']))
		#print ("sad: " + str(model['tristis']))
		#print ("god: " + str(model['deus']))
		
		extraFeatures = list()
		for text in rawTexts:
			print (len(text))
			numSummedWords = 0
			sumVector = np.zeros(shape=(vectorSize,))
			for word in text:
				try:
					sumVector = np.add(sumVector, model[word])
					numSummedWords += 1
				except:
					fakeVariable = 1
			sumVector = np.true_divide(sumVector, numSummedWords)
			
			happinessCosineSimilarity = np.dot(sumVector, happyVector)/(np.linalg.norm(sumVector)*np.linalg.norm(happyVector))
			happinessArray.append(happinessCosineSimilarity)

			sadnessCosineSimilarity = np.dot(sumVector, sadVector)/(np.linalg.norm(sumVector)*np.linalg.norm(sadVector))
			sadnessArray.append(sadnessCosineSimilarity)

			godCosineSimilarity = np.dot(sumVector, godVector)/(np.linalg.norm(sumVector)*np.linalg.norm(godVector))
			godArray.append(godCosineSimilarity)

			extraFeatures.append(sumVector.tolist())

		f = open(deeperDistinctionsTrainFileName, 'w')
		f.write("Id,Happiness,Sadness,Religiousness\n")
		for i in range(len(happinessArray)):
			f.write(str(i) + "," + str(happinessArray[i]) + "," + str(sadnessArray[i]) + "," + str(godArray[i]) + "\n");

		'''tfidf = TfidfVectorizer(min_df = 0.1, max_df = 0.9, ngram_range = (2, 4), sublinear_tf = False)

		extraFeatures = tfidf.fit_transform(rawTexts)

		densed = extraFeatures.todense()'''
		print ("features type: " + str(type(features)))
		print ("testFeatures type: " + str(type(testFeatures)))
		counter = 0
		for newFeature in extraFeatures:
			toAdd = np.array(newFeature).tolist()
			if(counter < trainLength):
				print ("adding to train: " + str(counter))
				#print (type(counter))
				#print (toAdd)
				features[counter].extend(toAdd)
			else:
				print ("HORRIBLE ERROR")
			counter += 1

		'''extraTestFeatures = tfidf.transform(testRawTexts)
		
		testDensed = extraTestFeatures.todense()'''

		testHappinessArray = list()
		testSadnessArray = list()
		testGodArray = list()

		extraTestFeatures = list()
		for text in testRawTexts:
			print (len(text))
			numSummedWords = 0
			sumVector = np.zeros(shape=(vectorSize,))
			for word in text:
				try:
					sumVector = np.add(sumVector, model[word])
					numSummedWords += 1
				except:
					fakeVariable = 1
			sumVector = np.true_divide(sumVector, numSummedWords)

			happinessCosineSimilarity = np.dot(sumVector, happyVector)/(np.linalg.norm(sumVector)*np.linalg.norm(happyVector))
			testHappinessArray.append(happinessCosineSimilarity)

			sadnessCosineSimilarity = np.dot(sumVector, sadVector)/(np.linalg.norm(sumVector)*np.linalg.norm(sadVector))
			testSadnessArray.append(sadnessCosineSimilarity)

			godCosineSimilarity = np.dot(sumVector, godVector)/(np.linalg.norm(sumVector)*np.linalg.norm(godVector))
			testGodArray.append(godCosineSimilarity)

			extraTestFeatures.append(sumVector.tolist())

		f = open(deeperDistinctionsTestFileName, 'w')
		f.write("Id,Happiness,Sadness,Religiousness\n")
		for i in range(len(testHappinessArray)):
			f.write(str(i) + "," + str(testHappinessArray[i]) + "," + str(testSadnessArray[i]) + "," + str(testGodArray[i]) + "\n");

		counter = 0
		for newFeature in extraTestFeatures:
			toAdd = np.array(newFeature).tolist()
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

		f = open(outputFileName, 'w')
		f.write(origTrainFeaturesHeader + "text,class\n")
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
		f.write(origTestFeaturesHeader + "text\n")
		for i in range(len(testFeatures)):
			#print (i)
			f.write(str(i) + ",")
			for j in range(0, len(testFeatures[i])):
				f.write(str(testFeatures[i][j]) + ",")
			f.write("fakeText")
			f.write("\n")


word2VecFeatureExtractor = Word2VecFeatureExtractor("extra_training_dataset.csv", "extra_test_dataset.csv");
word2VecFeatureExtractor.extractData(labelCol = 1799, deeperDistinctionsTrainFileName = "deeper_distinctions_train.txt", deeperDistinctionsTestFileName = "deeper_distinctions_test.txt", outputFileName = "extra_extra_training_dataset.csv", outputFileName2 = "extra_extra_test_dataset.csv")
