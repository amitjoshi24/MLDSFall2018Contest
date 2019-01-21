import tensorflow as tf
print(tf.__version__)
import numpy as np
#from ApplicationEntry import ApplicationEntry
#from TensorflowApplicationEntry import TensorflowApplicationEntry
import traceback
import os, shutil
import csv
import sys
csv.field_size_limit(sys.maxsize)

'''
author: Amit Joshi
email: amit.joshiusa@gmail.com
'''
class AutoEncoder():
	default_lr = 0.0005
	default_EPOCHS = 200000
	default_BATCH_SIZE = 1000
	numHiddenLayerNodes = 25
	numInputLayerNodes = 26
	numOutputLayerNodes = 1
	default_split_ratio = 1
	default_rseed = 724

	default_standard_deviation = .1
	default_bias_initialization_constant = 0.01
	
	default_layer_structure = [numInputLayerNodes, numHiddenLayerNodes, numHiddenLayerNodes, numInputLayerNodes]
	def __init__(self, totalEntries = None, modelName = None, csvFileName = None, layerStructure = default_layer_structure, rseed = default_rseed, standard_deviation = default_standard_deviation, lr = default_lr, EPOCHS = default_EPOCHS, BATCH_SIZE = default_BATCH_SIZE, bias_initialization_constant = default_bias_initialization_constant):
		tf.reset_default_graph()
		self.totalEntries = totalEntries
		self.layers = layerStructure
		self.rseed = rseed
		self.modelName = modelName
		self.bias_initialization_constant = bias_initialization_constant
		self.standard_deviation = standard_deviation
		self.x = tf.placeholder(tf.float32, shape = [None, self.layers[0]], name = 'x')

		self.csvFileName = csvFileName

		self.restoredModelName = None
		self.y_true = tf.placeholder(tf.float32, [None, self.layers[len(self.layers)-1]], name = 'y_true')

		self.lr = lr
		self.epochs = EPOCHS

		self.BATCH_SIZE = BATCH_SIZE
		self.weightMat = list()
		self.biasMat = list()

		self.session = None
		tf.set_random_seed(rseed)
		for i in range(len(self.layers)-1):
			weightLayerName = "weight" + str(i)
			biasLayerName = "bias" + str(i)
			self.weightMat.append(tf.Variable(tf.random_normal([self.layers[i], self.layers[i+1]], stddev = self.standard_deviation), name = weightLayerName, dtype = tf.float32))
			self.biasMat.append(tf.Variable(tf.constant(self.bias_initialization_constant, shape = [self.layers[i+1]]), name = biasLayerName, dtype = tf.float32)) 
		

	def setModelName(self, newModelName):
		self.modelName = newModelName



	def findCost(self, y_pred, y_true, testing):

		length = self.BATCH_SIZE
		if testing == 1:
			length = self.testingSize
		error = 0
		for i in range(length):
			print (i)
			x = abs((y_pred[i] - y_true[i])/y_true[i])
			error += ((100 * x) ** 1)
		print ("about to return: " + str(error))
		return error

	def unison_shuffled_copies(self, a, b, rseed):
	    assert len(a) == len(b)
	    np.random.seed(rseed)
	    p = np.random.permutation(len(a))
	    return a[p], b[p]

	def getBatch(self, data, counter):
		batch_xs = list()
		batch_ys = list()
		batchFeatures = data[0]
		batchLabels = data[1]
		dataTotalEntries = data[0].shape[0]
		counter = counter%dataTotalEntries
		index = 0
		for i in range(self.BATCH_SIZE):
			index = counter+i
			index = index%dataTotalEntries
			batch_xs.append(batchFeatures[index])
			batch_ys.append(batchLabels[index])

		batch_xs = np.asmatrix(batch_xs)
		batch_ys = np.asmatrix(batch_ys)
		return batch_xs, batch_ys, index

	def getFeaturesFromCSV(self, csvFileName = None):
		features = list()
		if csvFileName is None:
			csvFileName = self.csvFileName
		with open(csvFileName) as csvfile:
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
				features.append(row[1:len(row)-1])
		newFeatures = list()
		for feature in features:
			feature = [float(numeric_string) for numeric_string in feature]
			newFeatures.append(feature)

		features = newFeatures

		features = np.asarray(features)

		return features


	def getFeaturesAndLabelsFromCSV(self, labelCol, csvFileName = None):
		labels = list()
		features = list()
		if csvFileName is None:
			csvFileName = self.csvFileName

		with open(csvFileName) as csvfile:
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

				features.append(row[1:labelCol-1])
				if(row[labelCol] == "prose"):
					labels.append([0])
				else:
					labels.append([1])

		newFeatures = list()
		for feature in features:
			feature = [float(numeric_string) for numeric_string in feature]
			newFeatures.append(feature)

		features = newFeatures

		labels = np.asarray(labels)

		features = np.asarray(features)
		print ("--------------------Features retrieved-----------------------")
		print(features)
		print ("--------------------Features Printed-----------------------")

		features, labels = self.unison_shuffled_copies(features, labels, self.rseed)
		return features, labels

	def _get_beta_accumulators(self):
		return self._beta1_power, self._beta2_power

	def trainModel(self, features, labels, splitRatio = default_split_ratio, epochs = None, retrain = False, restoredModelName = None):
		if retrain == True:
			print ("about to go to restoredModel")
			self.restoreModel(restoredModelName)


		if epochs is not None:
			self.epochs = epochs

		self.totalEntries = len(labels)
		print ("Total entries is " + str(self.totalEntries))
		print (splitRatio)
		trainFeatures, testFeatures = features[0:int(splitRatio*self.totalEntries)], features[int(splitRatio*self.totalEntries):]

		testingDataLength = len(testFeatures)

		trainLabels, testLabels = labels[0:int(splitRatio*self.totalEntries)], labels[int(splitRatio*self.totalEntries):]

		trainingData = (trainFeatures, trainLabels)
		testingData = (testFeatures, testLabels)
		self.testingSize = len(testingData[1])

		trainingSize = len(trainingData[1])
		hidden_out = self.x
		for i in range(len(self.layers)-1):
			hidden_out = tf.add(tf.matmul(hidden_out, self.weightMat[i]), self.biasMat[i])
			#hidden_out = tf.nn.relu(hidden_out)
		y_pred = hidden_out


		#cross_entropy = tf.reduce_mean(self.findCost(y_pred, self.y_true, 0))
		#cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_pred, labels = self.y_true))
		cross_entropy = tf.reduce_sum(tf.square(y_pred - self.y_true))
		gd_step = tf.train.AdamOptimizer(self.lr).minimize(cross_entropy)

		init = tf.global_variables_initializer()
		newypred = tf.round(y_pred)

		with tf.Session() as sess:
			ans = 0
			sess.run(init)
			sess.run(tf.local_variables_initializer())
			if retrain == True:
				restorer = tf.train.Saver()
				restorer.restore(sess, self.restoredModelName)
			
			
			sufficientCounter = 0
			maxTestAccuracy = 0
			for i in range(self.epochs+1):
				totalCounter = 0
				counter = 0
				firstTrainingInEpoch = True
				trainFeatures = trainingData[0]
				trainLabels = trainingData[1]
				trainFeatures, trainLabels = self.unison_shuffled_copies(trainFeatures, trainLabels, self.rseed)
				trainingData = (trainFeatures, trainLabels)
				while totalCounter <= trainingSize:
					batch_xs, batch_ys, counter = self.getBatch(trainingData, counter)
					counter += self.BATCH_SIZE
					totalCounter += self.BATCH_SIZE
					if i%1 == 0 and firstTrainingInEpoch == True:
						firstTrainingInEpoch = False
						self.testing = 0
						ypred, ytrue, weights, hiddenout, crossentropy = sess.run([y_pred, self.y_true, self.weightMat, hidden_out, cross_entropy], feed_dict = {self.x: batch_xs, self.y_true: batch_xs})
						print ("step {}, cross_entropy {}".format(i, crossentropy))
					sess.run(gd_step, feed_dict = {self.x: batch_xs, self.y_true: batch_xs})
			
			self.saveModel(sess, finalModel = True)
			print ("saved model: " + str(self.testing))
			self.saveModel(sess, finalModel = False)
		
			self.restoreModel() #saves time


	def saveModel(self, sess, input_tensor = None, output_tensor = None, finalModel = False):
		self.session = sess
		saver = tf.train.Saver()

		saver.save(sess, self.modelName)
		if finalModel == True:
			saver.save(sess, "./finalAutoEncoderFolder/" + self.modelName)
		else:
			saver.save(sess, "./bestAutoEncoderFolder/" + self.modelName)




	def restoreModel(self, restoredModelName = None):
		print ("in restore model")
		if restoredModelName is None and self.restoredModelName is not None:
			return
		self.restoredModelName = restoredModelName
		if self.restoredModelName == None:
			self.restoredModelName = self.modelName

		print ("My restored model name is finally: " + self.restoredModelName)
	
	def executeModel(self, features, labels, outputFileName = None):

		hidden_out = self.x
		for i in range(len(self.layers)-1):
			hidden_out = tf.add(tf.matmul(hidden_out, self.weightMat[i]), self.biasMat[i])
			hidden_out = tf.nn.relu(hidden_out)

		y_pred = hidden_out
		newy_pred = tf.round(y_pred)

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			restorer = tf.train.Saver()
			print("In execute, " + self.restoredModelName)
			restorer.restore(sess, self.restoredModelName)
			#sess.run(tf.local_variables_initializer())
			newypred, ypred = sess.run([newy_pred, y_pred], feed_dict = {self.x: features})
			print ("Test cases: " + str(features) + "\n")
			decodedFeatures = list()
			for i in range(len(newypred)):
				decodedFeatures.append(ypred[i])

			f = open(outputFileName, 'w')
			f.write("ID,Personal Pronouns,Demonstrative Pronouns,Quidam,Reflexive Pronouns,Iste,Alius,Ipse,Idem,Priusquam,Antequam,Quominus,Dum,Quin,Ut,Conditionals,Prepositions,Interrogative Sentences,Superlatives,Atque + consonant,Relative Clauses,Mean Length Relative Clauses,Gerunds and Gerundives,Cum,Conjunctions,Vocatives,Mean Sentence Length,text,class\n")
			for i in range(len(decodedFeatures)):
				print (i)
				f.write(str(i) + ",")
				for j in range(0, len(decodedFeatures[i])):
					f.write(str(decodedFeatures[i][j]) + ",")
				f.write("fakeText,")
				if(labels[i][0] == 1):
					f.write("poetry")
				else:
					f.write("prose")
				f.write("\n")

			f.close()

autoencoder = AutoEncoder(modelName = "./autoencoder.ckpt", csvFileName = "clean_training_dataset.csv")
#applicationEntry = TensorflowApplicationEntry("creditdata", "postgres", "password", "localhost", 5433, 17)
#features, labels = model.getFeaturesAndLabelsFromDatabase(applicationEntry)
features, labels = autoencoder.getFeaturesAndLabelsFromCSV(28);
autoencoder.trainModel(features, labels)

autoencoder.executeModel(features, labels, "decoded_training_dataset.csv")

