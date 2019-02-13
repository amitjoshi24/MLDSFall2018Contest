import tensorflow as tf
print(tf.__version__)
import numpy as np
#from ApplicationEntry import ApplicationEntry
#from TensorflowApplicationEntry import TensorflowApplicationEntry
import traceback
import os, shutil
import csv
import sys
import random
csv.field_size_limit(sys.maxsize)

'''
author: Amit Joshi
email: amit.joshiusa@gmail.com
'''
class TensorflowModel():
	default_lr = 0.000005
	default_EPOCHS = 4000
	default_BATCH_SIZE = 20
	numHiddenLayerNodes = 50
	numInputLayerNodes = 26
	numOutputLayerNodes = 1
	default_split_ratio = 1.0
	default_rseed = 724

	default_standard_deviation = 0.1
	default_bias_initialization_constant = .1
	
	#default_layer_structure = [26, 20, 5, 40, 30, 50, 1]
	#default_layer_structure = [numInputLayerNodes, numHiddenLayerNodes, numHiddenLayerNodes, numHiddenLayerNodes, numHiddenLayerNodes, numOutputLayerNodes]
	#default_dropout_structure = [False, True, True, True, True, False]
	#default_layer_structure = [numInputLayerNodes, numHiddenLayerNodes*2, numHiddenLayerNodes, numOutputLayerNodes]
	
	#achieved 91.525 test accuracy with below structure
	#default_dropout_structure = [False, True, True, False]
	#default_layer_structure = [26, 20, 20, 1]
	
	#structure with extra features added on
	#default_dropout_structure = [False, False, False, False, False, False]
	#default_layer_structure = [2333, 1200, 500, 200, 100, 1]


	#default_dropout_structure = [False, False, False, False, False, False]
	#default_layer_structure = [332, 150, 75, 25, 10, 1]

	#default_dropout_structure = [False, False, False, False, False, False, False]
	#default_layer_structure = [951, 600, 200, 75, 40, 20, 1]

	#default_dropout_structure = [False, False, False, False, False, False, False]
	#default_layer_structure = [848, 1]

	#got 99.125
	#tfidf = TfidfVectorizer(min_df = 0.15, max_df = 0.95, ngram_range = (2, 4), sublinear_tf = False)
	#default_dropout_structure = [False, False, False, False, False, False, False]
	#default_layer_structure = [848, 500, 200, 100, 40, 20, 1]

	#got 99.125
	#tfidf = TfidfVectorizer(min_df = 0.1, max_df = 0.9, ngram_range = (2, 4), sublinear_tf = False)
	#default_dropout_structure = [False, False, False, False, False, False, False]
	#default_layer_structure = [1799, 1]

	#also got 99.125
	#tfidf = TfidfVectorizer(min_df = 0.3, max_df = 0.7, ngram_range = (1, 4), sublinear_tf = False)
	#default_dropout_structure = [False, False, False, False, False, False, False]
	#default_layer_structure = [1278, 1]

	#got 100%
	#default_dropout_structure = [False, False, False, False, False, False, False]
	#default_layer_structure = [995, 1]

	default_dropout_structure = [False, False, False, False, False, False, False]
	default_layer_structure = [1045, 1]

	#default_dropout_structure = [False, False, False, False, False, False, False]
	#default_layer_structure = [1799, 1200, 600, 200, 50, 1]

	#default_dropout_structure = [False, False, False, False, False, False, False]
	#default_layer_structure = [5208, 1]

	#default_dropout_structure = [False, False, False, False, False, False, False]
	#default_layer_structure = [848, 1]

	#default_dropout_structure = [False, False, False, False, False, False, False, False, False, False]
	#default_layer_structure = [848, 565, 379, 271, 167, 112, 74, 50, 30, 1]

	#default_dropout_structure = [False, False, False, False, False, False]
	#default_layer_structure = [848, 400, 150, 75, 30, 1]
	
	#default_layer_structure = [841, 300, 100, 30, 1]
	#default_dropout_structure = [False, False, False, False]
	
	#default_dropout_structure = [False, False, False, False]
	#default_layer_structure = [841, 300, 150, 50, 1]
	
	#default_dropout_structure = [False, False, False, False, False, False, False, False, False, False]
	#default_layer_structure = [26, 25, 25, 25, 25, 25, 25, 25, 25, 1]
	def __init__(self, totalEntries = None, modelName = None, csvFileName = None, layerStructure = default_layer_structure, dropoutStructure = default_dropout_structure, rseed = default_rseed, standard_deviation = default_standard_deviation, lr = default_lr, EPOCHS = default_EPOCHS, BATCH_SIZE = default_BATCH_SIZE, bias_initialization_constant = default_bias_initialization_constant, use_dropout = False):
		tf.reset_default_graph()
		
		self.totalEntries = totalEntries
		self.layers = layerStructure
		self.dropoutStructure = dropoutStructure
		self.rseed = rseed
		self.modelName = modelName
		self.bias_initialization_constant = bias_initialization_constant
		self.use_dropout = use_dropout
		self.standard_deviation = standard_deviation
		self.x = tf.placeholder(tf.float32, shape = [None, self.layers[0]], name = 'x')

		self.csvFileName = csvFileName
		#self.model = None
		self.restoredModelName = None
		self.y_true = tf.placeholder(tf.float32, [None, self.layers[len(self.layers)-1]], name = 'y_true')
		
		self.testing = 0
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
			#weightMat.append(tf.Variable(tf.zeros([layers[i], layers[i+1]]), dtype = tf.float32))
			self.biasMat.append(tf.Variable(tf.constant(self.bias_initialization_constant, shape = [self.layers[i+1]]), name = biasLayerName, dtype = tf.float32)) 
		

	def setModelName(self, newModelName):
		self.modelName = newModelName

	def loadDataToParse(self, applicationEntry):
	    dbname = applicationEntry.dbname
	    dbUserName = applicationEntry.dbUserName
	    dbPasswordName = applicationEntry.dbPasswordName
	    hostString = applicationEntry.hostString
	    portNum = applicationEntry.portNum
	    #dbUser = "trillion";
	    #dbPassword = "password";
	    #Read in the data from a csv file, converts into a dataFrame
	        #The nth row in the csv file is .loc[n] in this dataFrame
	    
	    
	    conn = psycopg2.connect(dbname=dbname, user=dbUserName, password=dbPasswordName, host = hostString, port=portNum)
	    
	    cur = conn.cursor()
	    
	    cur.execute("SELECT age, credit_amt, credit_history, dependants, duration, employment_partial_1, employment_partial_2, employment_partial_3, employment_time, checking_acct_status ,gender_marital_status ,debtors_guarantors_partial_1, debtors_guarantors_partial_2 ,has_telephone ,housing_partial_1, housing_partial_2, foreign_worker, fraudulent_flag, existing_credits, other_installments, property, purpose_partial_1, purpose_partial_2, residence_since, savings_acct_status FROM fraud_data;")
	    #cur.execute("SELECT age, credit_amt, credit_history, dependants, duration, employment_partial_1, employment_partial_2, employment_partial_3, employment_time, checking_acct_status ,gender_marital_status ,debtors_guarantors_partial_1, debtors_guarantors_partial_2 ,has_telephone ,housing_partial_1, housing_partial_2, foreign_worker, fraudulent_flag, existing_credits, other_installments, property, purpose_partial_1, purpose_partial_2, residence_since, savings_acct_status FROM creditdata;")
	    dataToParse = cur.fetchall()
	    return dataToParse


	def findCost(self, y_pred, y_true, testing):

		length = self.BATCH_SIZE
		if testing == 1:
			length = self.testingSize
		error = 0
		for i in range(length):
			error += (y_pred[i] - y_true[i]) ** 2
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


	def getFeaturesAndLabelsFromDatabase(self, applicationEntry):
		dataToParse = self.loadDataToParse(applicationEntry)
		resColumn = applicationEntry.resColumn
		labels = list()
		features = list()
		for i in range(len(dataToParse)):
			labels.append([dataToParse[i][resColumn]])
			feature = list()
			for j in range(len(dataToParse[i])):
				if j != resColumn:
					feature.append(dataToParse[i][j])
			features.append(feature)

		labels = np.asarray(labels)

		features = np.asarray(features)

		features, labels = self.unison_shuffled_copies(features, labels, self.rseed)
		return features, labels

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
				#print (row[1])
				#print ("------")
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
		
		'''startFile = open("started.txt", "w")
		startFile.write("I started")'''

		#df = pd.DataFrame({'features': [features], 'labels': [labels]})

		#np.random.shuffle(features)
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
		print (testingDataLength)

		#np.random.shuffle(labels)
		trainLabels, testLabels = labels[0:int(splitRatio*self.totalEntries)], labels[int(splitRatio*self.totalEntries):]

		trainingData = (trainFeatures, trainLabels)
		testingData = (testFeatures, testLabels)
		self.testingSize = len(testingData[1])

		trainingSize = len(trainingData[1])
		hidden_out = self.x
		for i in range(len(self.layers)-1):
			if i == len(self.layers)-2: #if it's the output layer, I'm using sigmoid activation
				hidden_out = tf.add(tf.matmul(hidden_out, self.weightMat[i]), self.biasMat[i])
				hidden_out = tf.nn.sigmoid(hidden_out)
			else:
				if self.dropoutStructure[i+1] == True and self.use_dropout == True:
					hidden_out = tf.add(tf.matmul(hidden_out, self.weightMat[i]), self.biasMat[i])
					#hidden_out = tf.nn.dropout(hidden_out, keep_prob = 0.5)
					hidden_out = tf.nn.relu(hidden_out)
				else:
					hidden_out = tf.add(tf.matmul(hidden_out, self.weightMat[i]), self.biasMat[i])
					#batch_mean, batch_var = tf.nn.moments(hidden_out, [0])
					#hidden_out = tf.nn.batch_normalization(hidden_out, mean = batch_mean, variance = batch_var, offset = 0, scale = 1, variance_epsilon = 1e-3)
					hidden_out = tf.nn.relu(hidden_out)

		y_pred = hidden_out


		#cross_entropy = tf.reduce_mean(tf.square(y_pred - self.y_true))
		#cross_entropy = y_pred - self.y_true
		cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_pred, labels = self.y_true))
		#l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.0007, scope=None)
		#l1_regularizer = tf.contrib.layers.l2_regularizer(scale = 0.001, scope=None)

		
		l1_regularizer = tf.contrib.layers.l1_l2_regularizer(scale_l1 = 0.00001, scale_l2 = 0.030, scope = None)
		weights = tf.trainable_variables() # all vars of your graph

		regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)


		gd_step = tf.train.AdamOptimizer(self.lr).minimize(cross_entropy + regularization_penalty)

		init = tf.global_variables_initializer()
		newypred = tf.round(y_pred)


		correct_mask = tf.equal(newypred, self.y_true)

		accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

		with tf.Session() as sess:
			ans = 0
			#f = open("output.txt", "w")
			#sess.run(init_op)
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
				realCounter = 0
				while totalCounter < trainingSize:

					batch_xs, batch_ys, counter = self.getBatch(trainingData, counter)
					counter += self.BATCH_SIZE
					totalCounter += self.BATCH_SIZE
					if i%1 == 0 and firstTrainingInEpoch == True:
						firstTrainingInEpoch = False
						self.testing = 0
						train_accuracy, ypred, ytrue, weights, hiddenout, crossentropy = sess.run([accuracy, y_pred, self.y_true, self.weightMat, hidden_out, cross_entropy], feed_dict = {self.x: batch_xs, self.y_true: batch_ys})
						print ("step {}, training accuracy {}, cross_entropy {}".format(i, train_accuracy, crossentropy))
					sess.run(gd_step, feed_dict = {self.x: batch_xs, self.y_true: batch_ys})
			
			self.saveModel(sess, finalModel = True)
			self.saveModel(sess, finalModel = False)
			print ("saved model: " + str(self.testing))
			self.testing = 1

			#print ("before testing accuracy: " + str(self.testing))
			acc, weights = sess.run([accuracy, self.weightMat], feed_dict={self.x: testingData[0], self.y_true: testingData[1]})
			
			weights = weights[0]

			#print (type(weights[0]))
			sortedWeights = list()
			for i in range(len(weights)):
				#sortedWeights.append((weights[i].tolist()[0], i))
				sortedWeights.append((i, weights[i].tolist()[0], i))

			sortedWeights.sort()
			for sw in sortedWeights:
				print ("column: " + str(sw[0] + 2) + " " + " weight: " + str(sw[1]))

			'''float maxWeight = -9999
			int maxIndex = -1
			for i in range(len(weights[0])):
				print (str(i) + ": " + str(weights[0][i]))
				if(weights[0][i] > maxWeight):
					maxWeight = weights[0][i]
					maxIndex = i

			print (maxIndex)
			print (maxWeight)'''

				#print (random.randint(1, 10))
			if acc > maxTestAccuracy:
						maxTestAccuracy = acc
						self.saveModel(sess, finalModel = False)
			
			#self.saveModel(sess, finalModel = True)
			print ("testing accuracy: " + str(acc))
			print ("max testing accuracy: " + str(maxTestAccuracy))
			print ("testing crossentropy: " + str(crossentropy))
			
			
			self.restoreModel() #saves time


	def saveModel(self, sess, input_tensor = None, output_tensor = None, finalModel = False):
		self.session = sess
		saver = tf.train.Saver()
		'''for i in range(len(self.layers)-1):
			weightLayerName = "weight" + str(i)
			biasLayerName = "bias" + str(i)
			tf.add_to_collection(weightLayerName, self.weightMat[0])
			tf.add_to_collection(biasLayerName, self.biasMat[0])'''

		saver.save(sess, self.modelName)
		if finalModel == True:
			saver.save(sess, "./finalModelFolder/" + self.modelName)
		else:
			saver.save(sess, "./bestModelFolder/" + self.modelName)
		#saveLocation = self.modelName[2:len(self.modelName)-5]

		#saveLocation = self.modelName[2:len(self.modelName)-5]


		'''inputsLunch = {'features': tf.saved_model.utils.build_tensor_info(input_tensor)}
		outputsLunch = {'labels': tf.saved_model.utils.build_tensor_info(output_tensor)}
		
		signature = tf.saved_model.signature_def_utils.build_signature_def(
			inputs=inputsLunch,
			outputs=outputsLunch,
			method_name='tensorflow/serving/predict')
		try:
			
			builder = tf.saved_model.builder.SavedModelBuilder(saveLocation)
			builder.add_meta_graph_and_variables(sess,[tf.saved_model.tag_constants.SERVING], signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})
			builder.save()
		except:
			folder = saveLocation
			try:
				if os.path.isfile(folder):
					os.unlink(folder)
				elif os.path.isdir(folder):
					shutil.rmtree(folder)
			except Exception as e:
				print ("ERROR SAVING")
			builder = tf.saved_model.builder.SavedModelBuilder(saveLocation)
			builder.add_meta_graph_and_variables(sess,[tf.saved_model.tag_constants.SERVING], signature_def_map={'serving_default': signature})
			builder.save()'''
		#print (saver.var_list)
		#self.model = sess




	def restoreModel(self, restoredModelName = None):
		print ("in restore model")
		if restoredModelName is None and self.restoredModelName is not None:
			return
		self.restoredModelName = restoredModelName
		if self.restoredModelName == None:
			self.restoredModelName = self.modelName

		print ("My restored model name is finally: " + self.restoredModelName)
		'''with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			restorer = tf.train.Saver()
			restorer.restore(sess, self.restoredModelName)
			self.session = sess'''
	
	def executeModel(self, features, outputFileName = None):
		'''if self.restoredModelName == None:
			self.restoredModelName = self.modelName'''

		'''if self.session == None:
			try:
				self.restoreModel(self.restoredModelName)
			except:
				print ("model can't be executed because it hasn't been trained or restored from a previous model")
		'''

		hidden_out = self.x
		for i in range(len(self.layers)-1):
			#weightLayerName = "w" + str(i)
			#biasLayerName = "b" + str(i)
			if i == len(self.layers)-2: #if it's the output layer, I'm using softmax activation
				hidden_out = tf.add(tf.matmul(hidden_out, self.weightMat[i]), self.biasMat[i])
				hidden_out = tf.nn.sigmoid(hidden_out)
				#hidden_out = tf.nn.sigmoid(hidden_out) + tf.constant(1.0, dtype = tf.float32)
				#hidden_out = tf.add(tf.matmul(hidden_out, weightMat[i]), 0)
				#hidden_out = tf.divide(hidden_out, tf.constant(10, dtype = tf.float32))
			else:
				hidden_out = tf.add(tf.matmul(hidden_out, self.weightMat[i]), self.biasMat[i])
				#hidden_out = tf.nn.sigmoid(hidden_out) + tf.constant(1.0, dtype = tf.float32)
				#batch_mean, batch_var = tf.nn.moments(hidden_out, [0])
				#hidden_out = tf.nn.batch_normalization(hidden_out, mean = batch_mean, variance = batch_var, offset = 0, scale = 1, variance_epsilon = 1e-3)
				hidden_out = tf.nn.relu(hidden_out)
				#hidden_out = tf.add(tf.matmul(hidden_out, weightMat[i]), 0)
				#hidden_out = tf.divide(hidden_out, tf.constant(10, dtype = tf.float32)

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
			predictionAndConfidenceArray = list()
			for i in range(len(newypred)):
				confidence = 1 - (2*abs(ypred[i][0] - newypred[i][0]))
				predictionAndConfidenceArray.append((newypred[i][0], ypred[i][0], confidence))

			f = open(outputFileName, 'w')
			f.write("Id,Expected\n")
			for i in range(len(predictionAndConfidenceArray)):
				print (predictionAndConfidenceArray[i])

				res = ""
				if(int(predictionAndConfidenceArray[i][0]) == 0):
					res = "prose"
				else:
					res = "poetry"
				if(i < len(predictionAndConfidenceArray) - 1):
					f.write(str(i) + "," + res + "\n")
				else:
					f.write(str(i) + "," + res)

			f.close()
			#print ("Labels: " + str(predictionAndConfidenceArray) + "\n")


model = TensorflowModel(modelName = "./tensorflowmodel.ckpt", csvFileName = "clean_extra_training_dataset.csv", use_dropout = False)
#applicationEntry = TensorflowApplicationEntry("creditdata", "postgres", "password", "localhost", 5433, 17)
#features, labels = model.getFeaturesAndLabelsFromDatabase(applicationEntry)
features, labels = model.getFeaturesAndLabelsFromCSV(1047);
model.trainModel(features, labels)

testFeatures = model.getFeaturesFromCSV(csvFileName = "clean_extra_test_dataset.csv")
model2 = TensorflowModel()
model2.restoreModel("./finalModelFolder/tensorflowmodel.ckpt")
model2.executeModel(testFeatures, outputFileName = "submission.csv")