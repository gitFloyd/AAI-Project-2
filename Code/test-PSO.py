import random
#random.seed(111)
import PSO
import Model
import Dataset as DS
from Log import Log
Dataset = DS.Dataset
Pistachio = DS.Pistachio

Layer = Model.Layer
DenseLayer = Model.DenseLayer
SparseLayer = Model.SparseLayer
FMLayer = Model.FuzzyMembershipLayer
InputLayer = Model.InputLayer

# particles, weights
# X = [1,2,3,4,5]
# y = [1,0,0]


pistachios = Pistachio(Dataset.LINUX_NL)
pistachios.Load()
pistachios.Shuffle()

offset = 0

data = pistachios.Fetch('Area', 'Solidity', 'Roundness', 'Compactness', 'Shapefactor_1', 'Class', limit = 100, offset = offset)
X = [row[0:-1] for row in data]
y = [row[-1] for row in data]

val_data = pistachios.Fetch('Area', 'Solidity', 'Roundness', 'Compactness', 'Shapefactor_1', 'Class', limit = 10, offset = offset + 100)
val_X = [row[0:-1] for row in val_data]
val_y = [row[-1] for row in val_data]

randConnections = SparseLayer.RandConnections(len(X[0]), len(X[0])*2, len(X[0]))
myModel = Model.Model([
	InputLayer(len(X[0])),
	#SparseLayer(randConnections, Layer.RELU),
	FMLayer(len(X[0]), len(X[0]) * 2, Layer.RELU),
	DenseLayer(16, Layer.RELU),
	DenseLayer(16, Layer.RELU),
	DenseLayer(2, Layer.SOFTMAX),
])


numInputs = 20


numParticles = 20
numWeights = myModel.NumWeights()
psoTest = PSO.PSO(myModel, (numParticles, myModel.NumWeights()))

records = psoTest.ExecuteMany(X[0:numInputs], y[0:numInputs], iterations=10)
# bestWeights = records[-1][1]
bestWeights = records.pop()[1]

for i in range(10):
	# print(records[i][1][0:10])
	pass


# Log.Print()
for i in range(len(val_X)):
	#print(val_X[i])
	print('Predict: {}; Actual: {}; val_X: {}'.format(myModel.Execute(val_X[i], bestWeights), val_y[i], val_X[i]))

print('---------------------')
for i in range(20):
	#print(val_X[i])
	print('Predict: {}'.format(myModel.Execute([random.randint(-1,0) + random.random() for _ in range(5)], bestWeights)))

Log.Add('bestWeights', '{}'.format(bestWeights))
Log.SaveAll()