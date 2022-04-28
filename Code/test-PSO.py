
import Init
import PSO
from Log import Log
import Model

Layer = Model.Layer
DenseLayer = Model.DenseLayer
InputLayer = Model.InputLayer

#particles, weights
X = [1,2,3,4,5]
y = [1,0,0]

myModel = Model.Model([
	InputLayer(len(X)),
	DenseLayer(10, Layer.RELU),
	DenseLayer(3, Layer.SOFTMAX),
])


numParticles = 20
numWeights = myModel.NumWeights()
psoTest = PSO.PSO(myModel, (numParticles, myModel.NumWeights()))
result = psoTest.Execute(X, y, iterations=10)
# print(result)

# Log.Print()
print(myModel.Execute([5,4,3,2,1], result))