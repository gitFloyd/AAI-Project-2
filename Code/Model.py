from Activation import Activation as Act

class Layer:

	SOFTMAX = 1
	RELU = 2

	def __init__(self, size:int, activation:int = None) -> None:
		self.size_ = size
		if (activation == None or activation not in [Layer.SOFTMAX, Layer.RELU]):
			activation = Layer.RELU
		self.activation_ = activation
	
	def Execute(self, X:list[float], weights:list[float]) -> list[float]:
		return []

	def Size(self) -> int:
		return self.size_

	def NumWeights(self, numInputs:int) -> int:
		return numInputs
	
class DenseLayer(Layer):
	def Execute(self, X:list[float], weights:list[float]) -> list[float]:
		if self.activation_ == Layer.RELU: 
			return [
				Act.ReLU(sum([
					x * weights[i + j * (len(X) - 1)] for (j, x) in enumerate(X)
				])) for i in range(self.size_)
			]
		else:
			outputs = [
				sum([
					x * weights[i + j * (len(X) - 1)] for (j, x) in enumerate(X)
				]) for i in range(self.size_)
			]
			return [Act.Softmax(outputs, output) for output in outputs]

	def NumWeights(self, numInputs:int) -> int:
		return self.size_ * numInputs
	
class SparseLayer(Layer):
	def __init__(self, connections:list[tuple[int, list[int]]] = None, activation:int = None) -> None:
		nodes = set()
		self.connections_ = connections
		for connection in connections:
			nodes.union(connection[1])
		super.__init__(len(nodes), activation)
	
	def Execute(self, X:list[float], weights:list[float]) -> list[float]:

		values = [0] * self.size_
		for connection in self.connections_:
			for index in connection[1]:
				values[index] += X[connection[0]] * weights[index]

		if self.activation_ == Layer.RELU: 
			return [Act.ReLU(value) for value in values]
		else:
			return [Act.Softmax(values, value) for value in values]

	def NumWeights(self, numInputs:int) -> int:
		return self.size_ * numInputs

class Model:

	def __init__(self, layers:list[Layer] = []) -> None:
		self.layers_ = layers

	def AddLayer(self, layer:Layer) -> None:
		self.layers_.append(layer)

	def Execute(self, X:list[float], weights:list[float]) -> list[float]:
		start = 0
		end = 0
		results = X
		for layer in self.layers_:
			start = end
			end += layer.NumWeights(len(results))
			results = layer.Execute(results, weights[start:end])
		return results

	def NumWeights(self, size:int) -> int:
		count = 0
		for layer in self.layers_:
			count += layer.NumWeights(size)
			size = layer.Size()
		return count