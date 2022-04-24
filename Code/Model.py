

class Layer:
	def __init__(self, size:int) -> None:
		self.size_ = size
	
	def Execute(weights:list[float]):
		pass

	def Size(self) -> int:
		return self.size_


class Model:

	def __init__(self) -> None:
		self.layers_ = []

	def AddLayer(self, layer:Layer) -> None:
		self.layers_.append(layer)

	def Execute(self, X:list[float], weights:list[float]) -> list[float]:
		[layer for layer in self.layers_]
		if True:
			pass