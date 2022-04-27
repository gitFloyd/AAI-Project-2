import math

class Activation:
	"""A class containing various activation functions. All methods in this
	class are invoked directly via Loss.L2(...), etc. Do not need to instantiate
	these since there is no state.

	Returns:
		_type_: _description_
	"""
	
	@staticmethod
	def ReLU(value:float) -> float:
		"""ReLU Activation Function

		Args:
			value (float): The value we are activating.

		Returns:
			float: The activated value.
		"""
		return max(0, value)
	
	@staticmethod
	def Softmax(values:list[float], index:int) -> float:
		"""Softmax Activation Function

		Args:
			values (list[float]): All the inputs to the output layer.
			index (int): The value from values we are activating.

		Returns:
			float: The activated value.
		"""
		return math.exp(values[index]) / sum([math.exp(value) for value in values])

