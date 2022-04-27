

class Loss:
	"""A class containing various loss functions. All methods in this class
	are invoked directly via Loss.L2(...), etc. Do not need to instantiate
	these since there is no state.
	"""
	
	@staticmethod
	def L2(X:list, y:list) -> float:
		"""The Euclidean norm.

		Args:
			X (list): A vector of predictions.
			y (list): A vector of known values.

		Raises:
			Exception: When X and y do not have the same number of elements.

		Returns:
			float: The amount of loss.
		"""
		if len(X) != len(y):
			raise Exception("Loss.L2 requires X and y to be the same size.")

		return sum([(m - n) ** 2 for (m, n) in zip(X, y)]) ** 0.5

