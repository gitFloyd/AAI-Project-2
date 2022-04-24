

class Loss:
	
	@staticmethod
	def L2(X:list, y:list) -> float:
		if len(X) != len(y):
			raise Exception("Loss.L2 requires X and y to be the same size.")

		return sum([(m - n) ** 2 for (m, n) in zip(X, y)]) ** 0.5

