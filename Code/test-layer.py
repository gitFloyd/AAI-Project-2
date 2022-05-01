import Model

Layer = Model.Layer

class ConovolutionLayer(Layer):

	# height x width
	ImageSize = (3, 5)
	WindowSize = (2, 2)
	Stride = 1
	
	def Execute(self, X:list[float], weights:list[float]) -> list[float]:
		
		image_data = []
		# i are rows 
		for i in range(ConovolutionLayer.ImageSize[0]):
			# inserting a blank row
			image_data.append([])
			# i are cols
			for j in range(ConovolutionLayer.ImageSize[1]):
				image_data[i].append(X[j + i * ConovolutionLayer.ImageSize[1]])

		# convolve!!!
		for a in _:
			for b in _:
				pass

		outputs = []
		# iterate the rows
		for row in image_data:
			# image_data, outputs, row
			# iterate a row's values
			for value in row:
				# image_data, outputs, row, value
				outputs.append(value)



		return outputs



# list of size 16 -> image size 4x4
# window 2x2, stride 1
# output size 3x3 -> list of 
foo = ConovolutionLayer(9, Layer.RELU)
print(foo.Execute([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], [0.1,0.5,-0.2,0.3]))