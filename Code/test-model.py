import Model
import Dataset as DS
Dataset = DS.Dataset
Pistachio = DS.Pistachio


pistachios = Pistachio(Dataset.LINUX_NL)
pistachios.Load()
#pistachios.Shuffle()
data = pistachios.Fetch('Area', 'Solidity', 'Roundness', 'Compactness', 'Shapefactor_1', 'Class', limit = 10, offset = 0)
X = [value[0:-1] for value in data]
y = [value[-1] for value in data]


d_in = Model.InputLayer(5)
d1 = Model.DenseLayer(10)
d2 = Model.DenseLayer(10)
d3 = Model.DenseLayer(2, Model.Layer.SOFTMAX)
model = Model.Model([d_in, d1, d2, d3])


for row in pistachios.Data():
	print(row[-1])
exit()


predict = model.Execute(X[0])
print(pistachios.Classes())
print(X[0])
print(y[0])
print(predict)


