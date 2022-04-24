
import PSO
from Log import Log

def model1(X:list[float], weights:list[float]) -> list[float]:
	return [sum([x * w for w in weights]) for x in X]

#particles, weights
size = (20,10)
X = [1,2,3,4,5]
y = [3,8,1,9,4]
psoTest = PSO.PSO(model1, size)
result = psoTest.Execute(X, y, iterations=10)
print(model1(X, result))

Log.Print()

""" 
print(psoNum.Execute(31))
print(psoStr.Execute('blahblah '))
 """