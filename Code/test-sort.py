import random
import math
data_ = [1,2,3,4]
data = []
for i in range(10):
	random.shuffle(data_)
	data.append(data_[:])


def LexOrder(item1, item2):

	num_fields = len(item1)
	for i in range(num_fields):
		if item1[i] != item2[i]:
			if item1[i] < item2[i]:
				return -1
			else:
				return 1
	return 0


def RowSort(rows):
	rows_len = len(rows)
	print('rows length: ', rows_len)
	if rows_len > 2:
		result1 = RowSort(rows[0: math.floor(rows_len * 0.5)])
		result2 = RowSort(rows[math.floor(rows_len * 0.5):])

		sorted_rows = []
		item1 = None
		item2 = None
		popped = 0
		saved = 0
		while len(result1) > 0 or len(result2) > 0:

			if len(result1) > 0 and len(result2) > 0 and item1 == None and item2 == None:
				item1 = result1.pop(0)
				item2 = result2.pop(0)
				popped += 2
			elif len(result1) > 0 and item1 == None:
				item1 = result1.pop(0)
				popped += 1
			elif len(result2) > 0 and item2 == None:
				item2 = result2.pop(0)
				popped += 1
			
			order = 0
			if item1 == None and item2 != None:
				order = 1
			elif item1 != None and item2 == None:
				order = -1
			else:
				order = LexOrder(item1, item2)
			
			if order == -1:
				sorted_rows.append(item1)
				saved += 1
				item1 = None
			elif order == 1:
				sorted_rows.append(item2)
				saved += 1
				item2 = None
			else:
				sorted_rows.append(item1)
				sorted_rows.append(item2)
				saved += 2
				item1 = None
				item2 = None

		if item1 != None:
			sorted_rows.append(item1)
			saved += 1
		if item2 != None:
			sorted_rows.append(item2)
			saved += 1

		print('popped: {}; saved: {}'.format(popped, saved))
		if popped != saved:
			raise Exception('foo')
		return sorted_rows

	elif rows_len == 1:
		return rows
	else:
		order = LexOrder(rows[0], rows[1])
		if order == 1:
			rows.reverse()
		return rows

print(RowSort(data))