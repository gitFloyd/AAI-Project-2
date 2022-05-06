from io import TextIOWrapper
import math
from typing import TypeVar
import random
import os
from Settings import Settings

class Dataset:

	DataT = TypeVar('DataT')

	WIN_NL = "\r\n"
	LINUX_NL = "\n"

	def __init__(self, path:str, filename:str, newline:str = WIN_NL) -> None:
		self.path_ = path
		self.filename_ = filename
		self.loaded_ = False
		self.parsed_ = False
		self.data_ = None
		self.nl = newline
		self.classes_ = set()
		self.attributes_ = []
		self.types_ = []
		self.data_ = []

	def Data(self) -> list:
		return self.data_

	def Attributes(self) -> list:
		return self.attributes_

	def Types(self) -> list:
		return self.types_

	def Classes(self) -> list:
		return self.classes_

	def Load(self, reload:bool = False) -> DataT:
		if not self.loaded_ or reload:
			self.file_ = open(os.sep.join([self.path_, self.filename_]))
			self.loaded_ = True
		# If we reload, then we want to reparse as well.
		return self.Parse_(reload)

	def Parse_(self, reparse:bool = False) -> DataT:
		if not self.loaded_:
			# Silently return instead of raising an exception because
			# this method is not intended to be used outside of the
			# class. Although, it can be used that way if needed.
			return
		if not self.parsed_ or reparse:
			self.Parse_Hook_(self.file_.read())
		return self.data_

	def Parse_Hook_(self, data:str) -> None:
		self.data_ = data

	def __del__(self):
		if self.loaded_:
			self.file_.close()

class ArffRow:

	ATTR_LABEL = '@ATTRIBUTE ' # need the space at the end here
	DATA_LABEL = '@DATA'
	ATTR_LEN = len(ATTR_LABEL)
	DATA_LEN = len(DATA_LABEL)

	Attributes = []
	Types = []
	Data = []
	Classes = set()
	IsCollecting_ = False

	@classmethod
	def Reset(cls):
		cls.Attributes = []
		cls.Types = []
		cls.Data = []
		cls.Classes = set()
		cls.IsCollecting_ = False

	def __init__(self, line:str, nl:str) -> None:
		self.line_ = line
		self.len_ = len(line)
		self.nl_ = nl
	
	def Len(self) -> str:
		return self.len_

	def HasAttributeLabel(self) -> bool:
		return self.len_ >= ArffRow.ATTR_LEN and self.line_[0:ArffRow.ATTR_LEN] == ArffRow.ATTR_LABEL

	def HasDataLabel(self) -> bool:
		return self.len_ >= ArffRow.DATA_LEN and self.line_[0:ArffRow.DATA_LEN] == ArffRow.DATA_LABEL

	def GetAttributeData(self) -> tuple[str, str]:
		namePosition = 0
		for (i, char) in enumerate(self.line_[ArffRow.ATTR_LEN:]):
			if char == '\t':
				namePosition = i + ArffRow.ATTR_LEN
				break

		return (self.line_[ArffRow.ATTR_LEN:namePosition], self.line_[namePosition + 1:])

	def Parse(self):
		if ArffRow.IsCollecting_ and self.len_ > 1:
			ArffRow.Data.append(self.line_.split(','))
			ArffRow.Classes.add(ArffRow.Data[-1][-1])
		elif self.HasDataLabel():
			ArffRow.IsCollecting_ = True
		elif self.HasAttributeLabel():
			attrData = self.GetAttributeData()
			ArffRow.Attributes.append(attrData[0])
			ArffRow.Types.append(attrData[1])





class ArffDataset(Dataset):
	# ARFF (Attribute-Relation File Format)

	#def __init__(self, path:str, filename:str, newline:str = Dataset.WIN_NL) -> None:
	#	super().__init__(path, filename, newline)
	#
	#	self.parser_ = {
	#		'attributesLoaded': False,
	#	}

	def Parse_Hook_(self, data:str) -> None:
		ArffRow.Reset()
		rows = [ArffRow(line, self.nl) for line in data.split(self.nl)]

		for row in rows:
			row.Parse()

		for attribute in ArffRow.Attributes:
			self.attributes_.append(attribute)
		for typeName in ArffRow.Types:
			self.types_.append(typeName)
		for datum in ArffRow.Data:
			self.data_.append(datum)
		self.classes_ = self.classes_.union(ArffRow.Classes)

		classes = list(self.classes_)
		attribute_maxes = {}


		for row in self.data_:

			classIndex = classes.index(row[-1])
			row[-1] = [1 if i == classIndex else 0 for (i, value) in enumerate(classes)]

			for i in range(len(row)):
				if self.types_[i] == 'REAL':
					row[i] = float(row[i])
				elif self.types_[i] == 'INTEGER':
					row[i] = int(row[i])
				else:
					continue
				
				if i not in attribute_maxes:
					attribute_maxes[i] = 0
				if abs(row[i]) > attribute_maxes[i]:
					attribute_maxes[i] = row[i]

			for i in range(len(row)):
				if self.types_[i] == 'REAL' or  self.types_[i] == 'INTEGER':
					row[i] = row[i] / attribute_maxes[i]

		self.data_ = self.RowSort(self.data_)
	
	def LexOrder(self, item1, item2):

		num_fields = len(item1)
		for i in range(num_fields):
			if item1[i] != item2[i]:
				if item1[i] < item2[i]:
					return -1
				else:
					return 1
		return 0

	def RowSort(self, rows):
		rows_len = len(rows)
		if rows_len > 2:
			result1 = self.RowSort(rows[0: math.floor(rows_len * 0.5)])
			result2 = self.RowSort(rows[math.floor(rows_len * 0.5):])

			sorted_rows = []
			item1 = None
			item2 = None
			while len(result1) > 0 or len(result2) > 0:

				if len(result1) > 0 and len(result2) > 0 and item1 == None and item2 == None:
					item1 = result1.pop(0)
					item2 = result2.pop(0)
				elif len(result1) > 0 and item1 == None:
					item1 = result1.pop(0)
				elif len(result2) > 0 and item2 == None:
					item2 = result2.pop(0)
				
				order = 0
				if item1 == None and item2 != None:
					order = 1
				elif item1 != None and item2 == None:
					order = -1
				else:
					order = self.LexOrder(item1, item2)
				
				if order == -1:
					sorted_rows.append(item1)
					item1 = None
				elif order == 1:
					sorted_rows.append(item2)
					item2 = None
				else:
					sorted_rows.append(item1)
					sorted_rows.append(item2)
					item1 = None
					item2 = None

			if item1 != None:
				sorted_rows.append(item1)
			if item2 != None:
				sorted_rows.append(item2)

			return sorted_rows

		elif rows_len == 1:
			return rows
		else:
			order = self.LexOrder(rows[0], rows[1])
			if order == 1:
				rows.reverse()
			return rows

	def Fetch(self, *fields:list[str], limit:int = None, offset:int = 0):
		cols = []
		data = []

		# iterate over the field names and find the column indices
		# for names that match the requested field names
		for (i, field) in enumerate(fields):
			try:
				cols.append(self.attributes_.index(field))
			except ValueError:
				pass

		end = limit
		if limit != None:
			end += offset
		for row in self.data_[offset:end]:
			data.append([row[i] for i in cols])
			
		return data

	def FetchFilter_(self, i, value):
		# Not used any more

		#if self.types_[i] == 'REAL':
		#	return float(value)
		#elif self.types_[i] == 'INTEGER':
		#	return int(value)
		#else:
		#	return value
		pass


	def Size(self):
		length = len(self.data_)
		if length == 0:
			return (len(self.data_), None)
		return (len(self.data_), len(self.data_[0]))

	def Shuffle(self):
		random.shuffle(self.data_)



class Pistachio(ArffDataset):

	SettingsKey = 'PistachioDataset'

	def __init__(self, newline:str = Dataset.WIN_NL) -> None:
		settings = Settings.Data()
		super().__init__(
			path 		= settings[Pistachio.SettingsKey]['Path'], 
			filename 	= settings[Pistachio.SettingsKey]['FileName'],
			newline		= newline
		)

#pist = Pistachio(Dataset.LINUX_NL)
#
#for row in pist.Load()[0:10]:
#	print(row)

