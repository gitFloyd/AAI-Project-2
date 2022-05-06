

class Log:
	log_ = {}

	@classmethod
	def Add(cls, label:str, text:str) -> None:
		if label not in cls.log_:
			cls.log_[label] = []
		cls.log_[label].append(text)
	
	@classmethod
	def PrintRange(cls, label:str, start:int = None, end:int = None) -> None:
		if label in cls.log_:
			for line in cls.log_[label][start:end]:
				print(line)

	@classmethod
	def Print(cls, label:str) -> None:
		cls.PrintRange(label)

	@classmethod
	def Clear(cls, label:str) -> None:
		if label in cls.log_:
			cls.log_[label].clear()

	@classmethod
	def Save(cls, label:str):
		with open('Logs\\log-{}.txt'.format(label), 'a+') as file:
			file.write("\n")
			if label in cls.log_:
				while len(cls.log_[label]) > 0:
					file.write(cls.log_[label].pop(0))

	@classmethod
	def SaveAll(cls):
		for key in cls.log_:
			cls.Save(key)
