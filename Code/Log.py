

class Log:
	log_ = []

	@classmethod
	def Add(cls, text:str) -> None:
		cls.log_.append(text)
	
	@classmethod
	def PrintRange(cls, start:int = None, end:int = None) -> None:
		[print(line) for line in cls.log_[start:end]]

	@classmethod
	def Print(cls) -> None:
		cls.PrintRange()

	@classmethod
	def Clear(cls) -> None:
		cls.log_.clear()
