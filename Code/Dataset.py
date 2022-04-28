import Init

class Dataset:
	def __init__(self, path, filename) -> None:
		self.Path = path
		self.FileName = filename
		self.Loaded = False

	def Load(self):
		self.File = open(self.Path + self.FileName)
		self.Loaded = True
		return self.Parse(self.File)

	def Parse(self, file):
		if not self.Loaded:
			raise Exception('The Dataset must be Loaded before invoking Parse.')
		return file.read()

	def __del__(self):
		if self.Loaded:
			self.File.close()