from io import TextIOWrapper
from typing import TypeVar
import Settings

class Dataset:

	DataT = TypeVar('DataT')

	def __init__(self, path:str, filename:str) -> None:
		self.path_ = path
		self.filename_ = filename
		self.loaded_ = False
		self.parsed_ = False
		self.data_ = None

	def Load(self, reload:bool = False) -> DataT:
		if not self.loaded_ or reload:
			self.file_ = open(self.path_ + self.filename_)
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
			self.data_ = self.file_.read()
		return self.data_

	def __del__(self):
		if self.loaded_:
			self.file_.close()


class Pistachio:

	SettingsKey = 'PistachioDataset'

	def __init__(self) -> None:
		data = Settings.Data()
		super.__init(data[Pistachio.SettingsKey]['Path'])
	
	@classmethod
	def foo():
