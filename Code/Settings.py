
import json

class Settings:

	Initialized_ = False
	Data_ = {}

	@classmethod
	def Data(cls):
		return cls.Data_

	@classmethod
	def Now(cls):
		if not cls.Initialized_:
			cls.LoadSettingsFile()
			cls.Initialized_ = True

	@classmethod
	def LoadSettingsFile(cls):
		file = None
		data = {}
		try:
			file = open('settings.json')
		except OSError:
			file = open('settings.json', 'w')
			cls.Data_['PistachioDataset'] = {
				'Path': 'C:\\Users\\Documents\\Dataset',
				'FileName': 'Pistachio_16_Features_Dataset.arff'
			}
			file.write(json.dumps(cls.Data_))
			print('No settings.json file was found. Creating one now...')
			print('Please take a moment to open settings.json and fill in appropriate values.')
		else:
			cls.Data_ = json.load(file)
			print('PathToPistachioDataset = {}'.format(cls.Data_['PistachioDataset']['Path']))
		finally:
			file.close()

Settings.Now()