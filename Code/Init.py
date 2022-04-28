
import json

class Init:

	Initialized = False

	@classmethod
	def Now(cls):
		if not cls.Initialized:
			cls.GetSettingsFile()
			cls.Initialized = True

	@classmethod
	def GetSettingsFile(cls):
		file = None
		data = {}
		try:
			file = open('settings.json')
		except OSError:
			file = open('settings.json', 'w')
			data['PistachioDataset'] = {
				'Path': 'C:\\Users\\Documents\\Dataset',
				'FileName': 'Pistachio_16_Features_Dataset.arff'
			}
			file.write(json.dumps(data))
			print('No settings.json file was found. Creating one now...')
			print('Please take a moment to open settings.json and fill in the missing values.')
		else:
			data = json.load(file)
			print('PathToPistachioDataset = {}'.format(data['PistachioDataset']['Path']))
		finally:
			file.close()

Init.Now()