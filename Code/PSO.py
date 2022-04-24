
from random import randint, random
from typing import Callable
from Loss import Loss


class Particle:
	def __init__(self, position:list[float], velocity:list[float] = None) -> None:
		self.position_ = position
		self.velocity_ = velocity
		self.best_ = None

	def Position(self, pos:list[float] = None) -> list[float]:
		if pos != None:
			self.position_ = pos
		return self.position_

	def Velocity(self, vel:list[float] = None) -> list[float]:
		if vel != None:
			self.velocity_ = vel
		return self.velocity_

	def Best(self, value: float = None, position:list[float] = None) -> tuple[float, list[float]]:
		if value == None:
			return self.best_

		if position == None:
			position = self.position_

		if self.best_ == None or self.best_[0] > value:
			self.best_ = (value, position)
		return self.best_


class PSO:

	POSITION = 1
	VELOCITY = 2
	PARAMS = {'Inertia': 1, 'Cognitive': 1, 'CognitiveRange': (0, 1), 'Social': 0.1, 'SocialRange': (0, 1)}

	def __init__(self, model:Callable[ [list[float], list[float]], list[float]], size:tuple[int,int], prand:tuple[int,int] = (-1,1), vrand:tuple[int,int] = (-1,1)) -> None:
		if not callable(model):
			raise Exception('The PSO class requires "Model" to be callable.')
		self.model = model
		self.size = size
		self.prand = prand
		self.vrand = vrand
		self.Reset()

	def Execute(self, X:list[float], y:list[float], weights:list[list[float]] = None, iterations:int = 30) -> list[float]:
		"""If particles is None, then a list of particles is generated
			with random elements using self.size and self.prand
			Otherwise, particles are assumed to be initialized already

		Args:
			X (list[float]): Feature data to use with the model's prediction and the
				loss function.
			y (list[float]): Classification data to use with the model's prediction and the
				loss function.

		Returns:
			list[list[float]]: The modified particle positions.
		"""
		outputs = []
		if weights == None:
			particles = self.RandParticles()
		else:
			particles = [Particle(position) for position in weights]
		if iterations < 1:
			iterations = 1
		for j in range(iterations):
			for i, particle in enumerate(particles):
				outputs.append(self.model(X, particle.Position()))
				
				particle.Best(Loss.L2(outputs[-1], y))

				if self.globalBest[0] == None or particle.Best()[0] < self.globalBest[0]:
					self.globalBest = particle.Best()
					self.globalBestRecords.append(self.globalBest)
					
				""" "isCleanSlate" means this is the first round of PSO
				# and some things are only done on this round
				# The neural network which is contained in self.model
				# is executed here
				"""
				if self.isCleanSlate:
					# Initialize the velocity vector
					particle.Velocity(self.__randList(self.size[1], PSO.VELOCITY))
			for i, particle in enumerate(particles):
				if self.isCleanSlate:
					# Initialize the velocity vector
					particle.Velocity(self.__randList(self.size[1], PSO.VELOCITY))
				else:
					self.UpdateVelocity(particle)

				self.UpdatePosition(particle)

			if self.isCleanSlate:
				self.isCleanSlate = False

		return self.globalBest[1]

	def UpdateVelocity(self, particle:Particle) -> None:
		(inertia, cognitive, social) = (
			PSO.PARAMS['Inertia'],
			PSO.PARAMS['Cognitive'] * randint(PSO.PARAMS['CognitiveRange'][0], PSO.PARAMS['CognitiveRange'][1]-1) + random(),
			PSO.PARAMS['Social'] * randint(PSO.PARAMS['SocialRange'][0], PSO.PARAMS['SocialRange'][1]-1) + random(),
		)
		particle.Velocity([sum(term) for term in zip(
			[inertia * v for v in particle.Velocity()],
			[cognitive * x for x in [p - z for (p, z) in zip(particle.Best()[1], particle.Position())]],
			[social * x for x in [g - z for (g, z) in zip(self.globalBest[1], particle.Position())]]
		)])

	def UpdatePosition(self, particle:Particle) -> None:
		particle.Position([sum(term) for term in zip(particle.Position(), particle.Velocity())])

	def Reset(self) -> None:
		self.globalBest = (None, None)
		self.isCleanSlate = True
		self.globalBestRecords = []

	# records are a list of (loss value, particle position)
	def GetRecords(self) -> list[tuple[float, list[float]]]:
		return self.globalBestRecords

	def __randList(self, size:int, ofType:int) -> list:
		rnd = None
		if ofType == PSO.POSITION:
			rnd = self.prand
		elif ofType == PSO.VELOCITY:
			rnd = self.vrand
		else:
			raise Exception('PSO.__randList requires ofType to be either PSO.POSITION or PSO.VELOCITY.')

		return [randint(rnd[0], rnd[1]-1) + random() for _ in range(size)]
		

	def RandParticles(self, size:tuple[int, int] = None) -> list[list]:
		if size == None:
			size = self.size
		if size[0] < 1 or size[1] < 1:
			raise Exception('PSO.RandParticles requires a "size" tuple with each component an integer greater than or equal to 1.')
		
		particles = []
		for i in range(size[0]):
			particles.append(Particle(self.__randList(size[1], PSO.POSITION)))

		return particles

	

