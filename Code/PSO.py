
from random import randint, random
import math
from typing import TypeVar
from Loss import Loss
from Log import Log
import Model


class Particle:

	NEXT_ID = 0

	# A a list of floats
	RealV = TypeVar('RealV', list[float], list[int])
	ValuePositionT = TypeVar('ValuePositionT', tuple[float, list[float]], tuple[float, list[int]])
	def __init__(self, position:RealV, velocity:RealV = None) -> None:
		self.position_ = position
		self.velocity_ = velocity
		self.best_ = None
		self.id_ = Particle.NEXT_ID
		if self.id_ < 10:
			self.id_ = '000{}'.format(self.id_)
		elif self.id_ < 100:
			self.id_ = '00{}'.format(self.id_)
		elif self.id_ < 1000:
			self.id_ = '0{}'.format(self.id_)
		else:
			self.id_ = '{}'.format(self.id_)
		Particle.NEXT_ID += 1

	def Position(self, pos:RealV = None) -> RealV:
		if pos != None:
			self.position_ = pos
		return self.position_

	def Velocity(self, vel:RealV = None) -> RealV:
		if vel != None:
			#if self.velocity_ != None:
			#	print('Particle change vector loss: ', Loss.L2(self.velocity_, vel))
			factor = sum([v**2 for v in vel]) ** 0.5
			if factor > 2:
				vel = [v/factor for v in vel]
			self.velocity_ = vel
		return self.velocity_

	def Boost(self):
		vel = self.Velocity()
		v_len = sum([v**2 for v in vel] )** 0.5
		boost = (1-v_len)/v_len
		self.Velocity([v * boost for v in vel])

	def Best(self, value:float = None, position:RealV = None) -> ValuePositionT:
		if value == None:
			return self.best_

		if position == None:
			position = self.position_

		if self.best_ == None or self.best_[0] > value:
			self.best_ = (value, position)
		return self.best_

	def ResetBest(self):
		self.best_ = None

class PSO:

	DEBUG = True

	POSITION = 1
	VELOCITY = 2
	PARAMS = {'Inertia': 0.5, 'Cognitive': 0.6, 'CognitiveRange': (0, 1), 'Social': 0.7, 'SocialRange': (0, 1)}
	
	RealV = TypeVar('RealV', list[float], list[int])
	ValuePositionT = TypeVar('ValuePositionT', tuple[float, list[float]], tuple[float, list[int]])
	IntBounds = TypeVar('IntBounds', tuple[int, int], tuple[int, int])

	def __init__(self, model:Model.Model, size:IntBounds, prand:IntBounds = (-1,1), vrand:IntBounds = (-1,1)) -> None:
		if not isinstance(model, Model.Model):
			raise Exception('The PSO class requires "model" to be of type Model.Model.')
		self.model = model
		self.size = size
		self.prand = prand
		self.vrand = vrand
		self.Reset()

	def Execute(self, X:RealV, y:RealV, weights:list[RealV] = None, particles = None, iterations:int = 30) -> list[Particle]:
		"""If particles is None, then a list of particles is generated
			with random elements using self.size and self.prand
			Otherwise, particles are assumed to be initialized already

		Args:
			X (RealV): Feature data to use with the model's prediction and the
				loss function.
			y (RealV): Classification data to use with the model's prediction and the
				loss function.

		Returns:
			list[RealV]: The modified particle positions.
		"""
		outputs = []
		if particles == None:
			if weights == None:
				particles = self.RandParticles()
			else:
				particles = [Particle(position) for position in weights]
		if iterations < 1:
			iterations = 1
		for j in range(iterations):
			for particle in particles:

				if j == 0:
					if PSO.DEBUG: Log.Add('pso-particle-position-{}'.format(particle.id_), "epoch: {}; y: {}\n".format(self.epoch, y))
					particle.ResetBest()

				output = self.model.Execute(X, particle.Position())
				outputs.append(output)

				loss = Loss.L2(output, y)
				if PSO.DEBUG: Log.Add('pso-particle-loss-{}'.format(particle.id_), "loss: {}\n".format(round(loss,2)))
				#print('iteration: {}; outputs: {}; loss: {}'.format(j, outputs[-1], loss))
				particle.Best(loss)


				if self.globalBest[0] == None or particle.Best()[0] < self.globalBest[0]:
					self.globalBest = particle.Best()
					self.globalBestRecords.append(self.globalBest)

					
			for particle in particles:
				""" "isCleanSlate" means this is the first round of PSO
				# and some things are only done on this round
				# The neural network which is contained in self.model
				# is executed here
				"""
				if self.isCleanSlate:
					# Initialize the velocity vector
					self.RandomizeVelocity(particle)
				else:
					self.UpdateVelocity(particle)
				self.UpdatePosition(particle)


			if self.isCleanSlate:
				self.isCleanSlate = False
		
		return particles

	def ExecuteMany(self, X:list[RealV], y:list[RealV], weights:list[RealV] = None, iterations:int = 30) -> list[ValuePositionT]:

		particles = None
		if weights == None:
			particles = self.RandParticles()
		else:
			particles = [Particle(position) for position in weights]
		for i in range(len(X)):
			self.globalBest = (None, None)
			particles = self.Execute(X[i], y[i], particles=particles, iterations=iterations)
			self.epoch += 1
			print('epoch: {}; gbest: {}'.format(i, self.globalBest[0]))
			for particle in particles:
				particle.Boost()


		for particle in particles:
			Log.Add('pso-particle-global-comparison', "diff: {}\n".format(round(Loss.L2(particle.Position(), self.globalBest[1]),4)))
			

		return self.GetRecords()

	def UpdateVelocity(self, particle:Particle) -> None:
		oldvelocity = particle.Velocity()
		(inertia, cognitive, social) = (
			PSO.PARAMS['Inertia'],
			PSO.PARAMS['Cognitive'] * (randint(PSO.PARAMS['CognitiveRange'][0], PSO.PARAMS['CognitiveRange'][1]-1) + random()),
			PSO.PARAMS['Social'] * (randint(PSO.PARAMS['SocialRange'][0], PSO.PARAMS['SocialRange'][1]-1) + random()),
		)
		particle.Velocity([sum(term) for term in zip(
			[inertia * v for v in particle.Velocity()],
			[cognitive * x for x in [p - z for (p, z) in zip(particle.Best()[1], particle.Position())]],
			[social * x for x in [g - z for (g, z) in zip(self.globalBest[1], particle.Position())]]
		)])
		if PSO.DEBUG: Log.Add('pso-particle-loss-{}'.format(particle.id_), "speed: {}\n".format(round(Loss.L2(particle.Velocity(), [0] * len(particle.Velocity())),3)))

	def UpdatePosition(self, particle:Particle) -> None:
		oldposition = particle.Position()
		newposition = particle.Position([sum(term) for term in zip(oldposition, particle.Velocity())])
		Log.Add('pso-particle-loss-{}'.format(particle.id_), "position change: {}\n".format(round(Loss.L2(newposition, oldposition),2)))
		Log.Add('pso-particle-position-{}'.format(particle.id_), "before: {}\n".format([round(item,2) for item in oldposition[0:10]]))
		Log.Add('pso-particle-position-{}'.format(particle.id_), "after : {}\n".format([round(item,2) for item in newposition[0:10]]))

	def Reset(
		self,
		globalBest:ValuePositionT = (None, None),
		isCleanSlate:bool = True,
		globalBestRecords:list[ValuePositionT] = [],
		epoch:int = 0
	) -> None:
		self.globalBest = globalBest
		self.isCleanSlate = isCleanSlate
		self.globalBestRecords = globalBestRecords
		self.epoch = epoch

	# records are a list of (loss value, particle position)
	def GetRecords(self) -> list[ValuePositionT]:
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
		

	def RandParticles(self, size:IntBounds = None) -> list[Particle]:
		if size == None:
			size = self.size
		if size[0] < 1 or size[1] < 1:
			raise Exception('PSO.RandParticles requires a "size" tuple with each component an integer greater than or equal to 1.')
		
		particles = []
		for i in range(size[0]):
			particles.append(Particle(self.__randList(size[1], PSO.POSITION)))

		return particles

	def RandomizeVelocity(self, particle:Particle):
		particle.Velocity(self.__randList(self.size[1], PSO.VELOCITY))

	

