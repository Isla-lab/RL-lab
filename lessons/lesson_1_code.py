from pydoc import render_doc
import os, sys, numpy, random
module_path = os.path.abspath(os.path.join('../tools'))

if module_path not in sys.path: sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools')))

#if module_path not in sys.path: sys.path.append(module_path)
from DangerousGridWorld import GridWorld


def random_dangerous_grid_world( environment ):
	"""
	Performs a random trajectory on the given Dangerous Grid World environment 
	
	Args:
		environment: OpenAI Gym environment
		
	Returns:
		trajectory: an array containing the sequence of states visited by the agent
	"""
	trajectory = []
	#
	# YOUR CODE HERE!
	#
	env = environment
	state = env.start_state
	goal = env.goal_state
	env.robot_state = state
	
	trajectory = []
	for step in range(10):
		action = numpy.random.randint(0, env.action_space)
		next_state = env.sample(action, state)
		reward = env.R[next_state]
		trajectory.append([state, action, reward])
		
		env.robot_state = next_state
		state = next_state

		print(f"\nStep {step+1}: action = {env.actions[action]}, reward = {reward}")
		env.render()

		if env.is_terminal(state):
			print("\n💀 Episode terminated!")
			break
	

	return trajectory


class RecyclingRobot():
	"""
	Class that implements the environment Recycling Robot of the book: 'Reinforcement
	Learning: an introduction, Sutton & Barto'. Example 3.3 page 52 (second edition).
		
	Attributes
	----------
		observation_space : int
			define the number of possible actions of the environment
		action_space: int
			define the number of possible states of the environment
		actions: dict
			a dictionary that translate the 'action code' in human languages
		states: dict
			a dictionary that translate the 'state code' in human languages
		
	Methods
	-------
		reset( self )
			method that reset the environment to an initial state; returns the state
		step( self, action )
			method that perform the action given in input, computes the next state and the reward; returns 
			next_state and reward
		render( self )
			method that print the internal state of the environment
	"""


	def __init__( self ):

		# Loading the default parameters
		self.alfa = 0.7
		self.beta = 0.7
		self.r_search = 0.5
		self.r_wait = 0.2

		# Defining the environment variables
		self.observation_space = 2
		self.action_space = 3
		self.actions = {0: 'search', 1: 'wait', 2: 'recharge'}
		self.states = {0: 'high', 1: 'low'}
		


	def reset( self ):
		#
		# YOUR CODE HERE!
		self.state = 0
		#
		return self.state


	def step( self, action ):
		

		reward = 0
		#
		# YOUR CODE HERE!
		#
		act = self.actions[action]
		print(f'selected action: {act}')
		if self.state == 0 and act == 'recharge':
			return self.state, reward, False, None
		elif self.state == 1 and act == 'recharge':
			self.state = 0
			reward += 0
		if act == 'search':
			if self.state == 0:
				new_state = numpy.random.choice(self.observation_space, p=[self.alfa, 1-self.alfa])
				self.state = new_state
				reward += self.r_search
			elif self.state == 1:
				new_state = numpy.random.choice(self.observation_space, p=[self.beta, 1-self.beta])
				if new_state == 0:
					reward += -3
				elif new_state == 1:
					reward += self.r_search
				self.state = new_state
		if act == 'wait':
			reward += self.r_wait
			
		#
		return self.state, reward, False, None


	def render( self ):

	#
		
	#
		return True


def main():
	print( "\n************************************************" )
	print( "*  Welcome to the first lesson of the RL-Lab!  *" )
	print( "*             (MDP and Environments)           *" )
	print( "************************************************" )

	print( "\nA) Random Policy on Dangerous Grid World:" )
	env = GridWorld()
	env.render()
	random_trajectory = random_dangerous_grid_world( env )
	print( "\nRandom trajectory generated:", random_trajectory )


	print( "\nB) Custom Environment: Recycling Robot" )
	env = RecyclingRobot()
	state = env.reset()
	ep_reward = 0
	
	for step in range(10):
		a = numpy.random.randint( 0, env.action_space )
		new_state, r, _, _ = env.step( a )
		ep_reward += r
		print( f"\tFrom state '{env.states[state]}' selected action '{env.actions[a]}': \t total reward: {ep_reward:1.1f}" )
		state = new_state


if __name__ == "__main__":
	main()
