import os, sys, numpy
module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path: sys.path.append(module_path)
from DangerousGridWorld import GridWorld


def epsilon_greedy(q, state, epsilon):
	"""
	Epsilon-greedy action selection function
	
	Args:
		q: q table
		state: agent's current state
		epsilon: epsilon parameter
	
	Returns:
		action id
	"""
	if numpy.random.random() < epsilon:
		return numpy.random.choice(q.shape[1])
	return q[state].argmax()


def q_learning(environment, episodes, alpha, gamma, expl_func, expl_param):
	"""
	Performs the Q-Learning algorithm for a specific environment
	
	Args:
		environment: OpenAI Gym environment
		episodes: number of episodes for training
		alpha: alpha parameter
		gamma: gamma parameter
		expl_func: exploration function (epsilon_greedy, softmax)
		expl_param: exploration parameter (epsilon, T)
	
	Returns:
		(policy, rewards, lengths): final policy, rewards for each episode [array], length of each episode [array]
	"""
	
	q = numpy.zeros((environment.observation_space, environment.action_space))  # Q(s, a)
	rews = numpy.zeros(episodes)
	lengths = numpy.zeros(episodes)
	#
	# YOUR CODE HERE!
	#
	policy = q.argmax(axis=1) # q.argmax(axis=1) automatically extract the policy from the q table
	return policy, rews, lengths


def sarsa(environment, episodes, alpha, gamma, expl_func, expl_param):
	"""
	Performs the SARSA algorithm for a specific environment
	
	Args:
		environment: OpenAI gym environment
		episodes: number of episodes for training
		alpha: alpha parameter
		gamma: gamma parameter
		expl_func: exploration function (epsilon_greedy, softmax)
		expl_param: exploration parameter (epsilon, T)
	
	Returns:
		(policy, rewards, lengths): final policy, rewards for each episode [array], length of each episode [array]
	"""

	q = numpy.zeros((environment.observation_space, environment.action_space))  # Q(s, a)
	rews = numpy.zeros(episodes)
	lengths = numpy.zeros(episodes)
	#
	# YOUR CODE HERE!
	#	
	policy = q.argmax(axis=1) # q.argmax(axis=1) automatically extract the policy from the q table
	return policy, rews, lengths


def main():
	print( "\n*************************************************" )
	print( "*  Welcome to the fourth lesson of the RL-Lab!   *" )
	print( "*        (Temporal Difference Methods)           *" )
	print( "**************************************************" )

	print("\nEnvironment Render:")
	env = GridWorld()
	env.render()

	# Learning parameters
	episodes = 500
	alpha = .3
	gamma = .9
	epsilon = .1

	# Executing the algorithms
	policy_qlearning, rewards_qlearning, lengths_qlearning = q_learning(env, episodes, alpha, gamma, epsilon_greedy, epsilon)
	policy_sarsa, rewards_sarsa, lengths_sarsa = sarsa(env, episodes, alpha, gamma, epsilon_greedy, epsilon)

	print( "\n4) Q-Learning" )
	env.render_policy( policy_qlearning )
	print( "\tExpected reward training with Q-Learning:", numpy.round(numpy.mean(rewards_qlearning), 2) )
	print( "\tAverage steps training with Q-Learning:", numpy.round(numpy.mean(lengths_qlearning), 2) )

	print( "\n5) SARSA" )
	env.render_policy( policy_sarsa )
	print( "\tExpected reward training with SARSA:", numpy.round(numpy.mean(rewards_sarsa), 2) )
	print( "\tAverage steps training with SARSA:", numpy.round(numpy.mean(lengths_sarsa), 2) )
	

if __name__ == "__main__":
	main()
