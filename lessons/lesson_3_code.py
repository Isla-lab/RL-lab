import random
import os, sys, numpy
module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path: sys.path.append(module_path)
from tools.DangerousGridWorld import GridWorld
from collections import defaultdict




def on_policy_mc_epsilon_soft( environment, maxiters=5000, eps=0.3, gamma=0.99 ):
	"""
	Performs the on policy version of the every-visit MC control starting from the same state
	
	Args:
		environment: OpenAI Gym environment
		maxiters: timeout for the iterations
		eps: random value for the eps-greedy policy (probability of random action)
		gamma: gamma value, the discount factor for the Bellman equation
		
	Returns:
		policy: 1-d dimensional array of action identifiers where index `i` corresponds to state id `i`
	"""

	p = [[0 for _ in range(environment.action_space)] for _ in range(environment.observation_space)]   
	Q = [[0 for _ in range(environment.action_space)] for _ in range(environment.observation_space)]
	
	#
	# YOUR CODE HERE!
	#
	env = environment
	
	N = [[0 for _ in range(len(Q[0]))] for _ in range(len(Q))]
	p = [[eps/environment.action_space for _ in range(environment.action_space)] for _ in range(environment.observation_space)]
	for i in range(maxiters):
		g = 0.0
		episode = env.sample_episode(p)
		for s, a, r in reversed(episode):
			g = r + g*gamma
			N[s][a] += 1
			Q[s][a] += (g - Q[s][a])/N[s][a]

		for state in range(len(p)):
			best_action = numpy.argmax(Q[state])
			for a in range(len(p[state])):
				p[state][a] = eps/env.action_space
			p[state][best_action] += 1 - eps
		for s in range(4): assert abs(sum(p[s]) - 1.0) < 1e-9

		"""if i % 500 == 0:
			det_policy = [int(numpy.argmax(p[s])) for s in range(len(p))]
			print(f"Episode {i}, expected reward: {environment.evaluate_policy(det_policy):.2f}")"""
	
	deterministic_policy = [numpy.argmax(p[state]) for state in range(environment.observation_space)]	
	return deterministic_policy
	
	
def on_policy_mc_exploring_starts( environment, maxiters=5000, eps=0.3, gamma=0.99 ):
	"""
	Performs the on policy version of the every-visit MC control starting from different states
	
	Args:
		environment: OpenAI Gym environment
		maxiters: timeout for the iterations
		eps: random value for the eps-greedy policy (probability of random action)
		gamma: gamma value, the discount factor for the Bellman equation
		
	Returns:
		policy: 1-d dimensional array of action identifiers where index `i` corresponds to state id `i`
	"""
	p = [[0 for _ in range(environment.action_space)] for _ in range(environment.observation_space)]   
	Q = [[0 for _ in range(environment.action_space)] for _ in range(environment.observation_space)]
	
	#
	# YOUR CODE HERE!
	#
	for row in range(0,len(p)):
		for action in range(0, environment.action_space):
			p[row][action] = 0.25

	
	returns = defaultdict(list)
	N = [[0 for _ in range(environment.action_space)] for _ in range(environment.observation_space)]
	for _ in range(maxiters):
		start = random.randint(0, environment.observation_space-1)
		start_action = random.randrange(environment.action_space)
		episode = environment.sample_episode(initial_state=start, policy=p, initial_action=start_action)
		g = 0.0
		for s,a,r in reversed(episode):
			N[s][a] += 1
			g = r + g*gamma
			returns[(s,a)].append(g)
			Q[s][a] += (g-Q[s][a])/N[s][a]

		for s in range(len(p)):
			best_action = numpy.argmax(Q[s])
			for action in range(environment.action_space):
				p[s][action] = eps/environment.action_space
			p[s][best_action] = 1 - eps + eps/environment.action_space


	
	deterministic_policy = [numpy.argmax(p[state]) for state in range(environment.observation_space)]	
	return deterministic_policy


def main():
	print( "\n*************************************************" )
	print( "*  Welcome to the third lesson of the RL-Lab!   *" )
	print( "*            (Monte Carlo RL Methods)            *" )
	print( "**************************************************" )

	print("\nEnvironment Render:")
	env = GridWorld()
	env.render()

	print( "\n3) MC On-Policy (with exploring starts)" )
	mc_policy = on_policy_mc_exploring_starts( env, maxiters=5000 )
	env.render_policy( mc_policy )
	print( "\tExpected reward following this policy:", env.evaluate_policy(mc_policy) )
	
	print( "\n3) MC On-Policy (for epsilon-soft policies)" )
	mc_policy = on_policy_mc_epsilon_soft( env, maxiters=5000 )
	env.render_policy( mc_policy )
	print( "\tExpected reward following this policy:", env.evaluate_policy(mc_policy) )
	

if __name__ == "__main__":
	main()
