import os, sys
import numpy as np; np.random.seed(6)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(6)


class MultiArmedBandit():
	"""
	A class that implements the N-armed testbed environment

	Attributes
	----------
		levers : int
			number of levers, same as the possible actions in the environment
		q_star : list
			value of each action, i.e., mean reward obtained when the corresponding action is selected
		sampling_variance: int
			variance of the returned reward selecting an action, the mean is q_star[action]
	
			
				The Multi-Armed Testbed
		environment consists of a set of
		N possible actions, from 1 to N.
		A mean value (q∗(a)) has been
		assigned to each action, sampled
		from a normal distribution with
		µ =0 and σ2 =1.
		For a given action a, the
		environment should return a
		reward sampled from a normal
		distribution with µ = q∗(a) and
		σ2 =1.	
	Methods
	-------
		action( action )
			return the reward obtained selecting the action 'action'; this value is obtained 
			sampling from a distribution with mean=q_star[action] and varaince=self.sampling_variance
	"""

	def __init__( self, levers ):
		#
		# YOUR CODE HERE!
		#	
		self.levers = levers
		self.q_star =  np.random.normal(0, 1, self.levers)
		self.sampling_variance = 1
			
	def action( self, action ):
		#
		# YOUR CODE HERE!
		reward = np.random.normal(self.q_star[action], self.sampling_variance)
		#
		return reward


def banditAlgorithm( env, eps=0, maxiters=1000 ):
	"""
	Implements the Simple Bandit Algorithm
	
	Args:
		env: instance of the multi-armed bandit environmet
		eps: random value for the eps-greedy policy (probability of random action)
		maxiters: number of steps to perform in the environment
		
	Returns:
		avg_reward: list of the rewards obtained during the training,, averaged from the first step to the last
		Q: the updated value function after the training
	"""	

	levers = env.levers
	Q = np.array([0 for _ in range(levers) ], dtype=float)
	N = np.array([0 for _ in range(levers) ], dtype=int)
	ep_reward = []; avg_reward = []
	
	for _ in range(maxiters):
		#
		# YOUR CODE HERE!
		act = None
		if np.random.rand() > eps:
			act = np.argmax(Q)	
		else:
			act = np.random.randint(levers)
			
		reward = env.action(act)
		N[act] += 1
		Q[act] += (1/N[act])*(reward-Q[act])
		

		ep_reward.append( reward )
		avg_reward.append( np.mean(ep_reward) )

	return avg_reward, Q


def main():
	print( "\n************************************************" )
	print( "*   Welcome to the second lesson of the RL-Lab!  *" )
	print( "*             (Multi-Armed Bandit)               *" )
	print( "**************************************************" )

	# Hyperparameters 
	n_armed = 10
	training_steps = 1000

	# Training phase
	print( f"\n{n_armed}-armed testbed with {training_steps} training_steps" )
	bandit = MultiArmedBandit( levers=n_armed )
	eps_00, q_00 = banditAlgorithm( bandit, eps=0.00, maxiters=training_steps ) 
	eps_01, q_01 = banditAlgorithm( bandit, eps=0.01, maxiters=training_steps ) 
	eps_10, q_10 = banditAlgorithm( bandit, eps=0.10, maxiters=training_steps )
	
	# Computing and plotting the reward of the last steps
	print( f"\n\tLast epsiodes reward (with eps=0   ):", np.mean(eps_00[-20:]) )
	print( f"\tLast epsiodes reward (with eps=0.01):", np.mean(eps_01[-20:]) )
	print( f"\tLast epsiodes reward (with eps=0.1 ):", np.mean(eps_10[-20:]) )
	
	# Computing and plotting the optimal action found
	print( "\n\tThe real optimal action is: ", bandit.q_star.argmax() )
	print( f"\tThe optimal actions found are: {q_00.argmax()} (eps=0), {q_01.argmax()} (eps=0.01), and {q_10.argmax()} (eps=0.1)" )

	# Repeat the experiment for 2000 episodes
	eps_00_average, eps_01_average, eps_10_average = [], [], []
	for _ in range(500):
		bandit = MultiArmedBandit( levers=n_armed )
		eps_00_average.append( banditAlgorithm( bandit, eps=0.00, maxiters=training_steps )[0] )
		eps_01_average.append( banditAlgorithm( bandit, eps=0.01, maxiters=training_steps )[0] )
		eps_10_average.append( banditAlgorithm( bandit, eps=0.10, maxiters=training_steps )[0] )

	eps_00_average = np.average(np.array(eps_00_average), axis=0)
	eps_01_average = np.average(np.array(eps_01_average), axis=0)
	eps_10_average = np.average(np.array(eps_10_average), axis=0)


	# Plot the results
	print( "\n\tPlotting data..." )
	t = np.arange(0, training_steps)
	_, ax = plt.subplots()
	ax.plot(t, eps_00_average, label="eps: 0", linewidth=3)
	ax.plot(t, eps_01_average, label="eps: 0.01", linewidth=3)
	ax.plot(t, eps_10_average, label="eps: 0.1", linewidth=3)
	plt.xlabel( "steps", fontsize=16)
	plt.ylabel( "average reward", fontsize=16)
	ax.grid()
	plt.ylim([0, None])
	plt.legend()
	save_path = os.path.join(os.path.dirname(__file__), "lesson2_plot.png")
	plt.savefig(save_path, dpi=150, bbox_inches="tight")
	print(f"\n\tPlot saved at {save_path}")
	

	

if __name__ == "__main__":
	main()
	