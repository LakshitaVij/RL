import gymnasium as gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

env = gym.make("CliffWalking-v1")
state, info = env.reset()
print(state, info)

"""
0 = up
1 = right
2 = down
3 = left
"""
next_state, reward, terminated, truncated, info = env.step(1)
print(next_state, reward, terminated, truncated, info)

"""
next_state = where you end up after the move, a number between 0-47 bc thats how big the grid is
reward = number env gives for that move 
-1 for ordinary step when u dont reach state
-100 for cliff 
terminated = did the episode end because something happened in env, like reaching goal or falling off of a cliff?
truncated = did the episode end for an artificial reason like max steps etc 
info = metadata

Here why terminated is false even though we went to 37 which is cliff is because it penalty and resets , and 
actually only terminates when final state is reached
"""
env.reset()
print(env.step(0))


nS = env.observation_space.n #how many states, 0-47 so 48
nA = env.action_space.n #how many actions, up down left right, so 4
print(nS, nA)

"""
Now we'll build a Q-table, which is essentially the agent's memory, essentially its current belief of how good
each state action pair is.
I.e if Q[36,1] = how much future reward do I expect if I am at state 36 and I move right?
At the start, the agent has taken 0 actions and has seen 0 rewards
So in the beginning since it sees nothing, we initialise the table to 0 since we dont know

the Q-table gets updated as the agent learns
Everytime it takes a step, or sees a reward, we nudge the relevant Q[state, action] towards the reward it actually observed. 

Over hundreds of episodes, the zeroes gradually turn into an accurate map of, "hey, this is what we've learned as good or bad actions"

"""
Q = np.zeros((nS, nA))
print(Q.shape)


"""
Epsilon greedy function
- Q -> takes the Q table we made, which judges essentially how good a move is
- state -> which square are we currently on between 0-47 for this case
- epsilon -> a number between 0 and 1 controlling how much exploration we wanna do.
epsilon = 0.1 would be like, "hey, 10% of the time, do something random, go explore king"
- nA = number of possible actions (in this case, 4)

So essentially, if your random decimal number is less than epsilon, explore, and do a random action,not doing 
whats considered the best necessarily and not exploiting


- q_values -> Grabs the row of Q-values for the current state — a list of 4 numbers, one per action. E.g., at the start, Q[36] might be [0.0, 0.0, 0.0, 0.0].
- max_q -> find the single highest value in that row
- best_actions = np.flatnonzero(q_values == max_q) -> compares every entry in q_values to max_q, and gives back a list of True/False — 
True wherever that action's value equals the max. 
If all four values are tied at 0.0, this gives [True, True, True, True], since every action equals the max.
- np.flatnonzero -> converts true/false to indices.
So if all four are tied, best_actions becomes [0, 1, 2, 3]
 meaning "every action is equally good, as far as I currently know."


- return np.random.choice(best_actions)->Instead of always picking the first tied action
 (what argmax was silently doing), this picks randomly among all the actions that are tied for best.
   So when everything's tied at zero, you get a genuinely random pick among all 4, not a hidden bias toward action 0.
"""

def epsilon_greedy(Q, state, epsilon, nA):
    if np.random.rand() < epsilon:
        return np.random.randint(nA)
    q_values = Q[state] 
    max_q = np.max(q_values)
    best_actions = np.flatnonzero(q_values==max_q)
    return np.random.choice(best_actions)

print(epsilon_greedy(Q, 36, epsilon=0.0, nA=4)) #always exploit, since the Q-table is all zeros, every action ties at 0.0.. np.argmax just returns the first max it finds, which is index 0. So this will always print 0, every single time, no matter how many times you run it.

print(epsilon_greedy(Q, 36, epsilon=0.1, nA=4)) #Only a 10% chance of hitting the random branch, so most of the time (like this run) it still falls through to argmax → 0.

print(epsilon_greedy(Q, 36, epsilon=0.5, nA=4)) #exploration 50 exploitation 50

print(epsilon_greedy(Q, 36, epsilon=1.0, nA=4)) #100 explore

def q_learning(env, epsilon, nA, Q, gamma, alpha):
    state, _ = env.reset()
    action = epsilon_greedy(Q,state, epsilon, nA)
    done = False
    step = 0
    while not done:
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_action = epsilon_greedy(Q,next_state, epsilon, nA)
        td_target = reward + gamma * np.max(Q[next_state]) * (not terminated)
        Q[state][action] += alpha * (td_target - Q[state, action])  
        state = next_state
        action = next_action
        step+=1
    return Q, step

num_episodes = 5000
episode_lengths = []

for i in range(num_episodes):
        Q, step= q_learning(env, epsilon=0.1, nA=nA, Q=Q, gamma=1, alpha=0.1)
        episode_lengths.append(step)
        if (i+1) %50 ==0:
            print(f"Episode {i+1}, steps = {step}")

state, _ = env.reset()
path = [state]
for _ in range(50):
     action = np.argmax(Q[state])
     state, reward, terminated, truncated, _ = env.step(action)
     path.append(state)
     if terminated or truncated:
          break
print(path)

import matplotlib.pyplot as plt

def moving_average(x, window=50):
    return np.convolve(x, np.ones(window)/window, mode='valid')

smoothed = moving_average(episode_lengths, window=50)

plt.figure(figsize=(10, 5))
plt.plot(episode_lengths, alpha=0.3, label='Raw episode length')
plt.plot(range(len(smoothed)), smoothed, label='Smoothed (50-episode avg)', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Episode length (steps)')
plt.title('Q-Learning on Cliff Walking — Episode Length over Training')
plt.legend()
plt.savefig('qlearning_cliffwalking_learning_curve.png')
plt.show()