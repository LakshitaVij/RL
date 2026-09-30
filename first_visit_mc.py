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

"""
We're starting off with Monte Carlo method, and MC learns from the actual total return of a full episode
not a step-by-step estimate
It cannot update Q[state,action] until it knows everything that happened after that point, so it has to play the whole episode out first, then go back and compute returns from the 
recorded rewards
"""

def generate_episode(env, Q, epsilon, nA):
    episode = []
    state, _ = env.reset()
    done = False
    while not done :
        action = epsilon_greedy(Q,state, epsilon,nA)
        next_state , reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action,reward))
        done = terminated or truncated
        state = next_state
    return episode


ep = generate_episode(env,Q, epsilon=0.3, nA=4)
#print(len(ep))
#print(ep[:5])

"""
- returns_count = how many times have I updated this state,action pair so far?

The update function:
- takes the Q table
- the returns counter
- the full episode we just recorded
- gamma -> the discount factor, i.e how much we value future rewards vs immediate ones
gamma = 1.0 means dont discount at all, treat future rewards equally

- G = 0, -> return, the total accumulated reward from a point onward. Start at 0 bc we build it up by walking backward through the episode
- visited -> marks which state action pairs we have already updated this episode. Its a set because
we only want to count the first occurence's return 
- for t in reversed -> we start from the last step of the episode and work our way to the first, because G is 
the return, i.e the total reward from this point until the very end, so we just start at the very end, where the remaining return is just
whatever the last reward was, and we add each earlier reward as we go

- unpack the state, action, reward from episode 
- G = G currently holds the return from everything after this step, and we multiply with gamma, a d then add the current step's reward on top
Now G becomes the return starting from this step onward
With gamma=1.0, this just becomes a running sum of all rewards from here to the end — exactly the MC return definition.
-> visited.add = add to the set, we saw it
-> returns_count -> increment how many episodes have contributed a return to state,action pair
-> Q[state][action] += (G - Q[state][action]) / n
 an incremental average update. average all the returns ive ever seen for this state,action pair but compute without needing to store every past return
 G - Q[state][action] = how far off wass my current estimate from this new return?

"""

returns_count = defaultdict(int)

def update_Q(Q, returns_count, episode, gamma):
    G = 0
    visited = set()
    for t in reversed(range(len(episode))):
        state,action,reward = episode[t]
        G = gamma * G + reward
        if (state,action) not in visited:
            visited.add((state,action))
            returns_count[(state,action)]+=1
            n = returns_count[(state,action)]
            Q[state][action] += (G-Q[state][action])/n
    return Q, returns_count

num_episodes = 5000
episode_lengths = []

for i in range(num_episodes):
    episode= generate_episode(env, Q, epsilon=0.1, nA=4)
    Q, returns_count = update_Q(Q, returns_count,episode, gamma=1.0)
    episode_lengths.append(len(episode))

    if (i+1) %50 ==0:
         print(f"Episode {i+1}, length: {len(episode)}")


"""
path = [state] creates a list and puts the starting state into it right away (state 36, before any steps are taken). 
Then, inside the loop, every time the agent takes a step and lands somewhere new, we do path.append(state) to add that new state onto the end of the list too.
"""
state , _ = env.reset()
path = [state]
for _ in range(50):
    action = np.argmax(Q[state])
    state, reward, terminated, truncated, _ = env.step(action)
    path.append(state)
    if terminated or truncated:
        break
print(path)


def moving_average(x, window=50):
    return np.convolve(x, np.ones(window)/window, mode='valid')

smoothed = moving_average(episode_lengths, window=50)

plt.figure(figsize=(10, 5))
plt.plot(episode_lengths, alpha=0.3, label='Raw episode length')
plt.plot(range(len(smoothed)), smoothed, label='Smoothed (50-episode avg)', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Episode length (steps)')
plt.title('MC Control on Cliff Walking — Episode Length over Training')
plt.legend()
plt.savefig('mc_cliffwalking_learning_curve.png')
plt.show()