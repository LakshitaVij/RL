# RL learning notes - SARSA and Q learning

### Overview

The aim here was to build an RL agent that learns to navigate the Cliff Walking environment using TD learning methods with epsilon greedy exploration.

I used Cliff Walking specifically since it has a solidly clear trade-off between a short risky path (along the cliff) vs a longer safer path , and this is good for observing whether the learned behavior makes sense

Moreover, I want to practice several different tabular RL methods without recreating the scaffolding , so Cliff Walking was a good pick for that as well!

The agent at the end did learn a working policy that reaches it goal terminal state, however there were some hurdles along the way.

Why I am writing a joint post about the SARSA and Q learning comparison is because 

## Environment

This is how Cliff Walking is structured:

- The grid is a 4x12 layout, with 48 states in total, labelled 0-47
- The agent starts at state 36 and goal is state 47 as depicted below:
    
    ![image.png](RL%20learning%20notes%20-%20SARSA%20and%20Q%20learning/image.png)
    
- States 37 to 46, the ones marked red in the figure above, are the cliff. Stepping on even one of them gives a big penalty of -100 to the agent, and resets the agent to the start without ending the episode, and I understood that the episode does not terminate if the agent steps on the cliff by playing around with the environment by hand.
- There are 4 possible discrete actions the agent can take:
    - UP = 0
    - RIGHT = 1
    - DOWN = 2
    - LEFT = 3
- Rewards = -1 per ordinary step, -100 for the cliff, and the episode only truly ends when the goal is reached
- Since every step costs atleast -1, the agent is incentivised to find the shortest path (which is not the cliff of course)

## Method

- While monte carlo methods need the whole episode to be completed to update the q-table, the TD learning methods of SARSA and Q learning update the table after every step.
- The Q-table is essentially the agent’s memory of what it learns. We start off by it being initialized to zero since it has not interacted with the environment yet and does not know better.
- Epsilon greedy is essentially how we determine the explore exploit trade-off for the agent. We set how much the agent should explore new outcomes, disregarding what the “best outcome” is supposed to be, and thereby also setting how much it should exploit, or just use the best outcome.  One essential change that I added here was, if the four action value’s are tied, it picks randomly amongst all tied actions , so there’s no hidden bias towards any one of them.
- While Monte Carlo methods have no bootstrapping, and you have to wait till the end of the episode for the q table to learn and update, TD methods use bootstrapping. After one step, we take the reward we get and add our current guess of how good the next state is, based on our own Q table. Essentially, since we are using an estimate to update another estimate , we are “bootstrapping”. Why this works is because the next state’s estimate itself is getting updated every time we visit it, and real rewards keep entering the table and gradually propagate backward through the estimates
- The trade off is:
    - MC is unbiased but it is noisy, since one full episode has lots of randomness in it
    - TD is biased early, and it leans on wrong guesses, but lower variance and it learns every step instead of once per episode

### SARSA

- stands for state, action, reward, next state, next action
- The next action is picked via epsilon greedy before the update, and that action is used in the target
- **The mechanism, in order:**
1. You're in state S_t and take action A_t.
2. The environment gives you a reward R_{t+1} and a new state S_{t+1}.
3. Before updating, you pick your next action A_{t+1} with epsilon-greedy.
4. You update Q(S_t, A_t) using the Q-value of that exact next action.



$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha\big[R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)\big]
$$



- To quickly explain the values here:
    - Q(s,a) is our current average estimate
    - α (alpha) → **the step size (learning rate).** A fixed number between 0 and 1 that you choose. It controls how far each update moves Q toward the target. α = 0.1 means "close 10% of the gap each time." MC used 1/N(s,a), which shrinks as you collect more data. TD uses a constant α, because the target itself is a moving estimate rather than a real observed return, so there's no reason to shrink the step toward zero.
    - t → time step index within the episode.
    - T→ the final step of the episode, the step at which the episode terminates.
    - $R_{t+1}$ → The reward an agent receives as a consequence of the action it took at time step t
    - γ (gamma)→ The discount factor, a number between 0 and 1. Basically it controls how we weight present rewards compared to future ones. Close to 0 means we just care about present/immediate rewards, and close to 1 means I care about future rewards as much as immediate rewards, i.e no discounting at all.
    - $\gamma Q(S_{t+1}, A_{t+1})$: your current guess of the future value, starting from the next state and the next action you've picked. γ discounts it (you used 1, so no discounting).
    - 
    
    $$
    \underbrace{R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})}_{\text{TD target}}
    $$
    
- $-\,Q(S_t, A_t)$: subtract your current estimate for the state-action pair you just acted from. This is the old belief.
- in our code since it is a recursive relationship, we just compute it as a simple backward pass
- Since the action in the target is the one it will actually take, and that action comes from the epsilon greedy policy with random exploration included, so SARSA learns the value of the policy its really following. If 10% of the time it randomly stumbles, that risk is baked into its Q-values.
- If the episode truly terminates (`terminated`), the bootstrap term is set to zero, since there's no future value beyond the goal. The reward still counts, so the update still happens. We use `terminated` and not `truncated`, because a time-limit cutoff doesn't mean nothing exists beyond that point
- After training, I evaluated the final policy exactly as in the MC post: always selecting `argmax(Q[state])` with no exploration, starting from state 36, and recording the states visited until the goal
- 

### Q Learning

- Very similar to SARSA, it just has one small change, which is in its update rule $Q(S_{t+1},A_{t+1})$ becomes $\max_a Q(S_{t+1},a)$.
- So instead of using the next action we will actually take, here we use the best next action according to the table



$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha\big[R_{t+1} + \gamma \max_a Q(S_{t+1},a) - Q(S_t,A_t)\big]
$$



$\max_a Q(S_{t+1},a)$ means the largest Q-value across all actions available in the next state. In code, `np.max(Q[next_state])` replaces `Q[next_state][next_action]`



- The TD target is now $R_{t+1} + \gamma \max_a Q(S_{t+1},a)$: the reward we just got plus our best guess of the future, assuming we act greedily from here on
- As in SARSA, the bootstrap term is zeroed when the episode truly terminates (`not terminated`), but the reward still counts
- This is off policy since it has two policies involved:
    - Behavior policy: what you actually do, which is epsilon greedy with random exploration
    - Target Policy : what you’re learning about, which is the purely greedy policy (always the max)
- SARSA updates towards the action it actually takes next, whereas Q-learning updates towards the best action available whatever it ends up doing

## Results

- I had set : α = 0.1, γ = 1, ε = 0.1, 5000 episodes, identical for both algorithms so the comparison is fair between SARSA and Q learning

![sarsa_cliffwalking_learning_curve.png](RL%20learning%20notes%20-%20SARSA%20and%20Q%20learning/sarsa_cliffwalking_learning_curve.png)

- SARSA's episode length starts near 900 steps and falls steeply in the first ~200 episodes, then declines gradually before settling at roughly 15-25 steps by around episode 2000, with very little variance afterward.
- The final learned policy looks like:

[36, 24, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 35, 47]


- This is a complete 16 step route that starts from 36 and ends at the terminal state 47 with no repeated states and no steps on the cliff, which means the agent learned to avoid high penalty cliff squares while finding a reasonably short path to the goal

![qlearning_cliffwalking_learning_curve.png](RL%20learning%20notes%20-%20SARSA%20and%20Q%20learning/qlearning_cliffwalking_learning_curve.png)

- Q-learning's episode length starts near 1000 steps in the first episode, then falls sharply, reaching roughly 15-20 steps by around episode 300-500. After that the curve is essentially flat through episode 5000, with only small spikes from epsilon-greedy exploration.
- Under identical hyperparameters (α = 0.1, γ = 1, ε = 0.1), Q-learning's curve flattens within a few hundred episodes, while SARSA's declines gradually and flattens around episode 2000.


[36, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 47]



- This is a complete 14 step route that starts from 36 and ends at the terminal state 47 with no repeated states and no steps on the cliff, which means the agent learned to avoid high penalty cliff squares while finding a reasonably short path to the goal

### Why paths differ

- SARSA updates using the action it will really take next, and that action can be a random one. So $Q(S_{t+1}, A_{t+1})$ already has the agent's own clumsiness baked in
- Picture a square right next to the cliff. Every step, there's a small chance that epsilon-greedy throws in a random move and that move happens to be "down", which sends the agent off the edge. That chance is about $\epsilon/4$ per step, because a random action picks one of 4 directions uniformly
- That -100 keeps leaking into the Q-values of the squares beside the cliff, which drags them down. SARSA basically learns "walking near the edge is dangerous for someone who sometimes slips" and takes the longer, safer route along the top
- Q-learning uses $\max_a Q(S_{t+1}, a)$, so it only ever asks what the best next move would be. It never accounts for the possibility of a slip, so the squares beside the cliff keep their high values and it happily takes the shortest path along the edge
- SARSA plans around its own mistakes, while Q-learning plans as if it will never make one.
- My plots track episode length, not reward, so they can't show the cost of Q-learning's riskier path during training: with ε = 0.1, it should fall off the cliff more often than SARSA. Plotting reward per episode is a natural next step.

### Analogy

two people learn to cross a icy ledge while sometimes stumbling at random. One hugs the edge and has the shortest route but keeps falling in during practice. The other takes the wider path and rarely falls. The edge-hugger has the better *route*, but the wider-path person had the better *practice runs*.

So "Q-learning has worse online reward" means: while training, it scores lower on average than SARSA, because its riskier path plus exploration causes more cliff falls.