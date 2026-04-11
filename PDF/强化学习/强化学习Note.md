#### 基础概念
![](assets/强化学习Note/file-20260411105133753.png)
- State：智能体相对于环境的状态，$s_1,\cdots,s_9$ 代表状态，其中$s_i = \{x_i,y_i\}$ 。
([Book-all-in-one](PDF/强化学习/Book-all-in-one.pdf#page=15&selection=43,0,48,1&color=note))
> The first concept to be introduced is the state, which describes the agent’s status with respect to the environment.
- State Space：将所有状态连接到一起得到状态空间，$S = \{s_1,\cdots,s_9\}$ 集合代表状态空间。
([Book-all-in-one](PDF/强化学习/Book-all-in-one.pdf#page=15&selection=63,29,78,1&color=note))
> The set of all the states is called the state space, denoted as S = {s1, . . . , s9}.
![](assets/强化学习Note/file-20260411105443333.png)
- Actions：每个状态智能体可能采取的行动，$a_1,\cdots,a_5$ 代表智能体在当前状态的行动。
> ([Book-all-in-one](PDF/强化学习/Book-all-in-one.pdf#page=15&selection=79,0,85,63&color=note))
> For each state, the agent can take five possible actions: moving upward, moving rightward, moving downward, moving leftward, and staying still.
- Action Spaces：将所有动作放在一起，$\mathcal{A} = \{a_1,\cdots,a_5\}$。
> ([Book-all-in-one](PDF/强化学习/Book-all-in-one.pdf#page=15&selection=94,37,111,1&color=note))
> he set of all actions is called the action space, denoted as A = {a1, . . . , a5}.

对于复杂样例，不能通过表格形式描述所有状态转移，此时引入条件概率进行描述：
$$
p (s_i|s_1,a_2)
$$
$s_1$采用行为$a_2$后到达各种状态的概率，其总和为1
![](assets/强化学习Note/file-20260411110940538.png)
- Policy：策略告诉智能体在每种状态采用的策略，如，告诉智能体$\pi(a_i|s1)$在$s_1$的状态下，采取$a_i$的概率是多少。
> ([Book-all-in-one](PDF/强化学习/Book-all-in-one.pdf#page=17&selection=351,0,357,25&color=note))
> A policy tells the agent which actions to take at every state. 
