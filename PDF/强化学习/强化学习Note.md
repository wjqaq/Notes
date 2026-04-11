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
