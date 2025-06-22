import numpy as np

GRID_SIZE=4
TERMINAL_STATES = [(0,0),(3,3)]
ACTIONS=['up','down','left','right']
THETA = 0.001
STATES=[(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3),(2,0),(2,1),(2,2),(2,3),(3,0),(3,1),(3,2),(3,3),(4,1)]
V=np.zeros((GRID_SIZE+1,GRID_SIZE))

def next_state(state,action):
    i, j =state
    if state in TERMINAL_STATES:
        return state
    
    if state == (4, 1):
        if action == 'up': return (3, 1)
        elif action == 'left': return (3, 0)
        elif action == 'right': return (3, 2)
        else: return (4, 1)

    if state==(3,1) and action=='down': return(4,1)

    if action=='up': return (max(i-1,0),j)
    elif action =='down': return (min(i+1,3),j)
    elif action =='left': return (i,max(j-1,0))
    else: return (i,min(3,j+1))

def policy_evaluation():
    global V
    iteration=0

    while True:
        delta=0
        new_V=np.zeros((GRID_SIZE+1,GRID_SIZE))
        for k in range(len(STATES)):
                i,j=STATES[k]
                if(i,j) in TERMINAL_STATES:
                    continue
                v=0
                for action in ACTIONS:
                    next_i,next_j= next_state((i,j),action)
                    v+=0.25*(-1+V[next_i,next_j])
                new_V[i,j]=v
                delta = max(delta, abs(v-V[i,j]))
        V=new_V.copy()
        iteration+=1
        print(f"Iteration  {iteration}:")
        print(np.round(V,1))
        print()
        if delta<THETA:
            break

policy_evaluation()
print("Final value function:")
print(np.round(V, 1))