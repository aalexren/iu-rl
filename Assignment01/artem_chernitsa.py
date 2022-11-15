from typing import List, Tuple
import numpy as np

class GridWorld:        
    def __init__(self, n: int, m: int,
                 gamma: float = 0.9,
                 default_reward: float = -1.,
                 cargo: List[Tuple[int, int]] = [],
                 obstacles: List[Tuple[int, int]] = [],
                 terminate_states: List[Tuple[int, int]] = [],
                 ):
        self.n = n # number of rows
        self.m = m # number of columns
        self.board = [(r, c) for r in range(n) for c in range(m)]
        self.gamma = gamma
        self.default_reward = default_reward
        self.cargo = cargo # coordinates of all cargo's cells
        self.cargo_pivot = self.get_cargo_pivot()
        self.obstacles = obstacles # cooridnates of all obstacles' cells
        # self.states = set(self.board).difference(
        #     set(self.obstacles)
        #     ) | set(self.cargo)
        # Здесь, конечно, прикол. Объясняю, значит,
        # мы можем считать svfs для препятствий, просто никогда
        # не будем туда ходить
        self.states = set(self.board) | set(self.cargo) | set(self.obstacles)
        self.svfs = {state: 0. for state in self.states} # state value funtctions
        self.terminate_states = set(terminate_states)
        self.policy = {cell: '.' for cell in self.board}


    def train_agent(self):
        eps = 10e-3 # epsilon to finish computation

        def svfs_diff(prev, next):
            res = []
            for key, _ in prev.items():
                res.append(abs(prev[key] - next[key]))
            return res

        t = 0
        while True:
            t += 1
            prev_svfs = self.svfs.copy()
            self.update_svfs() # execute term 0
            diff = svfs_diff(prev_svfs, self.svfs)
            if all(map(lambda el: el < eps, diff)):
                break
        
        print(f"Finished at a time step #{t}")
        print(self)
        self.show_svfs()
        self.show_policy()

        def get_path(policy, pivot, terminates):
            """
            Идёт от пивота по указанным policy,
            если два раза и более зашёл в одно и то же
            состояние, значит пути нет.
            """
            res = set()
            path = []
            actions = {
            "L": (0, -1),
            "R": (0, +1),
            "U": (-1, 0),
            "D": (+1, 0)
            }
            while True:
                if policy[pivot] == '.' and pivot not in terminates:
                    return ["No path"]
                if pivot in res:
                    return ["No path"]
                if pivot in terminates:
                    break
                res.add(pivot)

                r, c = pivot
                action = policy[pivot]
                r_d, c_d = actions[action]
                path.append(action)

                pivot = r + r_d, c + c_d
            
            return path
        
        self.path = get_path(self.policy, 
                             self.cargo_pivot,
                             self.terminate_states)


    def update_svfs(self):
        """
        Update state-value functions for current time step t.
        """
        new_svfs = self.svfs.copy()

        for state in self.states:
            if state in self.terminate_states:
                continue
            action_name, svf = self.compute_svf(state)
            new_svfs[state] = svf
            self.policy[state] = action_name

        self.svfs = new_svfs


    def compute_svf(self, state):
        """
        Compute state-value function for current time step t.
        """
        qvfs = []
        states = self.possible_actions(state) # new possible states
        for action_name, to_state in states:
            svf = self.default_reward + self.gamma * self.svfs[to_state]
            qvfs.append((action_name, svf))

        if not qvfs:
            return self.policy[state], self.svfs[state]
        return max(qvfs, key=lambda el: el[1])


    def possible_actions(self, from_state) -> List[Tuple[int, int]]:
        actions = {
            "L": (0, -1),
            "R": (0, +1),
            "U": (-1, 0),
            "D": (+1, 0)
            }
        r, c = from_state
        
        ret = []
        for name, action in actions.items():
            r_d, c_d = action
            to_state = r + r_d, c + c_d
            if self.is_action_possible(to_state):
                ret.append((name, to_state))
        
        return ret


    def is_action_possible(self, to_state) -> bool:
        """
        Надо проверить, не ломается ли фигура, при перемещении
        в данную ячейку, и не выходит ли за границы мира.
        """
        r, c = self.cargo_pivot
        r_prime, c_prime = to_state
        r_d, c_d = r_prime - r, c_prime - c # считаем насколько переместились
        cargo = [(r + r_d, c + c_d) for r, c in self.cargo] # считаем для каждой
                                                            # ячейки груза её 
                                                            # новые координаты

        in_boundaries = all(map(
            lambda el: 0 <= el[0] < self.n and 0 <= el[1] < self.m, cargo)
            )
        if not in_boundaries:
            return False
        
        for cell in cargo:
            # if cell in self.cargo and cell != self.cargo_pivot:
                # return False
            if cell in self.obstacles:
                return False
        
        return True


    def get_cargo_pivot(self):
        # pivot, *_ = sorted(
        #     self.cargo, key=lambda el: (el[0], el[1]), reverse=True
        #     ) # берём максимальную по row и максимальную по col ячейку за опорную
        #       # (как если бы мы за краешек фигуры перенесли её в новую точку)
        pivot_r = sorted(self.cargo)[-1][0]
        pivot_c = sorted(self.cargo, key=lambda el: el[1])[-1][1]
        pivot = pivot_r, pivot_c

        return pivot


    def show_svfs(self):
        grid = np.zeros((self.n, self.m))
        for state, value in self.svfs.items():
            r, c = state
            grid[r, c] = value
        
        print(grid)
    

    def show_policy(self):
        grid = np.empty([self.n, self.m], dtype=str)
        for state, action in self.policy.items():
            r, c = state
            grid[r, c] = action
        
        print(grid)


    def __str__(self):
        grid = np.zeros((self.n, self.m))
        for r, c in self.cargo:
            grid[r, c] = 2
        for r, c in self.obstacles:
            grid[r, c] = 1
        return str(grid)
    

    def __repr__(self):
        return str(self)


def find_path(infile, outfile):
    grid = []
    with open(f"{infile}.txt", 'r') as f:
        for line in f:
            grid.append(line.split())
    
    cargo = []
    obstacles = []
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if grid[r][c] == '1':
                obstacles.append((r, c))
            elif grid[r][c] == '2':
                cargo.append((r, c))
    
    n, m = len(grid), len(grid[0])
    terminate_states = [(n - 1, m - 1)]
    world = GridWorld(n, m, 
                  gamma=0.9,
                  default_reward=-1., 
                  cargo=cargo,
                  obstacles=obstacles,
                  terminate_states=terminate_states)
    
    world.svfs[terminate_states[-1]] = 10
    world.train_agent()
    print(' '.join(world.path))

    with open(f"{outfile}.txt", 'w') as f:
        f.write(' '.join(world.path))

find_path("test_in", "test_out")