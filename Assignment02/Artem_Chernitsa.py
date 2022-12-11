def complete_task(infile, outfile):
    from typing import List, Tuple, Dict
    from functools import lru_cache

    import numpy as np
    import random

    import torch.nn as nn
    import torch

    from matplotlib import pyplot as plt


    STATE_SIZE = 9
    ACTIONS_SIZE = 4
    HIDDEN_LAYER_SIZE = 2 ** STATE_SIZE

    class DQN(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Linear(STATE_SIZE, HIDDEN_LAYER_SIZE)
            self.hidden_1 = nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE // 2)
            self.hidden_2 = nn.Linear(HIDDEN_LAYER_SIZE // 2, HIDDEN_LAYER_SIZE // 2)
            self.output = nn.Linear(HIDDEN_LAYER_SIZE // 2, ACTIONS_SIZE)

            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(dim=0)
        

        def forward(self, x):
            x = self.input(x)
            x = self.relu(self.hidden_1(x))
            x = self.relu(self.hidden_2(x))
            x = self.output(x)
            
            return self.softmax(x)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DQNModel = DQN().to(device)

    class GridWorld:
        
        class Cargo:
            def __init__(self, name: int, coordinates: List[Tuple], world):
                self.name = name
                self.coordinates = coordinates
                self.init_coordinates = self.coordinates[:] # for resetting world
                self.boundaries = self.__get_cargo_boundaries()
                self.world = world
                self.reward = 0
            

            def reset(self):
                """
                Возвращаем груз в исходное положение для обучения в следующей эпохе.
                """
                self.coordinates = self.init_coordinates[:]
                self.boundaries = self.__get_cargo_boundaries()
                self.reward = 0


            def move(self, action):
                """
                Двигаем груз в одном из четырёх направлений, если возможно.
                """
                actions = {
                    "L": (0, -1),
                    "R": (0, +1),
                    "U": (-1, 0),
                    "D": (+1, 0)
                }

                new_coords = []
                for cell in self.coordinates:
                    r, c = cell
                    dr, dc = actions[action]
                    
                    # если не можем двигаться туда, то остаёмся на месте
                    if self.is_move_possible(action):
                        new_coords.append((r + dr, c + dc))
                    else:
                        return self.coordinates
                
                # обновляем границы груза, чтобы не вычислять их заново за О(n)
                (ur, uc), (dr, dc) = self.boundaries
                ur, uc = ur + actions[action][0], uc + actions[action][1]
                dr, dc = dr + actions[action][0], dc + actions[action][1]
                self.boundaries = (ur, uc), (dr, dc)

                
                self.coordinates = new_coords
                return new_coords


            def is_move_possible(self, action) -> bool:
                actions = {
                    "L": (0, -1),
                    "R": (0, +1),
                    "U": (-1, 0),
                    "D": (+1, 0)
                }
                rd, cd = actions[action]
                cargo = [(r + rd, c + cd) for r, c in self.coordinates]
                
                in_boundaries = all(
                    map(lambda el: 0 <= el[0] < self.world.r and 0 <= el[1] < self.world.c, cargo)
                )
                if not in_boundaries:
                    return False
                
                return True


            def __get_cargo_boundaries(self) -> Tuple:
                """
                Возвращает координаты верхнего левого и правого нижнего углов
                груза, достроенного до прямоугольника.
                """
                ur = min(self.coordinates, key=lambda x: x[0])[0]
                uc = min(self.coordinates, key=lambda x: x[1])[1]
                dr = max(self.coordinates, key=lambda x: x[0])[0]
                dc = max(self.coordinates, key=lambda x: x[1])[1]
                return (ur, uc), (dr, dc)


        def __init__(self, n: int, m: int,
                    default_reward: int,
                    cargos: List[Tuple[int, int, int]],
                    desirable_area: List[Tuple[int, int]]):
            self.r = n # number of rows
            self.c = m # number of columns
            self.board = [(r, c) for r in range(n) for c in range(m)]
            self.cargos = self.__get_cargos(cargos)
            self.desirable_area = desirable_area
            self.default_reward = default_reward


        def reset(self):
            """
            Возвращает все грузы на их исходное местоположение.
            """
            for cargo_name in self.cargos:
                self.cargos[cargo_name].reset()


        def make_step(self) -> Tuple:
            """
            Выбирает груз, который будет передвинут и возвращает состояние
            среды относительно него.
            """
            # for cargo_name, cargo in self.cargos.items():
            #     # если у груза максимальная награда, то его не надо двигать пока,
            #     # надо поискать другой груз, который нужно подвинуть
            #     if self.compute_cargo_reward(cargo_name) != len(cargo.coordinates):
            #         return self.get_state(cargo_name), cargo_name
            
            names = tuple(self.cargos.keys())
            cargo_name = random.choice(names)
            return self.get_state(cargo_name), cargo_name


        def move_cargo(self, name, action):
            self.cargos[name].move(action)
            self.cargos[name].reward -= self.default_reward # даём награду за ход


        def get_state(self, cargo_name) -> Tuple:
            """
            Возвращает кортеж из угловых координат терминальной зоны,
            угловых координат груза (вписанного в прямоугольник),
            награду для этого груза (части в терминальной зоне
            за вычетом пересечений с этим грузом).
            """
            return (
                self.get_desirable_area_boundaries(), 
                self.cargos[cargo_name].boundaries,
                self.compute_cargo_reward(cargo_name)
            )


        def compute_cargo_reward(self, cargo_name):
            """
            Вычисляет награду по количеству вошедших в терминальную зону частей
            груза, за вычетом пересечений с другими грузами. Персечения учитываются
            по всему миру, а не только по терминальной зоне.
            """
            area_lub, area_rdb = self.get_desirable_area_boundaries()
            cells = self.cargos[cargo_name].coordinates
            in_boundaries = sum(map(lambda cell: 
                        area_lub[0] <= cell[0] <= area_rdb[0] and 
                        area_lub[1] <= cell[1] <= area_rdb[1],
                        cells))
            
            intersections = {cell: 0 for cell in cells}
            for name, cargo in self.cargos.items():
                if name == cargo_name:
                    continue
                for cell in cargo.coordinates:
                    if cell in intersections:
                        intersections[cell] += 1
            
            # Несколько пересечений в одной точке суммируются
            return in_boundaries - sum(intersections.values())


        def check_terminate(self) -> bool:
            """
            Проверяет входят ли все грузы в терминальную зону без пересечений.
            """
            area_lub, area_rdb = self.get_desirable_area_boundaries()
            for cargo in self.cargos.values():
                cells = cargo.coordinates
                in_boundaries = all(
                    map(lambda cell: 
                        area_lub[0] <= cell[0] <= area_rdb[0] and 
                        area_lub[1] <= cell[1] <= area_rdb[1],
                        cells)
                )
                if not in_boundaries:
                    return False
            
            intersections = set()
            for cargo in self.cargos.values():
                for cell in cargo.coordinates:
                    if cell not in intersections:
                        intersections.add(cell)
                    else:
                        return False

            return True


        @lru_cache()
        def get_desirable_area_boundaries(self) -> Tuple:
            """
            Возвращает координаты правого верхнего и левого нижнего углов,
            по которым можно уникально определить расположение терминальной зоны.
            """
            return min(self.desirable_area), max(self.desirable_area)


        def __get_cargos(self, cargos) -> Dict[int, Cargo]:
            """
            Создаёт список объектов грузов по именованным координатам.
            """
            names = set()
            for name, *_ in cargos:
                names.add(name)
            
            tf_cargos = {}
            for name in names:
                coordinates = list(
                    map(lambda x: (x[1], x[2]), filter(lambda x: x[0] == name, cargos))
                )
                tf_cargos[name] = (self.Cargo(name, coordinates, self))

            return tf_cargos


        def __str__(self):
            board = np.full((self.r, self.c), '0', dtype=str)
            
            for cell in self.desirable_area:
                board[cell] = 'r'
            
            for name, cargo in self.cargos.items():
                for cell in cargo.coordinates:
                    board[cell] = str(cargo.name)
            
            return str(board)


        def __repr__(self):
            return str(self)


    def flatten_state(state):
        if isinstance(state, tuple):
            for x in state:
                yield from flatten_state(x)
        else:
            yield state


    def load_world(n, m, grid) -> GridWorld:
        cargos = []
        desirable_area = []
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 'r':
                    desirable_area.append((i, j))
                elif grid[i][j] != '0':
                    cargos.append((int(grid[i][j]), i, j))
        
        return GridWorld(n, m, default_reward=0, cargos=cargos, desirable_area=desirable_area)


    def complete_task_(infile, outfile):
        grid = []
        with open(f'{infile}.txt', 'r') as f:
            for line in f:
                grid.append(line.split())
        
        n, m = len(grid), len(grid[0])
        world = load_world(n, m, grid)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DQN().to(device) 
        model.load_state_dict(torch.load('model_v0.pt'))
        model.eval()

        print('Start accomplishment task...')
        print(world)

        acts_count = 0
        res = []
        while not world.check_terminate() and acts_count <= 1000:
            curr_state, cargo_name = world.make_step()
            curr_state = np.array(list(flatten_state(curr_state)))
            act_prob = model(torch.from_numpy(curr_state).float()).data.numpy()
            directions = ['L', 'R', 'U', 'D']
            action = directions[np.argmax(act_prob)]
            # action = np.random.choice(
            #     np.array(['L', 'R', 'U', 'D']),
            #     p=act_prob
            # )
            while len(directions) > 1 and not world.cargos[cargo_name].is_move_possible(action):
                escape = np.argmax(act_prob)
                directions.pop(escape)
                escape_prob = act_prob[escape]
                act_prob = np.array([act_prob[i] for i in range(len(act_prob)) if i != escape])
                act_prob += escape_prob / len(directions)
                # action = directions[np.argmax(act_prob)]
                action = np.random.choice(
                    np.array(directions),
                    p=act_prob
                )
            print(f"{cargo_name} {action}")
            res.append(f"{cargo_name} {action}")
            acts_count += 1
            world.move_cargo(cargo_name, action) # реализуем действие
            
            if len(res) % 100 == 0:
                with open(f'{outfile}.txt', 'w') as f:
                    f.write('\n'.join(res[-100:]))
        
        print(world)
        with open(f'{outfile}.txt', 'w') as f:
            f.write('\n'.join(res[-100:]))

    complete_task_(infile, outfile)

if __name__ == '__main__':
    complete_task('infile', 'outfile')