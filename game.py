import random, copy

WIDTH = 8
HEIGHT = 10
MOVES = 25
GEM_COUNT = 4
OBSTACLE_COUNT = 2
OBSTACLE_CHAR = '#'
EMPTY = -1

class Board:
    def __init__(self, rand):
        self._rand = rand
        grid = []
        for _ in range(HEIGHT):
            row = []
            for _ in range(WIDTH):
                row.append(self._rand.randint(0, GEM_COUNT + OBSTACLE_COUNT - 1))
            grid.append(row)
        self._grid = grid
    
    def __str__(self):
        return '\n'.join(
            ''.join((' ' if x == EMPTY else (OBSTACLE_CHAR if x >= GEM_COUNT else chr(ord('a') + x))) for x in row)
            for row in self._grid)
            
    @staticmethod
    def _drop_one(grid, rand):
        any_dropped = False
        for x in range(WIDTH):
            dropping = False
            for y in range(HEIGHT - 1, -1, -1):
                c = grid[y][x]
                dropping = dropping or c == EMPTY
                if dropping and y > 0:
                    grid[y][x] = grid[y - 1][x]
            if dropping:
                grid[0][x] = rand.randint(0, GEM_COUNT + OBSTACLE_COUNT - 1)
            any_dropped = any_dropped or dropping
        return any_dropped
    
    @staticmethod
    def _check_clear_obstacle(x, y, grid, to_clear):
        if x >= 0 and x < WIDTH and y >= 0 and y < HEIGHT and grid[y][x] >= GEM_COUNT:
            to_clear.append((x, y))
    
    @staticmethod
    def _match(grid):
        clear = []
        score = 0

        # Identify matches
        limits = (WIDTH, HEIGHT)
        for dir in [(1, 0), (0, 1)]:
            ortho = (1 - dir[0], 1 - dir[1])
            start = (0, 0)
            while start[0] < limits[0] and start[1] < limits[1]:
                run = 0
                prev = EMPTY
                cur = start
                while cur[0] <= limits[0] and cur[1] <= limits[1]:
                    c = grid[cur[1]][cur[0]] if (cur[0] < limits[0] and cur[1] < limits[1]) else EMPTY
                    if c == prev:
                        run += 1
                    else:
                        if prev < GEM_COUNT and run >= 3:
                            score += (min(run, 5) - 1) * (min(run, 5) + 1)
                            overwrite = (cur[0] - run * ortho[0], cur[1] - run * ortho[1])
                            while overwrite[0] < cur[0] or overwrite[1] < cur[1]:
                                clear.append(overwrite)
                                overwrite = (overwrite[0] + ortho[0], overwrite[1] + ortho[1])
                        prev = c
                        run = 1
                    cur = (cur[0] + ortho[0], cur[1] + ortho[1])
                start = (start[0] + dir[0], start[1] + dir[1])
        
        # Delete
        clearobstacles = []
        for x, y in clear:
            grid[y][x] = EMPTY
            Board._check_clear_obstacle(x + 1, y, grid, clearobstacles)
            Board._check_clear_obstacle(x - 1, y, grid, clearobstacles)
            Board._check_clear_obstacle(x, y + 1, grid, clearobstacles)
            Board._check_clear_obstacle(x, y - 1, grid, clearobstacles)
        
        for x, y in clearobstacles:
            grid[y][x] = EMPTY
    
        return score
    
    @staticmethod
    def _step_impl(grid, rand):
        ''' returns (anything_changed, score_delta) '''
        any_dropped = Board._drop_one(grid, rand)
        if any_dropped:
            return True, 0
        else:
            sdif = Board._match(grid)
            return sdif != 0, sdif
    
    @staticmethod
    def _move_in_place(grid, move):
        '''Returns True if succesful, False otherwise'''
        x, y, vert = move
        if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
            return False
        if vert:
            xp, yp = x, y + 1
        else:
            xp, yp = x + 1, y
        if xp >= WIDTH or yp >= HEIGHT:
            return False
        grid[y][x], grid[yp][xp] = grid[yp][xp], grid[y][x]
        return True
    
    @staticmethod
    def _matches_anything(grid, move):
        gridprime = copy.deepcopy(grid)
        if Board._move_in_place(gridprime, move):
            chg, sdif = Board._step_impl(gridprime, random)
            return sdif > 0
        else:
            return False
    
    def move(self, move):
        return Board._move_in_place(self._grid, move)
    
    def step(self):
        return Board._step_impl(self._grid, self._rand);
    
    def matching_moves(self):
        ret = []
        for x in range(WIDTH):
            for y in range(HEIGHT):
                for d in [False, True]:
                    mv = (x, y, d)
                    if Board._matches_anything(self._grid, mv):
                        ret.append(mv)
        return ret

class GameLogic:
    def __init__(self, seed = None):
        if seed == None:
            seed = random.getrandbits(128)
        rand = random.Random(seed)
        self._moves_left = MOVES
        self._score = 0
        self._board = Board(rand)
        changes = True
        while changes: # settle
            changes, _ = self._board.step()
    
    def score(self):
        return self._score
    
    def is_gameover(self):
        return self._moves_left <= 0
    
    def matching_moves(self):
        return self._board.matching_moves()
    
    def board(self):
        return str(self._board)
    
    def moves_left(self):
        return self._moves_left
    
    def play(self, move):
        interm = []
        sdif = 0
        if not self.is_gameover():
            self._board.move(move)
            changes = True
            while changes:
                interm.append(str(self._board))
                changes, delta = self._board.step()
                sdif += delta
            interm = interm[:-1]
            if sdif == 0: # invalid move
                sdif = -5
            self._score += sdif
        self._moves_left -= 1
        return str(self._board), sdif, self.is_gameover(), interm
