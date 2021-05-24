import sys
import heapq
from collections import defaultdict

class Node:
    def __init__(self, state, parent, action):
        self.state = state
        self.parent = parent
        self.action = action

from random import shuffle, randrange,randint
 
def make_maze(w = 16, h = 8):
    vis = [[0] * w + [1] for _ in range(h)] + [[1] * (w + 1)]
    ver = [["#  "] * w + ['#'] for _ in range(h)] + [[]]
    hor = [["###"] * w + ['#'] for _ in range(h + 1)]
 
    def walk(x, y):
        vis[y][x] = 1
 
        d = [(x - 1, y), (x, y + 1), (x + 1, y), (x, y - 1)]
        shuffle(d)
        for (xx, yy) in d:
            if vis[yy][xx]: continue
            if xx == x: hor[max(y, yy)][x] = "#  "
            if yy == y: ver[y][max(x, xx)] = "   "
            walk(xx, yy)
 
    walk(randrange(w), randrange(h))
 
    s = ""
    for (a, b) in zip(hor, ver):
        s += ''.join(a + ['\n'] + b + ['\n'])
    matrix=[[y for y in x] for x in s.splitlines()]
    matrix[randint(0,h)][randint(0,w)]='A'
    matrix[randint(0,h)][randint(0,w)]='B'
    
    matrix=[''.join(x) for x in matrix]
    s='\n'.join(matrix)
    return s



class StackFrontier:
    def __init__(self):
        self.frontier = []
        self.states = set()

    def add(self, node):
        self.frontier.append(node)
        self.states.add(node.state)

    def contains_state(self, state):
        return state in self.states

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            raise Exception('Empty frontier')
        else:
            node = self.frontier.pop()
            self.states.discard(node.state)
            return node
class ManhatanDistanceFrontier(StackFrontier):
    def __init__(self,goal):
        super().__init__()
        self.goal=goal
        self.keys=defaultdict(list)
    
    def manhatan_distance(self,state):
        return abs(self.goal[0]-state[0])+abs(self.goal[1]-state[1])
    
    def add(self,node,x=0):
        dis=self.manhatan_distance(node.state)
        self.keys[dis].append(node)
        heapq.heappush(self.frontier, dis)
        self.states.add(node.state)
    def remove(self):
        if self.empty():
            raise Exception('Empty frontier')
        else:
            dis = heapq.heappop(self.frontier)
            node=self.keys[dis].pop()
            self.states.discard(node.state)
            return node
    
class A_starFrontier(ManhatanDistanceFrontier):
    def add(self,node,curr_dist):
        dis=self.manhatan_distance(node.state)+curr_dist
        self.keys[dis].append(node)
        heapq.heappush(self.frontier, dis)
        self.states.add(node.state)
        
        

class QueueFrontier(StackFrontier):
    def __init__(self):
        from collections import deque
        self.frontier = deque()
        self.states = set()

    def remove(self):
        if self.empty():
            raise Exception('empty frontier')
        else:
            node = self.frontier.popleft()
            self.states.discard(node.state)
            return node



class Maze():
    def __init__(self, filename):
        # read file and set height and width of maze
        with open(filename, 'r') as f:
            contents = f.read()
        if contents.count('A') != 1:
            raise Exception('Maze must have exactly one start point')
        if contents.count('B') != 1:
            raise Exception('Maze must have exactly one goal')

        # Determine height and width of maze
        contents = contents.splitlines()
        contents = [[x for x in s] for s in contents]
        print(contents)

        self.height = len(contents)
        self.width = len(contents[0])
        self.walls = []

        for i in range(self.height):
            row = []
            for j in range(self.width):
                try:
                    cij = contents[i][j]
                    if cij == 'A':
                        self.start = (i, j)
                        row.append(False)
                    elif cij == 'B':
                        self.goal = (i, j)
                        row.append(False)
                    elif cij == ' ':
                        row.append(False)
                    else:
                        row.append(True)
                except IndexError:
                    row.append(False)
            self.walls.append(row)

        self.solution = None
    
    

    def print(self):
        solution = self.solution[1] if self.solution is not None else None
        print()
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):
                if col:
                    print(' ', end='')
                elif (i, j) == self.start:
                    print('A', end='')
                elif (i, j) == self.goal:
                    print('B', end='')
                elif solution is not None and (i, j) in solution:
                    print('*', end='')
                else:
                    print(' ', end='')
            print()
        print()

    def neighbours(self, state):
        row, col = state
        candidates = [
            ('up', (row - 1, col)),
            ('down', (row + 1, col)),
            ('left', (row, col - 1)),
            ('right', (row, col + 1))
        ]
        result = []
        for action, (r, c) in candidates:
            if 0 <= r < self.height and 0 <= c < self.width and not self.walls[r][c]:
                result.append((action, (r, c)))
        return result

    def outputImage(self, filename, show_solution=True, show_explored=False):
        from PIL import Image, ImageDraw
        cell_size = 50
        cell_border = 2

        # create a blank canvas
        img = Image.new(
            'RGBA',
            (self.width * cell_size, self.height * cell_size),
            'black')
        draw = ImageDraw.Draw(img)
        solution = self.solution[1] if self.solution is not None else None

        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):
                # Walls
                if col:
                    fill = (40, 40, 40)

                # start
                elif (i, j) == self.start:
                    fill = (255, 0, 0)

                # goal
                elif (i, j) == self.goal:
                    fill = (0, 255, 0)

                # solution
                elif solution is not None and show_solution and (i, j) in solution:
                    fill = (220, 235, 113)

                # explored
                elif solution is not None and show_explored and (i, j) in self.explored:
                    fill = (58, 90, 100)

                # empty
                else:
                    fill = (255, 255, 255)

                # draw cell
                draw.rectangle(
                    ([(j * cell_size + cell_border, i * cell_size + cell_border),
                      ((j + 1) * cell_size - cell_border, (i + 1) * cell_size - cell_border)]),
                    fill=fill
                )
        img.save(filename)

    def solve(self,method):
        '''finds a solution to maze, if one exists.'''

        # keep track of number of states explored
        self.num_explored = 0

        # initialize frontier to just starting position
        start = Node(state=self.start, parent=None, action=None)
        
        if method=='astar':
            distance=dict()
            distance[start]=0
            frontier=A_starFrontier(self.goal)
        elif method=='manhatan':
            frontier = ManhatanDistanceFrontier(self.goal)
        elif method=='bfs':
            frontier=QueueFrontier()
        elif method=='dfs':
            frontier=StackFrontier()
        if method in ['manhatan','dfs','bfs']:
            frontier.add(start)
        elif method=='astar':
            frontier.add(start,0)

        # initialize an empty explored set
        self.explored = set()

        while True:
            if frontier.empty():
                raise Exception('no solution')
            node = frontier.remove()
            self.num_explored += 1
            

            # if node is the goal, then we have a solution
            if node.state == self.goal:
                actions = []
                cells = []
                while node.parent is not None:
                    actions.append(node.action)
                    cells.append(node.state)
                    node = node.parent
                actions.reverse()
                cells.reverse()
                self.solution = (actions, cells)
                return

            # mark node as explored
            self.explored.add(node.state)

            # add neighbours to frontier
            for action, state in self.neighbours(node.state):
                if not frontier.contains_state(state) and state not in self.explored:
                    child = Node(state=state, parent=node, action=action)
                    if method=='astar':
                        distance[child]=distance[node]+1
                        frontier.add(child,distance[child]) #for A* frontier
                    elif method in ['bfs','dfs','manhatan']:    
                        frontier.add(child)
            



for i in range(1,10):
    # l=randint(5,50)
    # b=randint(5,50)
    file_name='maze'+str(i)
    # f=open(file_name+'.txt','w')
    # f.write(make_maze(l,b))
    # f.close()
    f=file_name+'.txt'
    maze=Maze(f)
    for method in ['dfs','bfs','manhatan','astar']:
        maze.solve(method)
        maze.outputImage(file_name+method+'.png',True,True)
    
    