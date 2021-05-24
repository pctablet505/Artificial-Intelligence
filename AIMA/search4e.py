from collections import defaultdict, deque, Counter
from itertools import combinations
import sys
import math
import heapq
import random
import matplotlib.pyplot as plt
import copy


# %matplotlib inline


class Problem(object):
    """The abstract class for a formal problem. A new domain subcasses this,
    overriding `action` and `results`, and perhaps other method.
    The default heuristic is 0 and default action cost is 1 for all states.
    When you create an instance of a subclass, specify `initial`, and `goal`
    states (or give `is_goal` method) and perhaps other keyword args for the
    subclass."""

    def __init__(self, initial=None, goal=None, **kwds):
        self.__dict__.update(initial=initial, goal=goal, **kwds)

    def actions(self, state):
        raise NotImplementedError

    def result(self, state, action):
        raise NotImplementedError

    def is_goal(self, state):
        """Returns true if given state is the goal state"""
        return state == self.goal

    def action_cost(self, s, a, s1):
        """Return the cost for taking action a from state s and getting to
        state s1."""
        return 1

    def h(self, node):
        """Heuristic function for problem, by default set to 0."""
        return 0

    def __str__(self):
        return '{} ({!r}, {!r})'.format(
            type(self).__name__, self.initial, self.goal)


class Node:
    """A Node in a search tree."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.__dict__.update(state=state, parent=parent,
                             action=action, path_cost=path_cost)

    def __repr__(self):
        return '<{}>'.format(self.state)

    def __len__(self):
        return 0 if self.parent is None else (1 + len(self.parent))

    def __lt__(self, other):
        return self.path_cost < other.path_cost


# Indicates an algorithm couldn't find a solution.
failure = Node('failure', path_cost=math.inf)
# indicates iterative deepening search was cut off.
cutoff = Node('cutoff', path_cost=math.inf)


def expand(problem, node):
    """Expand a node, generating the children nodes."""
    s = node.state
    for action in problem.actions(s):
        s1 = problem.result(s, action)
        cost = node.path_cost + problem.action_cost(s, action, s1)
        yield Node(s1, node, action, cost)


def path_actions(node):
    """The sequence of actions to get to this node."""
    if node.parent is None:
        return []
    return path_actions(node.parent) + [node.action]


def path_states(node):
    """The sequence of states to get to this node."""
    if node in (cutoff, failure, None):
        return []
    return path_states(node.parent) + [node.state]


FIFOQueue = deque
LIFOQueue = list


class PriorityQueue:
    """A queue in which the item with minimum f(item) is always popped first."""

    def __init__(self, items=(), key=lambda x: x):
        self.key = key
        self.items = []
        for item in items:
            self.add(item)

    def add(self, item):
        """Push items in PriorityQueue with appropriate ordering."""
        pair = (self.key(item), item)
        heapq.heappush(self.items, pair)

    def pop(self):
        """Pop and return the item with min f(item) value."""
        if len(self.items):
            return heapq.heappop(self.items)[1]

        raise IndexError('Trying to pop from empty PriorityQueue')

    def top(self):
        """Returns the minimum element in the priority queue."""
        if len(self.items):
            return self.items[0][1]
        raise Exception("Priority Queue is Empty")

    def __len__(self):
        return len(self.items)


def best_first_search(problem, f):
    """Search nodes with minimum f(node) value first."""
    node = Node(problem.initial)
    frontier = PriorityQueue([node], key=f)
    reached = {problem.initial: node}
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        for child in expand(problem, node):
            s = child.state
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                frontier.add(child)
    return failure


def best_first_tree_search(problem, f):
    """A version of best_first_search without `reached` table."""
    frontier = PriorityQueue([Node(problem.initial)], key=f)
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        for child in expand(problem, node):
            if not is_cycle(child):
                frontier.add(child)
    return failure


def g(node):
    """Return the cost to reach the node from initial state"""
    return node.path_cost


def astar_search(problem, h=None):
    """Search nodes with minimum f(n) = g(n) + h(n)."""
    h = h or problem.h
    return best_first_search(problem, f=lambda n: g(n) + h(n))


def astar_tree_search(problem, h=None):
    """Search nodes with minimum f(n) = g(n) + h(n),
    with no `reached` table."""
    h = h or problem.h
    return best_first_tree_search(problem, f=lambda n: g(n) + h(n))


def weighted_astar_search(problem, h=None, weight=1.4):
    """Search nodes with minimum f(n) = g(n) + weight * h(n)."""
    h = h or problem.h
    return best_first_search(problem, f=lambda n: g(n) + weight * h(n))


def greedy_bfs(problem, h=None):
    """Search nodes with minimum h(n)."""
    h = h or problem.h
    return best_first_search(problem, f=h)


def uniform_cost_search(problem):
    """Search nodes with minimum path cost first."""
    return best_first_search(problem, f=g)


def breadth_first_bfs(problem):
    """Search shallowest nodes in the search tree first; using best-first."""
    return best_first_search(problem, f=len)


def depth_first_bfs(problem):
    """Search deepest nodes in the search tree first; using best-first."""
    return best_first_search(problem, f=lambda n: -len(n))


def is_cycle(node, k=30):
    """Does this node form a cycle of length k or less?"""

    def find_cycle(ancestor, k):
        return (ancestor is not None and k > 0 and
                (ancestor.state == node.state or find_cycle(ancestor.parent, k - 1)))

    return find_cycle(node.parent, k)


# _____________________________________________________________________________

def breadth_first_search(problem):
    """Search shallowest nodes in the search tree first."""
    node = Node(problem.initial)
    if problem.is_goal(problem.initial):
        return node
    frontier = FIFOQueue([node])
    reached = {problem.initial}
    while frontier:
        node = frontier.pop()
        for child in expand(problem, node):
            s = child.state
            if problem.is_goal(s):
                return child
            if s not in reached:
                reached.add(s)
                frontier.appendleft(child)
    return failure


def iterative_deepening_search(problem):
    """Do depth-limited search with increasing depth limits.
    returns a node or failure."""
    for limit in range(1, sys.maxsize):
        result = depth_limited_search(problem, limit)
        if result != cutoff:
            return result


def depth_limited_search(problem, limit=10):
    """Depth first search with depth limited, if goal is 
    not found within the depth limit returns cutoff."""
    frontier = LIFOQueue([Node(problem.initial)])
    result = failure
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        elif len(node) >= limit:
            result = cutoff
        elif not is_cycle(node):
            for child in expand(problem, node):
                frontier.append(child)
    return result


def depth_first_recursive_search(problem, node=None):
    """Recursive version of depth first search."""
    if node is None:
        node = Node(problem.initial)
    if problem.is_goal(node.state):
        return node
    elif is_cycle(node):
        return failure
    else:
        for child in expand(problem, node):
            result = depth_first_recursive_search(problem, child)
            if result:
                return result
        return failure


def recursive_best_first_search(problem, h=None):
    """Returns a solution or failure"""
    h = h or problem.h

    def RBFS(problem, node, f_limit):
        """Returns a solution or failure, and a new f-cost limit"""
        if problem.is_goal(node.state):
            return node, 0
        successors = list(expand(problem, node))
        if not successors:
            return failure, math.inf
        for s in successors:
            # update f with value from previous search
            s.f = max(s.path_cost + h(s), node.f)
        while True:
            successors.sort(key=lambda x: x.f)
            best = successors[0]
            if best.f > f_limit:
                return failure, best.f
            if len(successors) > 1:
                alternative = successors[1].f
            else:
                alternative = math.inf
            result, best.f = RBFS(problem, best, min(f_limit, alternative))
            if result != failure:
                return result, best.f

    node = Node(problem.initial)
    node.f = h(node)
    solution, fvalue = RBFS(problem, node, math.inf)

    return solution


def bidirectional_best_first_search(problem_f, f_f, problem_b, f_b, terminated):
    """
    Searches in 2 directions, from initial to goal and from goal to initial.
    it has forward and backward problems and separate evaluation function
    for each problem.

    Parameters
    ----------
    problem_f :forward Problem
    f_f :heuristic for forward problem.
    problem_b : backward Problem
    f_b : heuristic for backward problem.
    terminated : if search is over or not

    Returns
    -------
    solution node, or failure

    """
    node_f = Node(problem_f.initial)
    node_b = Node(problem_f.initial)
    frontier_f = PriorityQueue([node_f], key=f_f)
    frontier_b = PriorityQueue([node_b], key=f_b)
    reached_f = {node_f.state: node_f}
    reached_b = {node_b.state: node_b}
    solution = failure

    while frontier_f and frontier_b and not terminated(solution, frontier_f, frontier_b):
        def S1(node, f):
            return str(int(f(node))) + ' ' + str(path_states(node))

        print('Bi:', S1(frontier_f.top(), f_f), S1(frontier_b.top(), f_b))
        if f_f(frontier_f.top()) < f_b(frontier_b.top()):
            solution = proceed('f', problem_f, frontier_f,
                               reached_f, reached_b, solution)
        else:
            solution = proceed('b', problem_b, frontier_b,
                               reached_b, reached_f, solution)
    return solution


def inverse_problem(problem):
    if isinstance(problem, CountCalls):
        return CountCalls(inverse_problem(problem._object))
    else:
        inv = copy.copy(problem)
        inv.initial, inv.goal = inv.goal, inv.initial
        return inv


def bidirectional_uniform_cost_search(problem_f):
    def terminated(solution, frontier_f, frontier_b):
        n_f, n_b = frontier_f.top(), frontier_b.top()
        return g(n_f) + g(n_b) > g(solution)

    return bidirectional_best_first_search(problem_f, g, inverse_problem(problem_f), g, terminated)


def bidirectional_astar_search(problem_f):
    def terminated(solution, frontier_f, frontier_b):
        n_f, n_b = frontier_f.top(), frontier_b.top()
        return g(n_f) + g(n_b) > g(solution)

    problem_b = inverse_problem(problem_f)
    return bidirectional_best_first_search(
        problem_f, lambda n: g(n) + problem_f.h(n),
        problem_b, lambda n: g(n) + problem_b.h(n),
        terminated)


def proceed(direction, problem, frontier, reached, reached2, solution):
    node = frontier.pop()
    for child in expand(problem, node):
        s = child.state
        print('proceed', direction, S(child))
        if s not in reached or child.path_cost < reached[s].path_cost:
            frontier.add(child)
            reached[s] = child
            if s in reached2:  # Frontiers collide; solution found
                solution2 = (join_nodes(child, reached2[s]) if direction == 'f'
                             else join_nodes(reached2[s], child))
                if solution2.path_cost < solution.path_cost:
                    solution = solution2
    return solution


S = path_states


# A-S-R + B-P-R => A-S-R-P + B-P


def join_nodes(nf, nb):
    """Join the reverse of the backward node `nb` to the forward node `nf`."""
    # print('join', S(nf), S(nb))
    join = nf
    while nb.parent is not None:
        cost = join.path_cost + nb.path_cost - nb.parent.path_cost
        join = Node(nb.parent.state, join, nb.action, cost)
        nb = nb.parent
        # print(' now join', S(join), 'with nb', S(nb), 'parent', S(nb.parent))

    return join


# A, B = uniform_cost_search(r1), uniform_cost_search(r2)
# path_states(A), path_states(B)
# path_states(join_nodes(A, B))


class RouteProblem(Problem):
    """A problem to find a route between locations on a `Map`.
    Create a problem with `RouteProblem(start,goal,map=Map(...))`.
    States are the vertices in the Map graph; actions are destination states."""

    def actions(self, state):
        """The places neighboring `state`."""
        return self.map.neighbors[state]

    def result(self, state, action):
        """Go to the `action` place, if the map says that is possible."""
        return action if action in self.map.neighbors[state] else state

    def action_cost(self, s, action, s1):
        """The distance (cost) to go from s to s1."""
        return self.map.distances[s, s1]

    def h(self, node):
        """Straight-line distance between state and the goal."""
        locs = self.map.locations
        return straight_line_distance(locs[node.state], locs[self.goal])


def straight_line_distance(A, B):
    """Straight-line distance between two points."""
    return sum(abs(a - b) ** 2 for (a, b) in zip(A, B)) ** 0.5


class Map:
    """ A map of places in a 2D world: a graph with vertexes and links between them.
    In `Map(links,locations)`, `links` can be either [(v1,v2),...] pairs,
    or a {(v1,v2): distance...} dict. Optional `locations` can be {v1: (x, y)}
    If `directed=False` then for every (v1, v2) link, we add a (v2,v1) link."""

    def __init__(self, links, locations=None, directed=False):
        if not hasattr(links, 'items'):  # Distances are 1 by default
            links = {link: 1 for link in links}
        if not directed:
            for v1, v2 in list(links):
                links[v2, v1] = links[v1, v2]
        self.distances = links
        self.neighbors = multimap(links)
        self.locations = locations or defaultdict(lambda: (0, 0))


def multimap(pairs):
    """Given (key, val) pairs, make a dict of {key: [val,...]}."""
    result = defaultdict(list)
    for key, val in pairs:
        result[key].append(val)
    return result


# Some specific RoutProblems

romania_links = {('Oradea', 'Zerind'): 71,
                 ('Oradea', 'Sibiu'): 151,
                 ('Arad', 'Zerind'): 75,
                 ('Arad', 'Sibiu'): 140,
                 ('Arad', 'Timisoara'): 118,
                 ('Lugoj', 'Timisoara'): 111,
                 ('Lugoj', 'Mehadia'): 70,
                 ('Drobeta', 'Mehadia'): 75,
                 ('Craiova', 'Drobeta'): 120,
                 ('Craiova', 'Rimnicu Vilcea'): 146,
                 ('Craiova', 'Pitesti'): 138,
                 ('Rimnicu Vilcea', 'Sibiu'): 80,
                 ('Fagaras', 'Sibiu'): 99,
                 ('Bucharest', 'Fagaras'): 211,
                 ('Bucharest', 'Pitesti'): 101,
                 ('Bucharest', 'Giurgiu'): 90,
                 ('Bucharest', 'Urziceni'): 85,
                 ('Hirsova', 'Urziceni'): 98,
                 ('Eforie', 'Hirsova'): 86,
                 ('Urziceni', 'Vaslui'): 142,
                 ('Iasi', 'Vaslui'): 92,
                 ('Iasi', 'Neamt'): 87,
                 ('Pitesti', 'Rimnicu Vilcea'): 97}

romania_locations = {'Arad': (76, 497),
                     'Bucharest': (400, 327),
                     'Craiova': (246, 285),
                     'Drobeta': (160, 296),
                     'Eforie': (558, 294),
                     'Fagaras': (285, 460),
                     'Giurgiu': (368, 257),
                     'Hirsova': (548, 355),
                     'Iasi': (488, 535),
                     'Lugoj': (162, 379),
                     'Mehadia': (160, 343),
                     'Neamt': (407, 561),
                     'Oradea': (117, 580),
                     'Pitesti': (311, 372),
                     'Rimnicu Vilcea': (227, 412),
                     'Sibiu': (187, 463),
                     'Timisoara': (83, 414),
                     'Urziceni': (471, 363),
                     'Vaslui': (535, 473),
                     'Zerind': (92, 539)}
romania = Map(romania_links, romania_locations)

r0 = RouteProblem('Arad', 'Arad', map=romania)
r1 = RouteProblem('Arad', 'Bucharest', map=romania)
r2 = RouteProblem('Neamt', 'Lugoj', map=romania)
r3 = RouteProblem('Eforie', 'Timisoara', map=romania)
r4 = RouteProblem('Oradea', 'Mehadia', map=romania)


# print(path_states(uniform_cost_search(r1))) # Lowest-cost path from Arab to Bucharest


class GridProblem(Problem):
    """Finding a path on a 2D grid with obstacles. Obstacles are (x, y) cells."""

    def __init__(self, initial=(15, 30), goal=(130, 30), obstacles=(), **kwds):
        super().__init__(initial=initial, goal=goal,
                         obstacles=set(obstacles) - {initial, goal}, **kwds)

    directions = [(-1, -1), (0, 1), (1, -1),
                  (-1, 0), (1, 0),
                  (-1, 1), (0, 1), (1, 1)]

    def action_cost(self, s, action, s1):
        return straight_line_distance(s, s1)

    def h(self, node):
        return straight_line_distance(node.state, self.goal)

    def result(self, state, action):
        """Both states and actions are represented by (x, y) pairs."""
        return action if action not in self.obstacles else state

    def actions(self, state):
        """You can move cell in any of `directions` to a non-obstacle cell."""
        x, y = state
        return {(x + dx, y + dy) for (dx, dy) in self.directions} - self.obstacles


class EraticVacuum(Problem):
    def actions(self, state):
        return ['suck', 'forward', 'backward']

    def results(self, state, action):
        return self.table[action][state]

    table = dict(suck={1: {5, 7}, 2: {4, 8}, 3: {7}, 4: {2, 4}, 5: {1, 5}, 6: {8}, 7: {3, 7}, 8: {6, 8}},
                 forward={1: {2}, 2: {2}, 3: {4}, 4: {
                     4}, 5: {6}, 6: {6}, 7: {8}, 8: {8}},
                 backward={1: {1}, 2: {1}, 3: {3}, 4: {3}, 5: {5}, 6: {5}, 7: {7}, 8: {7}})


# Some grid routing problems

# The following can be used to create obstacles:

def random_lines(X=range(15, 130), Y=range(60), N=150, lengths=range(6, 12)):
    """The set of cells in N random lines of the given lengths."""
    result = set()
    for _ in range(N):
        x, y = random.choice(X), random.choice(Y)
        dx, dy = random.choice(((0, 1), (1, 0)))
        result |= line(x, y, dx, dy, random.choice(lengths))
    return result


def line(x, y, dx, dy, length):
    """A line of `length` cells starting at (x, y) and going in (dx, dy) direction."""
    return {(x + i * dx, y + i * dy) for i in range(length)}


random.seed(42)  # To make this cell reproducible

frame = line(-10, 20, 0, 1, 20) | line(150, 20, 0, 1, 20)
cpu = line(102, 44, -1, 0, 15) | line(102, 20, -
1, 0, 20) | line(102, 44, 0, -1, 24)

d1 = GridProblem(obstacles=random_lines(N=100) | frame)
d2 = GridProblem(obstacles=random_lines(N=150) | frame)
d3 = GridProblem(obstacles=random_lines(N=200) | frame)
d4 = GridProblem(obstacles=random_lines(N=250) | frame)
d5 = GridProblem(obstacles=random_lines(N=300) | frame)
d6 = GridProblem(obstacles=cpu | frame)
d7 = GridProblem(obstacles=cpu | frame)


class EightPuzzle(Problem):
    """The problem of sliding tiles numbered from 1 to 8 on a 3x3 board,
    where one of the squares is a blank, trying to reach a goal configuration.
    A board state is represented as a tuple of length 9, where the element at
    index i represents the tile number at index i, or 0 if the empty square,
    e.g. the goal:
        1 2 3
        4 5 6 ==> (1,2,3,4,5,6,7,8,0)
        7 8 _
    """

    def __init__(self, initial, goal=(0, 1, 2, 3, 4, 5, 6, 7, 8)):
        assert inversions(initial) % 2 == inversions(goal) % 2  # Parity check
        self.initial, self.goal = initial, goal

    def actions(self, state):
        """The indices of the squares that the blank can move to."""
        """ 0 1 2
            3 4 5
            6,7,8
            """
        moves = ((1, 3), (0, 2, 4), (1, 5),
                 (0, 4, 6), (1, 3, 5, 7), (2, 4, 8),
                 (3, 7), (4, 6, 8), (7, 5))
        blank = state.index(0)
        return moves[blank]

    def result(self, state, action):
        """Swap the blamk with square numbered `action`."""
        s = list(state)
        blank = state.index(0)
        s[action], s[blank] = s[blank], s[action]
        return tuple(s)

    def h1(self, node):
        """The misplaced tiles heuristic."""
        return hamming_distance(node.state, self.goal)

    def h2(self, node):
        """The Manhattan heuristic."""
        X = (0, 1, 2, 0, 1, 2, 0, 1, 2)
        Y = (0, 0, 0, 1, 1, 1, 2, 2, 2)
        return sum(abs(X[s] - X[g]) + abs(Y[s] - Y[g])
                   for (s, g) in zip(node.state, self.goal) if s != 0)

    def h(self, node):
        return self.h2(node)


def hamming_distance(A, B):
    """Number of positions where vectors A and B are different."""
    return sum(a != b for a, b in zip(A, B))


def inversions(board):
    """The number of times a piece is a smaller number than a following piece."""
    return sum((a > b and a != 0 and b != 0) for a, b in combinations(board, 2))


def board8(board, fmt=(3 * '{} {} {}\n')):
    """A string representing an 8-puzzle board."""
    return fmt.format(*board).replace('0', '_')


class Board(defaultdict):
    empty = '.'
    off = '#'

    def __init__(self, board=None, width=8, height=8, to_move=None, **kwds):
        if board is not None:
            self.update(board)
            self.width, self.height = (board.width, board.height)
        else:
            self.width, self.height = (width, height)

    def __missing__(self, key):
        x, y = key
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return self.off
        else:
            return self.empty

    def __repr__(self):
        def row(y):
            return ' '.join(self[x, y] for x in range(self.width))

        return '\n'.join(row(y) for y in range(self.height))

    def __hash__(self):
        return hash(tuple(sorted((self.items())))) + hash(self.to_move)


# Some specific EightPuzzle problems


e1 = EightPuzzle((1, 4, 2, 0, 7, 5, 3, 6, 8))
e2 = EightPuzzle((1, 2, 3, 4, 5, 6, 7, 8, 0))
e3 = EightPuzzle((4, 0, 2, 5, 1, 3, 7, 8, 6))
e4 = EightPuzzle((7, 2, 4, 5, 0, 6, 8, 3, 1))
e5 = EightPuzzle((8, 6, 7, 2, 5, 4, 3, 0, 1))


# Solve an 8 puzzle problem and print out each state

# for s in path_states(astar_search(e5)):
#     print(board8(s))

class PourProblem(Problem):
    """Problem about pouring water between jugs to achieve some water level.
    Each state is a tuples of water levels. In the initialization, also provide a tuple of
    jug sizes, e.g. `PourProblem(initial=(0, 0), goal=4, sizes=(5, 3))`,
    which means two jugs of sizes 5 and 3, initially both empty, with the goal
    of getting a level of 4 in either jug."""

    def actions(self, state):
        """The actions executable in this state."""
        jugs = range(len(state))
        return ([('Fill', i) for i in jugs if state[i] < self.sizes[i]] +
                [('Dump', i) for i in jugs if state[i]] +
                [('Pour', i, j) for i in jugs if state[i] for j in jugs if i != j])

    def result(self, state, action):
        """The state that results from executing this action in this state."""
        result = list(state)
        act, i, *_ = action
        if act == 'Fill':  # Fill i to capacity
            result[i] = self.sizes[i]
        elif act == 'Dump':  # Empty i
            result[i] = 0
        elif act == 'Pour':  # Pour from i into j
            j = action[2]
            amount = min(state[i], self.sizes[j] - state[j])
            result[i] -= amount
            result[j] += amount
        return tuple(result)

    def is_goal(self, state):
        """True if goal level is in any one of the jugs."""
        return self.goal in state


class GreenPourProblem(PourProblem):
    """A `PourProblem` in which the cost is the amount of water used."""

    def action_cost(self, s, action, s1):
        """The cost is the amount of water used."""
        act, i, *_ = action
        return self.sizes[i] - s[i] if act == 'Fill' else 0


# Some specific PourProblems

p1 = PourProblem((1, 1, 1), 13, sizes=(2, 16, 32))
p2 = PourProblem((0, 0, 0), 21, sizes=(8, 11, 31))
p3 = PourProblem((0, 0), 8, sizes=(7, 9))
p4 = PourProblem((0, 0, 0), 21, sizes=(8, 11, 31))
p5 = PourProblem((0, 0), 4, sizes=(3, 5))

g1 = GreenPourProblem((1, 1, 1), 13, sizes=(2, 16, 32))
g2 = GreenPourProblem((0, 0, 0), 21, sizes=(8, 11, 31))
g3 = GreenPourProblem((0, 0), 8, sizes=(7, 9))
g4 = GreenPourProblem((0, 0, 0), 21, sizes=(8, 11, 31))
g5 = GreenPourProblem((0, 0), 4, sizes=(3, 5))


# Solve the PourProblem of getting 13 in some jug, and show the actions and states
# soln = breadth_first_search(p1)
# path_actions(soln), path_states(soln)

class PancakeProblem(Problem):
    """A PancakeProblem the goal is always `tuple(range(1, n+1))`, where the
    initial state is a permutation of `range(1, n+1)`. An act is the `i` of
    the top `i` pancakes that will be flipped."""

    def __init__(self, initial):
        self.initial, self.goal = tuple(initial), tuple(sorted(initial))

    def actions(self, state):
        return range(2, len(state) + 1)

    def result(self, state, i):
        return state[:i][::-1] + state[i:]

    def h(self, node):
        """The gap heuristic."""
        s = node.state
        return sum(abs(s[i] - s[i - 1]) > 1 for i in range(1, len(s)))


c0 = PancakeProblem((2, 1, 4, 6, 3, 5))
c1 = PancakeProblem((4, 6, 2, 5, 1, 3))
c2 = PancakeProblem((1, 3, 7, 5, 2, 6, 4))
c3 = PancakeProblem((1, 7, 2, 6, 3, 5, 4))
c4 = PancakeProblem((1, 3, 5, 7, 9, 2, 4, 6, 8))


# Solve a pancake problem
# path_states(astar_search(c0))

class JumpingPuzzle(Problem):
    """Try to exchange L and R by moving one ahead or hopping two ahead."""

    def __init__(self, N=2):
        self.initial = N * 'L' + '.' + N * 'R'
        self.goal = self.initial[::-1]

    def actions(self, state):
        """Find all possible move or hop moves."""
        idxs = range(len(state))
        return ({(i, i + 1) for i in idxs if state[i:i + 2] == 'L.'}
                | {(i, i + 2) for i in idxs if state[i:i + 3] == 'LR.'}
                | {(i + 1, i) for i in idxs if state[i:i + 2] == '.R'}
                | {(i + 2, i) for i in idxs if state[i:i + 3] == '.LR'})

    def result(self, state, action):
        """An action (i, j) means swap the pieces at positions i and j."""
        i, j = action
        result = list(state)
        result[i], result[j] = state[j], state[i]
        return ''.join(result)

    def h(self, node):
        return hamming_distance(node.state, self.goal)


# j3 = JumpingPuzzle(N=3)
# j9 = JumpingPuzzle(N=9)
# print(path_states(astar_search(j3)))


# ______________________________________________________________________________


class RouteProblem(Problem):
    """The problem of moving the Hybrid Wumpus Agent from one place to other."""

    def __init__(self, initial, goal, allowed, dimrow):
        """Define goal state and initialize a problem."""
        super().__init__(initial, goal)
        self.dimrow = dimrow
        self.goal = goal
        self.allowed = allowed

    def actions(self, state):
        """Return the actions that can be executed in the given state.
        The result would be a list, since there are only three possible actions
        in any given state of the environment."""

        possible_actions = ['Forward', 'TurnLeft', 'TurnRight']
        x, y = state.get_location()
        orientation = state.get_orientation()

        # Prevent Bumps
        if x == 1 and orientation == 'LEFT':
            if 'Forward' in possible_actions:
                possible_actions.remove('Forward')
        if y == 1 and orientation == 'DOWN':
            if 'Forward' in possible_actions:
                possible_actions.remove('Forward')
        if x == self.dimrow and orientation == 'RIGHT':
            if 'Forward' in possible_actions:
                possible_actions.remove('Forward')
        if y == self.dimrow and orientation == 'UP':
            if 'Forward' in possible_actions:
                possible_actions.remove('Forward')

        return possible_actions

    def result(self, state, action):
        """Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state."""

        x, y = state.get_location()
        proposed_loc = []

        # Move Forward
        if action == 'Forward':
            if state.get_orientateion() == 'UP':
                proposed_loc = [x, y + 1]
            elif state.get_orientateion() == 'DOWN':
                proposed_loc = [x, y - 1]
            elif state.get_orientateion() == 'LEFT':
                proposed_loc = [x - 1, y]
            elif state.get_orientateion() == 'RIGHT':
                proposed_loc = [x + 1, y]
            else:
                raise Exception('InvalidOrientation')

        # Rotate counter-clockwise
        elif action == 'TurnLeft':
            if state.get_orientateion() == 'UP':
                state.set_orientation('LEFT')
            elif state.get_orientateion() == 'DOWN':
                state.set_orientation('RIGHT')
            elif state.get_orientateion() == 'LEFT':
                state.set_orientation('DOWN')
            elif state.get_orientateion() == 'RIGHT':
                state.set_orientation('UP')
            else:
                raise Exception('InvalidOrientation')

        # Rotate clockwise
        elif action == 'TurnRight':
            if state.get_orientateion() == 'UP':
                state.set_orientation('RIGHT')
            elif state.get_orientateion() == 'DOWN':
                state.set_orientation('LEFT')
            elif state.get_orientateion() == 'LEFT':
                state.set_orientation('UP')
            elif state.get_orientateion() == 'RIGHT':
                state.set_orientation('DOWN')
            else:
                raise Exception('InvalidOrientation')

        if proposed_loc in self.allowed:
            state.set_location(proposed_loc[0], [proposed_loc[1]])

        return state

    def is_goal(self, state):
        """Given a state, return True if state is goal state or False otherwise."""
        return state.get_location() == tuple(self.goal)

    def h(self, node):
        """Return the heuristic value for a given state."""

        # Manhattan Heuristic Function
        x1, y1 = node.state.get_location()
        x2, y2 = self.goal
        return abs(x2 - x1) + abs(y2 - y1)


# ______________________________________________________________________________


class CountCalls:
    """Delegate all attribute gets to the object, and count them in `._count`."""

    def __init__(self, obj):
        self._object = obj
        self._counts = Counter()

    def __getattr__(self, attr):
        """Delegate to the original object, after incrementing a counter."""
        self._counts[attr] += 1
        return getattr(self._object, attr)


def report(searchers, problems, verbose=True):
    for searcher in searchers:
        print(searcher.__name__ + ':')
        total_counts = Counter()
        for p in problems:
            prob = CountCalls(p)
            soln = searcher(prob)
            counts = prob._counts
            counts.update(actions=len(soln), cost=soln.path_cost)
            total_counts += counts
            if verbose:
                report_counts(counts, str(p)[:40])
        report_counts(total_counts, 'TOTAL\n')


def report_counts(counts, name):
    """Print one line of counts report."""
    print('{:9,d} nodes|{:9,d} goal |{:5.0f} cost |{:8,d} actions | {}'.format(
        counts['result'], counts['is_goal'], counts['cost'], counts['actions'], name))


def astar_misplaced_tiles(problem):
    return astar_search(problem, h=problem.h1)


# report([breadth_first_search, astar_misplaced_tiles, astar_search], [e1, e2, e3, e4, e5])
# report([astar_search,uniform_cost_search],[c1,c2,c3,c4])
# report([astar_search, astar_tree_search], [e1, e2, e3, e4, r1, r2, r3, r4])

def extra_weighted_astar_search(problem):
    return weighted_astar_search(problem, weight=2)


# report([greedy_bfs, extra_weighted_astar_search, weighted_astar_search, astar_search, uniform_cost_search],
#     [r0, r1, r2, r3, r4, e1, d1, d2, j3, e2, d3, d4, d6, d7, e3, e4])

# report((astar_search, uniform_cost_search,  breadth_first_search, breadth_first_bfs,
#         iterative_deepening_search, depth_limited_search, greedy_bfs,
#         weighted_astar_search, extra_weighted_astar_search),
#        (p1, g1, p2, g2, p3, g3, p4, g4, r0, r1, r2, r3, r4, e1))

# for plotting
def best_first_search(problem, f):
    """Search nodes with minimum f(node) value first."""
    global reached  # <<<<< Only change here
    node = Node(problem.initial)
    frontier = PriorityQueue([node], key=f)
    reached = {problem.initial: node}
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        for child in expand(problem, node):
            s = child.state
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                frontier.add(child)
    return failure


def plot_grid_problem(grid, solution, reached=(), title='Search', show=True):
    """Use matplotlib to plot the grid, obstacles, solution, and reached."""
    reached = list(reached)
    plt.figure(figsize=(16, 10))
    plt.axis('off')
    plt.axis('equal')
    plt.scatter(*transpose(grid.obstacles), marker='s', color='darkgrey')
    plt.scatter(*transpose(reached), 1 ** 2, marker='.', color='blue')
    plt.scatter(*transpose(path_states(solution)), marker='s', color='blue')
    plt.scatter(*transpose([grid.initial]), 9 ** 2, marker='D', color='green')
    plt.scatter(*transpose([grid.goal]), 9 ** 2, marker='8', color='red')
    plt.show()
    print('{} {} search: {:.1f} path cost, {:,d} states reached'
          .format(' ' * 10, title, solution.path_cost, len(reached)))


def plots(grid, weights=(1.4, 2)):
    """Plot the results of 4 heuristic search algorithms for this grid."""
    solution = astar_search(grid)
    plot_grid_problem(grid, solution, reached, 'A* search')
    for weight in weights:
        solution = weighted_astar_search(grid, weight=weight)
        plot_grid_problem(grid, solution, reached, '(b) Weighted ({}) A* search'.format(weight))
    solution = greedy_bfs(grid)
    plot_grid_problem(grid, solution, reached, 'Greedy best-first search')


def transpose(matrix):
    return list(zip(*matrix))
