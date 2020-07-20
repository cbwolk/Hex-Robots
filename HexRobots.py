# Colton Wolk
# Computational Geometry Lab, Tufts University
# Last Modified: 06.21.2020

# HexRobots.py
# This program implements two classes (Module and HexGrid) which represent
# the connected hex robot. The HexGrid class performs actions such as finding
# the levels, cut vertices, and possible pivoting positions (under construction).

import math
import ast
import numpy as np
from queue import Queue
from numpy import inf

from matplotlib import pyplot as plt
from matplotlib.patches import RegularPolygon
import seaborn as sns
plt.style.use('seaborn') # pretty matplotlib plots

# Offset for neighbor modules
# Adding values to any module gives its 6 neighbors
# TODO: MAKE MORE CLEAR WHAT THIS IS ###
# SE, NE, N, NW, SW, S
DIR_OFFSET = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
allPivots = []

# NOTE: unit and module (lowercase) are used interchangably

# A Module is an individual unit of the robot,
# similar to a vertex in a graph
class Module:
    # Module is represented by axial coordinates q,r
    # q,r corresponds to x,z in cube coordinates
    def __init__(self, q, r):
        self.q = q
        self.r = r

    # Customize hash and equality for dictionary/list
    def __hash__(self):
        return hash((self.q, self.r))

    def __eq__(self, other):
        if type(self) is type(other):
            return self.q == other.q and self.r == other.r
        return NotImplemented

    # Print out point nicely
    def __str__(self):
        return '(' + str(self.q) + ',' + str(self.r) + ')'

    # TO BE REMOVED, replace with method in hexgrid
    def getModule(self, grid, a, b):
        return grid.get(hash(a, b))

    def get_q(self):
        return self.q

    def get_r(self):
        return self.r

# This class only stores the list of connected modules and their levels
# Essentially, it just stores the whole robot. Maybe rename to "Robot" to
# make it more clear we are NOT just dealing with some n x n grid
class HexGrid:
    def __init__(self, modules=set(), levels=1):
        self.modules = modules
        self.levels = levels

    def __eq__(self, other):
        # TODO: consider levels in equality
        if type(self) is type(other):
            return self.modules == other.modules
        return NotImplemented

    # Print points in a nice list
    def __str__(self):
        retStr = ''
        notFirst = False
        # Only add a comma if there is another point after
        for unit in self.modules:
            if notFirst:
                retStr = retStr + ', '
            retStr = retStr + '(' + str(unit.get_q()) + ',' + str(unit.get_r()) + ')'
            notFirst = True
        return retStr

    # TO BE REMOVED, replace with getModule()
    def inGrid(self, a, b):
        return Module(a, b) in self.modules

    # Return module if it exists in the graph
    def getModule(self, a, b, graph=None):
        if graph is None:
            graph = self.modules

        unit = Module(a, b)
        if unit in graph:
            return unit
        return None

    def getAllModules(self):
        return self.modules

    # Add a module to the robot
    # TODO: verify it is connected before adding
    def addModule(self, unit):
        self.modules.add(unit)

    # Return a list of neighbors of given module
    def getNeighbors(self, unit, graph=None):
        if graph is None:
            graph = self.modules

        neighbors = []

        # Add offsets to given module to get each neighbor
        # Only add to list if it is an actual non-empty module in the robot
        for d in DIR_OFFSET:
            neighbor = self.getModule(unit.get_q() + d[0], unit.get_r() + d[1], graph)
            if neighbor is not None:
                neighbors.append(neighbor)
        return neighbors

    # Get lowest module in given graph in terms of y pixel coordinate
    def getLowest(self, startUnit=None, levelModules=None):
        if levelModules is None:
            levelModules = self.modules

        currBest = None
        currBestNum = -inf

        # Do math to convert from x, z cub coordinates to y pixel coordinate
        # Then find the lowest y value and record
        for unit in levelModules:
            # loopNum = 2. * np.sin(np.radians(60)) * (-2 * unit.get_r() - unit.get_q()) / 3.
            loopNum = math.sqrt(3) / 2 * unit.get_q() + math.sqrt(3) * unit.get_r()

            if loopNum > currBestNum:
                currBestNum = loopNum
                currBest = unit
            elif loopNum == currBestNum and startUnit is not None:
                if self.shortestPathLength(startUnit, unit) < self.shortestPathLength(startUnit, currBest):
                    currBestNum = loopNum
                    currBest = unit
        return currBest, currBestNum

    # Get highest module in terms of y pixel coordinate
    def getHighest(self, startUnit=None, levelModules=None):
        if levelModules is None:
            levelModules = self.modules

        currBest = None
        currBestNum = inf

        # Do math to convert from x, z cub coordinates to y pixel coordinate
        # Then find the highest y value and record
        for unit in levelModules:
            # loopNum = 2. * np.sin(np.radians(60)) * (-2 * unit.get_r() - unit.get_q()) / 3.
            loopNum = math.sqrt(3) / 2 * unit.get_q() + math.sqrt(3) * unit.get_r()

            if loopNum < currBestNum:
                currBestNum = loopNum
                currBest = unit
            elif loopNum == currBestNum and startUnit is not None:
                if self.shortestPathLength(startUnit, unit) < self.shortestPathLength(startUnit, currBest):
                    currBestNum = loopNum
                    currBest = unit
        return currBest, currBestNum

    # Get shortest path length
    def shortestPathLength(self, start, goal, graph=None):
        if graph is None:
            graph = self.modules

        moduleQueue = Queue()
        moduleQueue.put(start)

        # Only insert first module and set its "visited" value to true
        visited = {start: True}
        dist = {start: 0}

        # Modified BFS loop; it breaks when the given "end" node is dequeued
        while not moduleQueue.empty():
            current = moduleQueue.get()

            # Stop BFS when goal node reached
            if current == goal:
                return dist[goal]

            for nextModule in self.getNeighbors(current, graph):
                if nextModule not in visited:
                    dist[nextModule] = dist[current] + 1
                    moduleQueue.put(nextModule)
                    visited[nextModule] = True

        return dist[goal]

    # Get leftmost module in terms of the "q" coordinate (equivalent to x in pixel)
    def getLeftmost(self, levelModules=None):
        if levelModules is None:
            levelModules = self.modules

        currBest = None
        currBestNum = inf

        # Then find the leftmost q value and record
        for unit in levelModules:
            loopNum = unit.get_q()

            if loopNum < currBestNum:
                currBestNum = loopNum
                currBest = unit
        return currBest, currBestNum

    # Get rightmost module in terms of the "q" coordinate (equivalent to x in pixel)
    def getRightmost(self, levelModules=None):
        if levelModules is None:
            levelModules = self.modules

        currBest = None
        currBestNum = -inf

        # Then find the rightmost q value and record
        for unit in levelModules:
            loopNum = unit.get_q()

            if loopNum > currBestNum:
                currBestNum = loopNum
                currBest = unit
        return currBest, currBestNum

    # Run DFS on given graph, return visited list
    # visited[] has ALL nodes as keys; values initially set to False
    def DFS(self, start, graph=None):
        if graph is None:
            graph = self.modules

        moduleStack = [start]

        # Add each unit to dictionary and set "visited" value to false
        visited = {unit: False for unit in graph}
        visited[start] = True

        # Standard DFS loop; get next from queue and look at neighbors
        while moduleStack:
            current = moduleStack.pop()
            for nextModule in self.getNeighbors(current, graph):
                if nextModule not in visited:
                    moduleStack.append(nextModule)
                    visited[nextModule] = True
        return visited

    # Run modified BFS on given graph in range [start, end)
    # visited[] ONLY includes nodes which are visited
    # Thus, values are always true for nodes in visited[]
    def BFS(self, start, end, graph=None):
        if graph is None:
            graph = self.modules

        moduleQueue = Queue()
        moduleQueue.put(start)

        # Only insert first module and set its "visited" value to true
        visited = {start: True}

        # Modified BFS loop; it breaks when the given "end" node is dequeued
        while not moduleQueue.empty():
            current = moduleQueue.get()

            # Stop BFS when given node reached
            if current == end:
                visited.pop(current)
                return visited

            for nextModule in self.getNeighbors(current, graph):
                if nextModule not in visited:
                    moduleQueue.put(nextModule)
                    visited[nextModule] = True
        return visited

    # Return a list of levels in the graph
    # STEP 1: Find highest and lowest module in current graph
    # STEP 2: Test if we have a cut vertex. If not, mark units in level and continue.
    # STEP 3: Make a copy of graph, do a virtual flip, and repeat from STEP 1.
    def getLevels(self):
        currModules = self.modules.copy()
        levels = []

        # The virtual flip. state 1 is original orientation, 0 is flipped
        state = 1
        start = next(iter(currModules))
        while len(currModules) != 0:

            # Assign level start and end values
            # Note: the "end" value is not included in the current level
            if state == 1:
                start = self.getLowest(start, currModules)[0]
                end = self.getHighest(start, currModules)[0]
            else:
                end = self.getLowest(start, currModules)[0]
                start = self.getHighest(start, currModules)[0]

            # For testing levels
            print(self.canMove(start, end, currModules))
            print('start: ' + str(start) + ' end: ' + str(end))

            # Check if this is a cut vertex
            # If so, stop and assign the rest the same level
            if self.canMove(start, end, currModules):
                thisLev = []
                for unit in currModules:
                    thisLev.append(unit)
                levels.append(thisLev)
                break

            # Get the list vertices from [start, end) by doing modified BFS
            currLev = self.BFS(start, end, currModules)
            levels.append(currLev)

            # Now that we've identified the level, delete those modules
            # from the graph copy to make it easier
            for key in currLev.keys():
                currModules.remove(key)

            # Flip the virtual orientation
            if state == 1:
                state = 0
            elif state == 0:
                state = 1

        return levels

    # Tests whether a single node is a cut vertex
    # Remove node and test if the graph is still connected
    def canMove(self, start, end=None, graph=None):
        if graph is None:
            graph = self.modules

        newGraph = graph.copy()
        if end is not None:
            newGraph.remove(end)
        visited = self.DFS(start, newGraph)

        # Test if each node was visited in DFS
        # If not then the graph must not be connected
        for val in visited.values():
            if not val:
                return False
        return True

    # NOT TESTED, meant to find all empty neighbors
    def getAdjSpaces(self, graph=None):
        if graph is None:
            graph = self.modules

        adjSpaces = []
        # Get all neighbors equal to none (empty spaces)
        for unit in graph:
            for d in DIR_OFFSET:
                emptyNeighbor = self.getModule(unit.get_q() + d[0], unit.get_r() + d[1], graph)
                if emptyNeighbor is None:
                    adjSpaces.append(emptyNeighbor)
        return adjSpaces


    # NOT YET IMPLEMENTED
    # IDEA:   Find where innermost non-cut vertex can move
    # STEP 1: Make a new graph consisting of ALL empty spaces adjacent
    #         to any node in G, inlcuding but not limited to outer shell
    # STEP 2: Attempt to traverse new graph from given vertex, mark
    #         vertices which can be accessed by both moves. Do this by
    #         testing different possibilities at current position of module.
    def getPivotingOptions(self, unit, originalUnit=None, graph=None, pivotList=[]):
        if graph is None:
            graph = self.modules
        if originalUnit is None:
            originalUnit = unit

        print("\nThe unit:      ", unit)

        # adjSpaces = self.getAdjSpaces(graph)

        ##### RESTRICTED MOVE #####
        # if not self.canMove(unit):
        #     return

        hexNeighbors = self.getNeighbors(unit, graph)

        allAdjPos = []

        # Add offsets to given module to get each neighbor
        # Only add to list if it is an actual non-empty module in the robot
        for d in DIR_OFFSET:
            neighbor = self.getModule(unit.get_q() + d[0], unit.get_r() + d[1], graph)
            allAdjPos.append(neighbor)

        for i in range(3):
            if allAdjPos[i] is not None and allAdjPos[i + 3] is not None:
                return

        curr = allAdjPos[0]

        # pL1 = [[], [], [], [], [], []]
        # pL2 = [[], [], [], [], [], []]

        for i in range(6):
            if allAdjPos[i] is not None:
                curr = allAdjPos[i]
                critical1 = allAdjPos[(i + 1) % 6]
                critical2 = allAdjPos[(i + 2) % 6]
                critical3 = allAdjPos[(i + 3) % 6]
                critical4 = allAdjPos[(i + 4) % 6]
                critical5 = allAdjPos[(i + 5) % 6]

                print("ADJACENT: ", (allAdjPos[(i + 1) % 6]), " ", (allAdjPos[(i + 2) % 6]), " ",
                      (allAdjPos[(i + 3) % 6]), " ", (allAdjPos[(i + 4) % 6]), " ", (allAdjPos[(i + 5) % 6]))

                criticalLeft = False
                criticalRight = False

                if critical3 is None and critical4 is None and critical5 is None:
                    pivotModuleL = Module(curr.get_q() + DIR_OFFSET[(i + 4) % 6][0], curr.get_r() + DIR_OFFSET[(i + 4) % 6][1])

                    if pivotModuleL not in allPivots and pivotModuleL not in self.modules:
                        print("Critical Left: ", pivotModuleL)
                        criticalLeft = True
                        allPivots.append(pivotModuleL)
                        # pivotList.append(pivotModule)
                        newGraphL = graph.copy()
                        newGraphL.remove(unit)
                        newGraphL.add(pivotModuleL)

                if critical3 is None and critical2 is None and critical1 is None:
                    pivotModuleR = Module(curr.get_q() + DIR_OFFSET[(i + 2) % 6][0], curr.get_r() + DIR_OFFSET[(i + 2) % 6][1])

                    if pivotModuleR not in allPivots and pivotModuleR not in self.modules:
                        print("Critical Right:", pivotModuleR)
                        criticalRight = True
                        allPivots.append(pivotModuleR)
                        # pivotList.append(pivotModule)
                        newGraphR = graph.copy()
                        newGraphR.remove(unit)
                        newGraphR.add(pivotModuleR)

                if criticalLeft:
                    self.getPivotingOptions(pivotModuleL, originalUnit, newGraphL, allPivots)

                if criticalRight:
                    self.getPivotingOptions(pivotModuleR, originalUnit, newGraphR, allPivots)

        if originalUnit not in allPivots:
            allPivots.append(originalUnit)

        # for i in range(6):
        #     result = []
        #     if pL1[i] or pL2[i]:
        #         result += pL1[i] + pL2[i]
        # if result:
        #     return result
        # curr = current free module
        # n = curr's neighbors
        # Let n's neighboring grid positions be labeled A, B, C, D, E, F in some radial ordering
        # There must be at least one non-empty module, WLOG let this be C
        # Then there exists two hex grid spaces adjacent to both v and each one of its non-empty neighbors, including C
        # if one of these spaces is empty (WLOG, let this be adjacent to v and C)
        #    AND two spaces directly before/after C are empty (WLOG, let these be D and E)
        #    *** AND there is a free space adjacent to the future location of v and directly across from C ***
        #    THEN v can perform a restricted move and pivot around C

        ##### MONKEY MOVE #####
        # SAME as above, except remove line: *** AND ... C ***s

    # Roughly based on code from:
    # https://stackoverflow.com/questions/46525981/how-to-plot-x-y-z-coordinates-in-the-shape-of-a-hexagonal-grid
    def visualize(self, graph=[], pivotList=[]):
        if not graph:
            graph.append(self.modules)

        # List of nearly all matplotlib colors
        colorList = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        colors = []
        count = 0

        # Set a color for each level and assign to each hexagon to be printed
        for lev in graph:
            print('Level ' + str(count))
            for unit in lev:
                print(unit)
                colors.append(colorList[count % 7])
            count = count + 1

        # Convert hex to pixel coordinates
        hcoord = [unit.get_q() for lev in graph for unit in lev]
        vcoord = [2. * np.sin(np.radians(60)) * (-2 * unit.get_r() - unit.get_q()) / 3. for lev in graph for unit in
                  lev]

        fig, ax = plt.subplots(1)
        ax.set_aspect('equal')

        # Print each hexagon with proper specifications
        for x, y, c in zip(hcoord, vcoord, colors):
            hex = RegularPolygon((x, y), numVertices=6, radius=2. / 3.,
                                 orientation=np.radians(30), facecolor=c, alpha=0.2, edgecolor='k')
            ax.add_patch(hex)

        lowYUnit = self.getLowest()[0]
        highYUnit = self.getHighest()[0]
        lowXUnit = self.getLeftmost()[0]
        highXUnit = self.getRightmost()[0]

        allQ = [lowYUnit.get_q(), highYUnit.get_q(), lowXUnit.get_q(), highXUnit.get_q()]
        allR = [lowYUnit.get_r(), highYUnit.get_r(), lowXUnit.get_r(), highXUnit.get_r()]
        maxQ = max(allQ)
        minQ = min(allQ)
        maxR = max(allR)
        minR = min(allR)

        print(maxR)
        ax.set(xlim=(minQ - 2, maxQ + 2), ylim=(min(vcoord) - 2, max(vcoord) + 2))

        empty_hcoord = []
        empty_vcoord = []

        # Find coordinates of empty hexagons for background
        for q in np.arange(minQ - 10, maxQ + 10):
            for r in np.arange(minR - 10, maxR + 10):
                if self.getModule(q,r) is None:
                    empty_hcoord.append(q)
                    empty_vcoord.append(2. * np.sin(np.radians(60)) * (-2 * r - q) / 3.)

        # Draw empty hexagons for background
        for x, y in zip(empty_hcoord, empty_vcoord):
            hex = RegularPolygon((x, y), numVertices=6, radius=2. / 3.,
                                 orientation=np.radians(30), facecolor='w', alpha=0.2, edgecolor='k')
            ax.add_patch(hex)

        scatter_hcoord = []
        scatter_vcoord = []

        # Draw dots in positions where unit can pivot
        if allPivots:
            for unit in allPivots:
                scatter_hcoord.append(unit.get_q())
                scatter_vcoord.append(2. * np.sin(np.radians(60)) * (-2 * unit.get_r() - unit.get_q()) / 3.)

        ax.scatter(scatter_hcoord, scatter_vcoord, alpha=0.5)
        plt.show()

    # implement hopcroft-tarjan cut vertex alg
    def isCutVertex(self):
        return None

    def convertHexTiler(self, inp=None):
        if inp is None:
            inp = input("Enter hex tiler grid: ")

        inp = inp.split('[', 1)[1]
        # inp = '[' + str(inp)
        inp = inp.rsplit(']', 1)[0]
        # inp = str(inp) +']'

        inp = list(ast.literal_eval(inp))
        inp = [[i for i in row if i != ''] for row in inp]

        print(inp)

        for idxr, row in enumerate(inp):
            for idxi, item in enumerate(row):
                if item != '?':
                    q = idxr
                    r = idxi - (idxr + (idxr & 1)) / 2
                    self.addModule(Module(q, r))

# test levels functionality with preset modules
def main():
    # Create an easy test grid
    # hg = HexGrid({Module(-2, 5), Module(-2, 4), Module(-2, 3), Module(-2, 2), Module(-2, 1),
    #                Module(-1, 0), Module(0, 0), Module(1, 0), Module(1, 1), Module(2, 1),
    #                Module(3, 1), Module(4, 1), Module(5, 0), Module(6, -1), Module(7, -1), Module(8,-1)})

    # hg = HexGrid({Module(0, 0), Module(1, 0), Module(0, 1)})

    # hg = HexGrid({Module(2, 0)})
    hg = HexGrid()
    hg.convertHexTiler()
    # Get the levels in the grpah and print them in different colors
    # hg = HexGrid({Module(2, 0)})
    hg.getPivotingOptions(Module(12, -1))
    hg.visualize(hg.getLevels())
    # hg = HexGrid({Module(2, 0), Module(3, 0)})
    # hg.visualize(hg.getLevels())

if __name__ == '__main__': main()


# goal: take a robot and move along a path