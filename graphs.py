from collections import defaultdict

#   Author: Laurent Lanteigne
#   Date: 2024-10-04
#   source: https://leetcode.com/explore/interview/card/leetcodes-interview-crash-course-data-structures-and-algorithms/707/traversals-trees-graphs/4721/


class GraphDFS:
    def findCircleNum(self, isConnected: list[list[int]]) -> int:
        graph = defaultdict(list)
        seen = set()

        def dfs(node):
            for neighbor in graph[node]:
                # the next 2 lines are needed to prevent cycles
                if neighbor not in seen:
                    seen.add(neighbor)
                    dfs(neighbor)

        # build the graph
        n = len(isConnected)
        # traverse upper right triangle only to remove repetition
        for i in range(n):
            for j in range(i + 1, n):
                if isConnected[i][j]:
                    graph[i].append(j)
                    graph[j].append(i)

        ans = 0

        # Go through the the dimension of number of cities
        for i in range(n):
            if i not in seen:
                # add all nodes of a connected component to the set
                ans += 1
                seen.add(i)
                dfs(i)

        return ans

    def numIslands(self, grid: list[list[str]]) -> int:
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        seen = set()

        m = len(grid)
        n = len(grid[0])

        def valid(row, col):
            return 0 <= row < m and 0 <= col < n and grid[row][col] == "1"

        ans = 0

        def dfs(row, col):
            for dx, dy in directions:
                next_row, next_col = row + dy, col + dx
                if valid(next_row, next_col) and (next_row, next_col) not in seen:
                    seen.add((next_row, next_col))
                    dfs(next_row, next_col)

        for row in range(m):
            for col in range(n):
                if grid[row][col] == "1" and (row, col) not in seen:
                    ans += 1
                    seen.add((row, col))
                    dfs(row, col)

        return ans

    def minReorder(self, n: int, connections: list[list[int]]) -> int:
        roads = set()
        seen = set(0)
        graphs = defaultdict(list)

        for x, y in connections:
            graphs[x].append(y)
            graphs[y].append(x)
            roads.add((x, y))

        ans = 0

        def dfs(node):
            for neighbor in graphs[node]:
                if neighbor not in seen:
                    if (node, neighbor) in roads:
                        ans += 1
                    seen.add(neighbor)
                    ans += dfs(ans)

        return dfs(0)
