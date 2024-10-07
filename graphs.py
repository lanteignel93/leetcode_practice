from collections import defaultdict, deque

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

            return ans

        return dfs(0)

    def canVisitAllRooms(self, rooms: list[list[int]]) -> bool:
        seen = {0}

        def dfs(node):
            for neighbor in rooms[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    dfs(neighbor)

        dfs(0)
        return len(seen) == len(rooms)

    def validPath(
        self, n: int, edges: list[list[int]], source: int, destination: int
    ) -> bool:
        graph = collections.defaultdict(list)
        for a, b in edges:
            graph[a].append(b)
            graph[b].append(a)

        seen = [False] * n

        def dfs(curr_node):
            if curr_node == destination:
                return True
            if not seen[curr_node]:
                seen[curr_node] = True
                for next_node in graph[curr_node]:
                    if dfs(next_node):
                        return True
            return False

        return dfs(source)

    def countComponents(self, n: int, edges: list[list[int]]) -> int:
        graphs = defaultdict(list)
        for x, y in edges:
            graphs[x].append(y)
            graphs[y].append(x)

        visited = set()
        count = 0

        def dfs(index: int) -> None:
            if index in visited:
                return
            visited.add(index)
            for neighbor in graphs[index]:
                dfs(neighbor)

        for index in range(n):
            if index not in visited:
                dfs(index)
                count += 1
        return count

    def maxAreaOfIsland(self, grid: list[list[int]]) -> int:
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        seen = set()

        m = len(grid)
        n = len(grid[0])

        def valid(row, col):
            return 0 <= row < m and 0 <= col < n and grid[row][col] == 1

        self.area = 0
        ans = 0

        def dfs(row, col):
            for dx, dy in directions:
                next_row, next_col = row + dy, col + dx
                if valid(next_row, next_col) and (next_row, next_col) not in seen:
                    self.area += 1
                    seen.add((next_row, next_col))
                    dfs(next_row, next_col)

        for row in range(m):
            for col in range(n):
                if grid[row][col] == 1 and (row, col) not in seen:
                    self.area += 1
                    seen.add((row, col))
                    dfs(row, col)
                    ans = max(ans, self.area)
                    self.area = 0

        return ans

    def reachableNodes(
        self, n: int, edges: list[list[int]], restricted: list[int]
    ) -> int:
        graphs = defaultdict(list)
        for x, y in edges:
            graphs[x].append(y)
            graphs[y].append(x)

        seen = [False] * n

        for node in restricted:
            seen[node] = True

        self.ans = 0

        def dfs(curr_node):
            self.ans += 1
            seen[curr_node] = True

            for next_node in graphs[curr_node]:
                if not seen[next_node]:
                    dfs(next_node)

        dfs(0)
        return self.ans


class TreeNode:
    def __init__(self, val: int):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None


class GraphBFS:
    def shortestPathBinaryMatrix(self, grid: list[list[int]]) -> int:
        if grid[0][0] == 1:
            return -1

        n = len(grid)
        seen = {(0, 0)}
        queue = deque([(0, 0, 1)])
        directions = [
            (0, 1),
            (1, 0),
            (1, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (0, -1),
            (-1, 0),
        ]

        def valid(row, col):
            return 0 <= row < n and 0 <= col < n and grid[row][col] == 0

        while queue:
            row, col, steps = queue.popleft()
            if (row, col) == (n - 1, n - 1):
                return steps

            for dx, dy in directions:
                next_row, next_col = row + dy, col + dx
                if valid(next_row, next_col) and (next_row, next_col) not in seen:
                    seen.add((next_row, next_col))
                    queue.append((next_row, next_col, steps + 1))

        return -1

    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> list[int]:
        def dfs(node, parent):
            if not node:
                return

            node.parent = parent
            dfs(node.left, node)
            dfs(node.right, node)

        dfs(root, None)
        queue = deque([target])
        seen = {target}
        distance = 0

        while queue and distance < k:
            current_length = len(queue)
            for _ in range(current_length):
                node = queue.popleft()
                for neighbor in [node.left, node.right, node.parent]:
                    if neighbor and neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)

            distance += 1

        return [node.val for node in queue]

    def updateMatrix(self, mat: list[list[int]]) -> list[list[int]]:
        m = len(mat)
        n = len(mat[0])
        queue = deque()
        seen = set()

        def valid(row, col):
            return 0 <= row < n and 0 <= col < m and mat[row][col] == 1

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for row in range(m):
            for col in range(n):
                if mat[row][col] == 0:
                    queue.append((row, col, 1))
                    seen.add((row, col))

        while queue:
            row, col, steps = queue.popleft()

            for dx, dy in directions:
                next_row, next_col = row + dx, col + dy
                if valid(next_row, next_col) and (next_row, next_col) not in seen:
                    seen.add((next_row, next_col))
                    queue.append((next_row, next_col, steps + 1))
                    mat[next_row, next_col] = steps

        return mat

    def shortestPath(self, grid: list[list[int]], k: int) -> int:
        m = len(grid)
        n = len(grid[0])
        queue = deque([(0, 0, k, 0)])
        seen = {(0, 0, k)}
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        def valid(row, col):
            return 0 <= row < m and 0 <= col < n

        while queue:
            row, col, remain, steps = queue.popleft()
            if row == m - 1 and col == n - 1:
                return steps

            for dx, dy in directions:
                next_row, next_col = row + dy, col + dx
                if valid(next_row, next_col):
                    if grid[next_row, next_col] == 0:
                        if (next_row, next_col, remain) not in seen:
                            seen.add((next_row, next_col, remain))
                            queue.append((next_row, next_col, remain, steps + 1))

                    elif remain and (next_row, next_col, remain - 1) not in seen:
                        seen.add((next_row, next_col, remain - 1))
                        queue.append((next_row, next_col, remain - 1, steps + 1))

        return -1

    def snakesAndLadders(self, board: list[list[int]]) -> int:
        n = len(board)
        cells = [None] * (n**2 + 1)
        label = 1
        columns = list(range(0, n))
        for row in range(n - 1, -1, -1):
            for column in columns:
                cells[label] = (row, column)
                label += 1
            columns.reverse()
        dist = [-1] * (n * n + 1)
        q = deque([1])
        dist[1] = 0
        while q:
            curr = q.popleft()
            for next in range(curr + 1, min(curr + 6, n**2) + 1):
                row, column = cells[next]
                destination = board[row][column] if board[row][column] != -1 else next
                if dist[destination] == -1:
                    dist[destination] = dist[curr] + 1
                    q.append(destination)
        return dist[n * n]
