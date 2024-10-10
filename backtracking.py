from collections import deque


#   Author: Laurent Lanteigne
#   Date: 2024-10-10
#   source: https://leetcode.com/explore/interview/card/leetcodes-interview-crash-course-data-structures-and-algorithms/711/backtracking/4535/
#
class BackTracking:
    def permute(self, nums: list[int]) -> list[list[int]]:
        ans = []

        def backtrack(curr: list[int]) -> None:
            if len(curr) == len(nums):
                ans.append(curr[:])
                return

            for num in nums:
                if num not in curr:
                    curr.append(num)
                    backtrack(curr)
                    curr.pop()

        backtrack([])
        return ans

    def subset(self, nums: list[int]) -> list[list[int]]:
        ans = []

        def backtrack(curr: list[int], i: int) -> None:
            if i > len(nums):
                return

            ans.append(curr[:])
            for j in range(i, len(nums)):
                curr.append(nums[j])
                backtrack(curr, j + 1)
                curr.pop()

        backtrack([], 0)
        return ans

    def combine(self, n: int, k: int) -> list[list[int]]:
        ans = []

        def backtrack(curr: list[int], i: int) -> None:
            if len(curr) == k:
                return

            for num in range(i, n + 1):
                curr.append(num)
                backtrack(curr, num + 1)
                curr.pop()

        backtrack([], 1)
        return ans

    def allPathsSourceTarget(self, graph: list[list[int]]) -> list[list[int]]:
        target = len(graph) - 1
        ans = []
        path = [0]

        def backtrack(curr_node: int, path: list[int]) -> None:
            # if we reach the target, no need to explore further.
            if curr_node == target:
                ans.append(path[:])
                return
            # explore the neighbor nodes one after another.
            for next_node in graph[curr_node]:
                path.append(next_node)
                backtrack(next_node, path)
                path.pop()

        # kick of the backtracking, starting from the source node (0).
        backtrack(0, path)

        return ans

    def letterCombinations(self, digits: str) -> list[str]:
        if len(digits) == 0:
            return []

        ans = []
        letters = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz",
        }

        def backtrack(index: int, path: list[str]) -> None:
            if len(path) == len(digits):
                ans.append("".join(path))
                return

            possible_letters = letters[digits[index]]
            for letter in possible_letters:
                path.append(letter)
                backtrack(index + 1, path)
                path.pop()

        backtrack(0, [])
        return ans

    def combinationSum(self, candidates: list[int], target: int) -> list[list[int]]:
        ans = []

        def backtrack(path, start, curr):
            if curr == target:
                ans.append(path[:])
                return

            for i in range(start, len(candidates)):
                num = candidates[i]
                if curr + num <= target:
                    path.append(num)
                    backtrack(path, i, curr + num)
                    path.pop()

        backtrack([], 0, 0)
        return ans

    def totalNQueens(self, n: int):
        def backtrack(row: int, diagonals: set, anti_diagonals: set, cols: set):
            # Base case - N queens have been placed
            if row == n:
                return 1

            solutions = 0
            for col in range(n):
                curr_diagonal = row - col
                curr_anti_diagonal = row + col
                # If the queen is not placeable
                if (
                    col in cols
                    or curr_diagonal in diagonals
                    or curr_anti_diagonal in anti_diagonals
                ):
                    continue

                # "Add" the queen to the board
                cols.add(col)
                diagonals.add(curr_diagonal)
                anti_diagonals.add(curr_anti_diagonal)

                # Move on to the next row with the updated board state
                solutions += backtrack(row + 1, diagonals, anti_diagonals, cols)

                # "Remove" the queen from the board since we have already
                # explored all valid paths using the above function call
                cols.remove(col)
                diagonals.remove(curr_diagonal)
                anti_diagonals.remove(curr_anti_diagonal)

            return solutions

        return backtrack(0, set(), set(), set())

    def exist(self, board: list[list[str]], word: str) -> bool:
        def valid(row: int, col: int):
            return 0 <= row < m and 0 <= col < n

        def backtrack(row: int, col: int, i: int, seen: set[tuple[int, int]]) -> bool:
            if i == len(word):
                return True

            for dx, dy in directions:
                next_row, next_col = row + dy, col + dx
                if valid(next_row, next_col) and (next_row, next_col) not in seen:
                    if board[next_row][next_col] == word[i]:
                        seen.add((next_row, next_col))
                        if backtrack(next_row, next_col, i + 1, seen):
                            return True
                        seen.remove((next_row, next_col))

            return False

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        m = len(board)
        n = len(board[0])

        for row in range(m):
            for col in range(n):
                if board[row][col] == word[0] and backtrack(row, col, 1, {(row, col)}):
                    return True

        return False

    def generateParenthesis(self, n: int) -> list[str]:
        def isValid(p_string):
            left_count = 0
            for p in p_string:
                if p == "(":
                    left_count += 1
                else:
                    left_count -= 1

                if left_count < 0:
                    return False

            return left_count == 0

        answer = []
        queue = deque([""])
        while queue:
            cur_string = queue.popleft()

            # If the length of cur_string is 2 * n, add it to `answer` if
            # it is valid.
            if len(cur_string) == 2 * n:
                if isValid(cur_string):
                    answer.append(cur_string)
                continue
            queue.append(cur_string + ")")
            queue.append(cur_string + "(")

        return answer

    def numsSameConsecDiff(self, n: int, k: int) -> list[int]:
        if n == 1:
            return [i for i in range(10)]

        # initialize the queue with candidates for the first level
        queue = [digit for digit in range(1, 10)]

        for level in range(n - 1):
            next_queue = []
            for num in queue:
                tail_digit = num % 10
                # using set() to avoid duplicates when K == 0
                next_digits = set([tail_digit + k, tail_digit - k])

                for next_digit in next_digits:
                    if 0 <= next_digit < 10:
                        new_num = num * 10 + next_digit
                        next_queue.append(new_num)
            # start the next level
            queue = next_queue

        return queue

    def combinationSum3(self, k: int, n: int) -> list[list[int]]:
        results = []

        def backtrack(remain, comb, next_start):
            if remain == 0 and len(comb) == k:
                # make a copy of current combination
                # Otherwise the combination would be reverted in other branch of backtracking.
                results.append(list(comb))
                return
            elif remain < 0 or len(comb) == k:
                # exceed the scope, no need to explore further.
                return

            # Iterate through the reduced list of candidates.
            for i in range(next_start, 9):
                comb.append(i + 1)
                backtrack(remain - i - 1, comb, i + 1)
                # backtrack the current choice
                comb.pop()

        backtrack(n, [], 0)

        return results
