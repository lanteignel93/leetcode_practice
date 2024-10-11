from functools import lru_cache


#   Author: Laurent Lanteigne
#   Date: 2024-08-05
#   source: https://leetcode.com/explore/learn/card/dynamic-programming/
#
class DynamicProgramming:
    # top_bottom
    def maximumScore(self, nums: list[int], multipliers: list[int]) -> int:
        n, m = len(nums), len(multipliers)

        @lru_cache(2000)
        def dp(i: int, left: int):
            if i == m:
                return 0

            mult = multipliers[i]
            right = n - 1 - (i - left)

            return max(
                mult * nums[left] + dp(i + 1, left + 1),
                mult * nums[right] + dp(i + 1, left),
            )

        return dp(0, 0)

    # bottom_up
    def maximumScore_(self, nums: list[int], multipliers: list[int]) -> int:
        n, m = len(nums), len(multipliers)
        dp = [[0] * (m + 1) for _ in range(n + 1)]

        for i in range(m - 1, -1, -1):
            for left in range(i, -1, -1):
                mult = multipliers[i]
                right = n - 1 - (i - left)
                dp[i][left] = max(
                    mult * nums[left] + dp[i + 1][left + 1],
                    mult * nums[right] + dp[i + 1][left],
                )

        return dp[0][0]

    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        n, m = len(text1), len(text2)

        dp = [[0] * (m + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[n][m]

    def maximalSquare(self, matrix: list[list[str]]) -> int:
        m, n = len(matrix), len(matrix[0])

        dp = [[0] * (n + 1) for _ in range(m + 1)]
        max_sq_len = 0
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if matrix[i - 1][j - 1] == "1":
                    dp[i][j] = (
                        min(min(dp[i][j - 1], dp[i - 1][j]), dp[i - 1][j - 1]) + 1
                    )
                max_sq_len = max(max_sq_len, dp[i][j])

        return max_sq_len**2

    # top_bottom
    def minDifficulty(self, jobDifficulty: list[int], d: int) -> int:
        n = len(jobDifficulty)
        # If we cannot schedule at least one job per day,
        # it is impossible to create a schedule
        if n < d:
            return -1

        hardest_job_remaining = [0] * n
        hardest_job = 0
        for i in range(n - 1, -1, -1):
            hardest_job = max(hardest_job, jobDifficulty[i])
            hardest_job_remaining[i] = hardest_job

        @lru_cache(None)
        def dp(i, day):
            # Base case, it's the last day so we need to finish all the jobs
            if day == d:
                return hardest_job_remaining[i]

            best = float("inf")
            hardest = 0
            # Iterate through the options and choose the best
            for j in range(i, n - (d - day)):  # Leave at least 1 job per remaining day
                hardest = max(hardest, jobDifficulty[j])
                best = min(best, hardest + dp(j + 1, day + 1))  # Recurrence relation

            return best

        return dp(0, 1)

    # bottom_up
    def minDifficulty_(self, jobDifficulty: list[int], d: int) -> int:
        n = len(jobDifficulty)
        # If we cannot schedule at least one job per day,
        # it is impossible to create a schedule
        if n < d:
            return -1

        dp = [[float("inf")] * (d + 1) for _ in range(n)]

        # Set base cases
        dp[-1][d] = jobDifficulty[-1]

        # On the last day, we must schedule all remaining jobs, so dp[i][d]
        # is the maximum difficulty job remaining
        for i in range(n - 2, -1, -1):
            dp[i][d] = max(dp[i + 1][d], jobDifficulty[i])

        for day in range(d - 1, 0, -1):
            for i in range(day - 1, n - (d - day)):
                hardest = 0
                # Iterate through the options and choose the best
                for j in range(i, n - (d - day)):
                    hardest = max(hardest, jobDifficulty[j])
                    # Recurrence relation
                    dp[i][day] = min(dp[i][day], hardest + dp[j + 1][day + 1])

        return dp[0][1]

    def coinChange(self, coins: list[int], amount: int) -> int:
        dp = [amount + 1] * (amount + 1)
        dp[0] = 0

        for i in range(1, amount + 1):
            for c in coins:
                if i - c >= 0:
                    dp[i] = min(dp[i], 1 + dp[i - c])
        return dp[-1] if dp[-1] != amount + 1 else -1

    def wordBreak(self, s: str, wordDict: list[str]) -> bool:
        dp = [False] * len(s)
        for i in range(len(s)):
            for word in wordDict:
                if i >= (len(word) - 1) and (i == (len(word) - 1) or dp[i - len(word)]):
                    if s[i - len(word) + 1 : i + 1] == word:
                        dp[i] = True
                        break
        return dp[-1]

    def lengthOfLIS(self, nums: list[int]) -> int:
        n = len(nums)
        dp = [1] * n

        smallest_val = nums[-1]
        for i in range(n - 2, -1, -1):
            smallest_val = max(nums)
            for j in range(i + 1, n - 1, 1):
                if dp[j] == max(dp):
                    smallest_val = min(smallest_val, nums[j])
            if nums[i] < smallest_val:
                dp[i] = dp[i + 1] + 1
            else:
                dp[i] = dp[i + 1]

        print(dp)
        return dp[0]

    def maxProfit_1(self, k: int, prices: list[int]) -> int:
        @lru_cache(None)
        def dp(i, transaction_remaining, holding):
            if transaction_remaining == 0 or i == len(prices):
                return 0

            do_nothing = dp(i + 1, transaction_remaining, holding)
            do_something = 0

            if holding:
                do_something = prices[i] + dp(
                    i + 1, transaction_remaining - 1, 0
                )  # Sell stock

            else:
                do_something = -prices[i] + dp(
                    i + 1, transaction_remaining, 1
                )  # Buy stock

            return max(do_something, do_nothing)

        return dp(0, k, 0)

    def maxProfit_2(self, prices: list[int]) -> int:
        @lru_cache(None)
        def dp(i: int, h: bool, cd: bool) -> int:
            if i == len(prices):
                return 0

            no_action = dp(i=i + 1, h=h, cd=0)

            if h and not cd:
                action = prices[i] + dp(i + 1, 0, 1)

            elif not h and not cd:
                action = -prices[i] + dp(i + 1, 1, 0)

            else:
                action = no_action
            return max(action, no_action)

        return dp(0, 0, 0)

    def minCostClimbingStairs(self, cost: list[int]) -> int:
        if len(cost) == 1:
            return cost[0]

        if len(cost) == 2:
            return min(cost[0], cost[1])

        dp = [0] * (len(cost) + 1)

        for i in range(len(cost) - 1, -1, -1):
            if i == len(cost) - 1:
                dp[i] = cost[i]
            else:
                dp[i] = cost[i] + min(dp[i + 1], dp[i + 2])

        return min(dp[0], dp[1])

    def climbStairs(self, n: int) -> int:
        if n <= 2:
            return n

        dp = [0] * n

        dp[0] = 1
        dp[1] = 2

        for i in range(2, n):
            dp[i] = dp[i - 1] + dp[i - 2]

        return dp[n - 1]

    def numWays(self, n: int, k: int) -> int:
        dp = [0, k, k**2]
        for _ in range(n - 2):
            dp.append(sum(dp[-2:]) * (k - 1))

        return dp[n]

    def change(self, amount: int, coins: list[int]) -> int:
        n = len(coins)
        dp = [[0] * (amount + 1) for _ in range(n + 1)]

        for i in range(n):
            dp[i][0] = 1

        for i in range(n - 1, -1, -1):
            for j in range(1, amount + 1):
                if coins[i] > j:
                    dp[i][j] = dp[i + 1][j]
                else:
                    dp[i][j] = dp[i + 1][j] + dp[i][j - coins[i]]

        print(dp)
        return dp[0][amount]

    @lru_cache(None)
    def recursiveWithMemo(self, index, s) -> int:
        # If you reach the end of the string
        # Return 1 for success.
        if index == len(s):
            return 1

        # If the string starts with a zero, it can't be decoded
        if s[index] == "0":
            return 0

        if index == len(s) - 1:
            return 1

        answer = self.recursiveWithMemo(index + 1, s)
        if int(s[index : index + 2]) <= 26:
            answer += self.recursiveWithMemo(index + 2, s)

        return answer

    def numDecodings(self, s: str) -> int:
        return self.recursiveWithMemo(0, s)

    def maxProfit_3(self, prices: list[int]) -> int:
        buy_price = prices[0]
        profit = 0

        for p in prices[1:]:
            if buy_price > p:
                buy_price = p

            profit = max(profit, p - buy_price)

        return profit

    def maxSubarraySumCircular(self, nums: list[int]) -> int:
        curr = float("inf")
        mins = float("inf")
        curr1 = float("-inf")
        maxs1 = float("-inf")
        total = 0
        for i in nums:
            curr1 = max(curr1 + i, i)
            maxs1 = max(
                maxs1, curr1
            )  # check whether cur max is bigger or prev max is bigger
        for i in nums:
            total += i
            curr = min(curr + i, i)
            mins = min(mins, curr)
        if mins == total:
            return maxs1
        return max(maxs1, total - mins)

    def rob(self, nums: list[int]) -> int:
        if len(nums) == 1:
            return nums[0]

        n = len(nums)
        dp = [0] * n

        # Base vases
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])

        for i in range(2, n):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])

        return dp[n - 1]

    def mostPoints(self, questions: list[list[int]]) -> int:
        n = len(questions)
        if n == 1:
            return questions[0][0]

        dp = [0] * (n + 1)

        for i in range(n - 1, -1, -1):
            j = i + questions[i][1] + 1
            dp[i] = max(questions[i][0] + dp[min(j, n)], dp[i + 1])

        return dp[0]

    def minFallingPathSum(self, matrix: list[list[int]]) -> int:
        n = len(matrix)

        if n == 1:
            return matrix[0][0]

        def valid(row: int, col: int) -> bool:
            return 0 <= row < n and 0 <= col < n

        directions = [(-1, -1), (0, -1), (1, -1)]
        dp = [[0] * n for _ in range(n)]

        def find_min_available_node(row: int, col: int) -> int:
            curr_min = float("inf")
            for dx, dy in directions:
                next_row, next_col = row + dy, col + dx
                if valid(next_row, next_col):
                    curr_min = min(curr_min, dp[next_row][next_col])

            return curr_min

        for j in range(0, n):
            dp[0][j] = matrix[0][j]

        for i in range(1, n):
            for j in range(n):
                dp[i][j] = matrix[i][j] + find_min_available_node(i, j)

        return min(dp[n - 1])

    def uniquePathsWithObstacles(self, obstacleGrid: list[list[int]]) -> int:
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])

        # If the starting cell has an obstacle, then simply return as there would be
        # no paths to the destination.
        if obstacleGrid[0][0] == 1:
            return 0

        # Number of ways of reaching the starting cell = 1.
        obstacleGrid[0][0] = 1

        # Filling the values for the first column
        for i in range(1, m):
            obstacleGrid[i][0] = int(
                obstacleGrid[i][0] == 0 and obstacleGrid[i - 1][0] == 1
            )

        # Filling the values for the first row
        for j in range(1, n):
            obstacleGrid[0][j] = int(
                obstacleGrid[0][j] == 0 and obstacleGrid[0][j - 1] == 1
            )

        # Starting from cell(1,1) fill up the values
        # No. of ways of reaching cell[i][j] = cell[i - 1][j] + cell[i][j - 1]
        # i.e. From above and left.
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 0:
                    obstacleGrid[i][j] = obstacleGrid[i - 1][j] + obstacleGrid[i][j - 1]
                else:
                    obstacleGrid[i][j] = 0

        # Return value stored in rightmost bottommost cell. That is the destination.
        return obstacleGrid[m - 1][n - 1]


if __name__ == "__main__":
    nums = [-5, -3, -3, -2, 7, 1]
    multipliers = [-10, -5, 3, 4, 6]

    solution = DynamicProgramming()
    print(solution.maximumScore(nums=nums, multipliers=multipliers))
    print(solution.maximumScore_(nums=nums, multipliers=multipliers))
