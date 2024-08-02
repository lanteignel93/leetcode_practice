from functools import lru_cache


#   Author: Laurent Lanteigne
#   Date: 2024-08-01
#   source: https://leetcode.com/explore/learn/card/dynamic-programming/
#
class Solution:
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
        dp = [[0] * (m + 1) for _ in range(m + 1)]

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

    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [amount + 1] * (amount + 1)
        dp[0] = 0

        for i in range(1, amount + 1):
            for c in coins:
                if i - c >= 0:
                    dp[i] = min(dp[i], 1 + dp[i - c])
        return dp[-1] if dp[-1] != amount + 1 else -1

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        dp = [False] * len(s)
        for i in range(len(s)):
            for word in wordDict:
                if i >= (len(word) - 1) and (i == (len(word) - 1) or dp[i - len(word)]):
                    if s[i - len(word) + 1 : i + 1] == word:
                        dp[i] = True
                        break
        return dp[-1]

    def lengthOfLIS(self, nums: List[int]) -> int:
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

    def maxProfit_1(self, k: int, prices: List[int]) -> int:
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

    def maxProfit_2(self, prices: List[int]) -> int:
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


if __name__ == "__main__":
    nums = [-5, -3, -3, -2, 7, 1]
    multipliers = [-10, -5, 3, 4, 6]

    solution = Solution()
    print(solution.maximumScore(nums=nums, multipliers=multipliers))
    print(solution.maximumScore_(nums=nums, multipliers=multipliers))
