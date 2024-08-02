from functools import lru_cache


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


if __name__ == "__main__":
    nums = [-5, -3, -3, -2, 7, 1]
    multipliers = [-10, -5, 3, 4, 6]

    solution = Solution()
    print(solution.maximumScore(nums=nums, multipliers=multipliers))
    print(solution.maximumScore_(nums=nums, multipliers=multipliers))
