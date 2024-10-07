#   Author: Laurent Lanteigne
#   Date: 2024-08-12
#   source: https://leetcode.com/explore/interview/card/leetcodes-interview-crash-course-data-structures-and-algorithms/703/arraystrings/
#
class ArraysStrings:
    def is_palindrome(self, s: str) -> bool:
        left = 0
        right = len(s) - 1

        while left < right:
            if s[left] == s[right]:
                left += 1
                right += 1
            else:
                return False

        return True

    def check_target(self, nums: list[int], target: int) -> bool:
        l = 0
        r = len(nums) - 1

        while l < r:
            val = nums[l] + nums[r]
            if val > target:
                r -= 1
            elif val < target:
                l += 1
            else:
                return True

        return False

    def is_subsequence(self, s: str, t: str) -> bool:
        i = j = 0
        while i < len(s) and j < len(t):
            if s[i] == t[j]:
                i += 1
            j += 1

        return i == len(s)

    def reverseString(self, s: list[str]) -> None:
        for i in range(len(s) // 2):
            tmp = s[-(i + 1)]
            s[-(i + 1)] = s[i]
            s[i] = tmp

    def sortedSquares(self, nums: list[int]) -> list[int]:
        n = len(nums)
        res = [0] * n
        l = 0
        r = n - 1

        for i in range(n - 1, -1, -1):
            if abs(nums[l]) > abs(nums[r]):
                res[i] = nums[l] ** 2
                l += 1
            else:
                res[i] = nums[r] ** 2
                r -= 1

        return res

    def findMaxAverage(self, nums: list[int], k: int) -> float:
        best = now = sum(nums[:k])
        for i in range(k, len(nums)):
            now += nums[i] - nums[i - k]
            if now > best:
                best = now
        return best / k

    def longestOnes(self, nums: list[int], k: int) -> int:
        l = r = 0
        for r in range(len(nums)):
            if nums[r] == 0:
                k -= 1
            if k < 0:
                if nums[l] == 0:
                    k += 1
                l += 1
        return r - l + 1

    def answer_queries(
        self, nums: list[int], queries: list[tuple[int, int]], limit: int
    ) -> list[bool]:
        prefix = [nums[0]]
        for i in range(1, len(nums)):
            prefix.append(nums[i] + prefix[-1])

        res = []
        for x, y in queries:
            if prefix[y] - prefix[x] + nums[x] > limit:
                res.append(False)
            else:
                res.append(True)

        return res

    def ways_to_split_array(self, nums: list[int]) -> int:
        n = len(nums)

        prefix = [nums[0]]
        for i in range(1, n):
            prefix.append(nums[i] + prefix[-1])

        ans = 0
        for i in range(n - 1):
            left_section = prefix[i]
            right_section = prefix[-1] - prefix[i]
            if left_section >= right_section:
                ans += 1

        return ans

    # Better Space O(1) versus O(n) previous answer
    def ways_to_split_array_2(self, nums: list[int]) -> int:
        ans = left_section = 0
        total = sum(nums)

        for i in range(len(nums) - 1):
            left_section += nums[i]
            right_section = total - left_section
            if left_section >= right_section:
                ans += 1

        return ans

    def minStartValue(self, nums: list[int]) -> int:
        curr_min = float("inf")
        curr_sum = 0
        for i in range(len(nums)):
            curr_sum += nums[i]
            curr_min = min(curr_min, curr_sum)

        return max(1 - curr_min, 1)

    def getAverages(self, nums: list[int], k: int) -> list[int]:
        if 2 * k >= len(nums):
            return [-1] * len(nums)

        res = [-1] * k
        prefix = [nums[0]]

        for i in range(1, len(nums)):
            prefix.append(nums[i] + prefix[-1])

        for i in range(k, len(nums) - k):
            res.append((prefix[i + k] - prefix[i - k] + nums[i - k]) // (2 * k + 1))

        res += [-1] * k

        return res
