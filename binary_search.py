import bisect
import math

#   Author: Laurent Lanteigne
#   Date: 2024-10-09
#   source: https://leetcode.com/explore/interview/card/leetcodes-interview-crash-course-data-structures-and-algorithms/710/binary-search/4696/#


class BinarySearch:
    def general_implementation_binary_search(self, arr: list[int], target: int) -> int:
        left = 0
        right = len(arr) - 1

        arr = sorted(arr)
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return

            if arr[mid] > target:
                right = mid - 1

            else:
                left = mid + 1

        return left

    def search(self, nums: list[int], target: int) -> int:
        left = 0
        right = len(nums) - 1

        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid

            if arr[mid] > target:
                right = mid - 1

            else:
                left = mid + 1

        return -1

    def searchMatrix(self, matrix: list[list[int]], target: int) -> bool:
        m, n = len(matrix), len(matrix[0])
        left, right = 0, m * n - 1

        while left <= right:
            mid = (left + right) // 2
            row = mid // n
            col = mid % n
            num = matrix[row][col]

            if num == target:
                return True

            if num < target:
                left = mid + 1
            else:
                right = mid - 1

        return False

    def successfulPairs(
        self, spells: list[int], potions: list[int], success: int
    ) -> list[int]:
        def binary_search(arr, target):
            left = 0
            right = len(arr) - 1
            while left <= right:
                mid = (left + right) // 2
                if arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1

            return left

        potions.sort()
        ans = []
        m = len(potions)

        for spell in spells:
            i = binary_search(potions, success / spell)
            ans.append(m - i)

        return ans

    def searchInsert(self, nums: list[int], target: int) -> int:
        left = 0
        right = len(nums) - 1

        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

        return left

    def answerQueries(self, nums: list[int], queries: list[int]) -> list[int]:
        # Get the prefix sum array of the sorted 'nums'.
        nums.sort()
        for i in range(1, len(nums)):
            nums[i] += nums[i - 1]

        answer = []

        # For each query, find its insertion index to the prefix sum array.
        for query in queries:
            index = bisect.bisect_right(nums, query)
            answer.append(index)

        return answer

    def minEatingSpeed(self, piles: list[int], h: int) -> int:
        def check(k):
            hours = 0
            for bananas in piles:
                hours += ceil(bananas / k)

            return hours <= h

        left = 1
        right = max(piles)
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                right = mid - 1
            else:
                left = mid + 1

        return left

    def minimumEffortPath(self, heights: list[list[int]]) -> int:
        def valid(row, col):
            return 0 <= row < m and 0 <= col < n

        def check(effort):
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            seen = {(0, 0)}
            stack = [(0, 0)]

            while stack:
                row, col = stack.pop()
                if (row, col) == (m - 1, n - 1):
                    return True

                for dx, dy in directions:
                    next_row, next_col = row + dy, col + dx
                    if valid(next_row, next_col) and (next_row, next_col) not in seen:
                        if (
                            abs(heights[next_row][next_col] - heights[row][col])
                            <= effort
                        ):
                            seen.add((next_row, next_col))
                            stack.append((next_row, next_col))

            return False

        m = len(heights)
        n = len(heights[0])
        left = 0
        right = max(max(row) for row in heights)
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                right = mid - 1
            else:
                left = mid + 1

    def minSpeedOntime(self, dist: list[int], hour: float) -> int:
        if len(dist) > ceil(hour):
            return -1

        def check(k):
            t = 0
            for d in dist:
                t = math.ceil(t)
                t += d / k

            return t <= hour

        left = 1
        right = 10**7
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                right = mid - 1
            else:
                left = mid + 1

        return left

    def smallestDivisor(self, nums: list[int], threshold: int) -> int:
        def check(k):
            t = 0
            for num in nums:
                t += math.ceil(num / k)

            return t <= threshold

        left = 1
        right = 10**7
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                right = mid - 1
            else:
                left = mid + 1

        return left

    def maximizeSweetness(self, sweetness: list[int], k: int) -> int:
        # Initialize the left and right boundaries.
        # left = 1 and right = (total sweetness) / (number of people).
        number_of_people = k + 1
        left = min(sweetness)
        right = sum(sweetness) // number_of_people

        while left < right:
            # Get the middle index between left and right boundary indexes.
            # cur_sweetness stands for the total sweetness for the current person.
            # people_with_chocolate stands for the number of people that have
            # a piece of chocolate of sweetness greater than or equal to mid.
            mid = (left + right + 1) // 2
            cur_sweetness = 0
            people_with_chocolate = 0

            # Start assigning chunks to the current person.
            for s in sweetness:
                cur_sweetness += s

                # If the total sweetness is no less than mid, this means we can break off
                # the current piece and move on to assigning chunks to the next person.
                if cur_sweetness >= mid:
                    people_with_chocolate += 1
                    cur_sweetness = 0

            if people_with_chocolate >= k + 1:
                left = mid
            else:
                right = mid - 1

        return right

    def splitArray(self, nums: list[int], k: int) -> int:
        # Edge case: When there's no splits required at all
        if k == 1:
            return sum(nums)

        # logic
        left, right = max(nums), sum(nums)
        while left <= right:
            mid = (left + right) // 2
            summ = 0
            splits = 1
            for i in nums:
                summ += i
                if summ <= mid:
                    continue
                else:
                    splits += 1
                    summ = i

            # no. of splits exceeds the required splits
            if splits > k:
                left = mid + 1

            # if no. of splits are equal or lower than required splits
            # This case is also applicable in 'splits == k', because ???? (refer the pinned comment)
            else:
                right = mid - 1

        return left
