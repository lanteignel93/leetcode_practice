import heapq
from collections import Counter


#   Author: Laurent Lanteigne
#   Date: 2024-10-09
#   source:  https://leetcode.com/explore/interview/card/leetcodes-interview-crash-course-data-structures-and-algorithms/708/heaps/4638/
#
class Heap:
    def lastStoneWeight(self, stones: list[int]) -> int:
        stones = [-stone for stone in stones]
        heapq.heapify(stones)
        while len(stones) > 1:
            first = abs(heapq.heappop(stones))
            second = abs(heapq.heappop(stones))
            if first != second:
                heapq.heappush(stones, -abs(first - second))

        return -stones[0] if stones else 0

    def halveArray(self, nums: list[int]) -> int:
        target = sum(nums) / 2
        heap = [-num for num in nums]
        heapq.heapify(heap)

        ans = 0
        while target > 0:
            ans += 1
            x = heapq.heappop(heap)
            target += x / 2
            heapq.heappush(heap, x / 2)

        return ans

    def minStoneSum(self, piles: list[int], k: int) -> int:
        piles = [-pile for pile in piles]
        heapq.heapify(piles)

        for _ in range(k):
            x = heapq.heappop(piles)
            x = -math.ceil(-x / 2)
            heapq.heappush(piles, x)

        return -sum(piles)

    def connectSticks(self, sticks: list[int]) -> int:
        if len(sticks) == 1:
            return 0

        heapq.heapify(sticks)

        cost = 0
        while len(sticks) > 1:
            first = heapq.heappop(sticks)
            second = heapq.heappop(sticks)

            combined = first + second
            cost += combined
            heapq.heappush(sticks, combined)

        return cost

    def topKFrequent(self, nums: list[int], k: int) -> list[int]:
        counts = Counter(nums)
        heap = []

        for key, val in counts.items():
            heapq.heappush(heap, (val, key))
            if len(heap) > k:
                heapq.heappop(heap)

        return [pair[1] for pair in heap]

    def findClosestElements(self, arr: list[int], k: int, x: int) -> list[int]:
        heap = []

        for num in arr:
            distance = abs(x - num)
            heapq.heappush(heap, (-distance, -num))
            if len(heap) > k:
                heapq.heappop(heap)

        return sorted([-pair[1] for pair in heap])

    def findKthLargest(self, nums: list[int], k: int) -> int:
        nums = [-num for num in nums]
        heapq.heapify(nums)

        for _ in range(k - 1):
            heapq.heappop(nums)

        return -heapq.heappop(nums)

    def kClosest(self, points: list[list[int]], k: int) -> list[list[int]]:
        def distance(point):
            x = point[0]
            y = point[1]
            return x * x + y * y

        # Use a min-heap to store the points based on their distance
        h = []
        for point in points:  # n
            d = distance(point)
            heapq.heappush(h, (d, point))  # heap size: n

        # Extract the k closest points
        res = []
        for i in range(k):  # k
            first_pt_d, first_pt = heapq.heappop(h)  # heap size: n
            res.append(first_pt)
        return res
