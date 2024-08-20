from collections import deque

#   Author: Laurent Lanteigne
#   Date: 2024-08-20
#   source: https://leetcode.com/explore/interview/card/leetcodes-interview-crash-course-data-structures-and-algorithms/706/stacks-and-queues/4514/
#


class StacksQueues:
    def isValid(self, s: str) -> bool:
        stack = []
        matching = {"(": ")", "[": "]", "{": "}"}

        for c in s:
            if c in matching:  # if c is an opening bracket
                stack.append(c)
            else:
                if not stack:
                    return False

                previous_opening = stack.pop()
                if matching[previous_opening] != c:
                    return False

        return not stack

    def removeDuplicates(self, s: str) -> str:
        stack = []
        for c in s:
            if stack and stack[-1] == c:
                stack.pop()
            else:
                stack.append(c)

        return "".join(stack)

    def backspaceCompare(self, t: str, s: str) -> bool:
        def build(s: str):
            stack = []
            for c in stack:
                if c == "#":
                    stack.pop()
                else:
                    stack.append(c)

            return "".join(stack)

        return build(t) == build(s)

    def simplifyPath(self, path: str) -> str:
        path = path.replace("//", "/")
        path_stack = deque()
        for sub_path in path.split("/"):
            if sub_path:
                path_stack.append(sub_path)
        new_path_stack = deque()
        up_dir = 0
        while path_stack:
            sub_path = path_stack.pop()
            if sub_path == "..":
                up_dir += 1
                continue
            if sub_path != "." and sub_path != ".." and up_dir == 0:
                new_path_stack.appendleft(sub_path)
            elif up_dir != 0 and sub_path != ".":
                up_dir -= 1
        new_path = "/".join(new_path_stack)
        return f"/{new_path}"

    def makeGood(self, s: str) -> str:
        n = len(s)
        ans = []
        for c in s:
            if ans and abs(ord(ans[-1]) - ord(c)) == 32:
                ans.pop()
            else:
                ans.append(c)
        return "".join(ans)

    class RecentCounter:
        def __init__(self):
            self.queue = deque()

        def ping(self, t: int) -> int:
            while self.queue and self.queue[0] < t - 3000:
                self.queue.popleft()

            self.queue.append(t)
            return len(self.queue)

    class MovingAverage:
        def __init__(self, size: int):
            self.queue = deque(maxlen=size)

        def next(self, val: int) -> float:
            queue = self.queue
            queue.append(val)
            return float(sum(queue)) / len(queue)

    def dailyTemperatures(self, temperatures: list[int]) -> list[int]:
        stack = []
        answer = [0] * len(temperatures)

        for i in range(len(temperatures)):
            while stack and temperatures[stack[-1]] < temperatures[i]:
                j = stack.pop()
                answer[j] = i - j
            stack.append(i)

        return answer

    def maxSlidingWindow(self, nums: list[int], k: int) -> list[int]:
        ans = []
        queue = deque()
        for i in range(len(nums)):
            # maintain monotonic decreasing.
            # all elements in the deque smaller than the current one
            # have no chance of being the maximum, so get rid of them
            while queue and nums[i] > nums[queue[-1]]:
                queue.pop()

            queue.append(i)

            # queue[0] is the index of the maximum element.
            # if queue[0] + k == i, then it is outside the window
            if queue[0] + k == i:
                queue.popleft()

            # only add to the answer once our window has reached size k
            if i >= k - 1:
                ans.append(nums[queue[0]])

        return ans

    def longestSubarray(self, nums: list[int], limit: int) -> int:
        increasing = deque()
        decreasing = deque()
        left = ans = 0

        for right in range(len(nums)):
            # maintain the monotonic deques
            while increasing and increasing[-1] > nums[right]:
                increasing.pop()
            while decreasing and decreasing[-1] < nums[right]:
                decreasing.pop()

            increasing.append(nums[right])
            decreasing.append(nums[right])

            # maintain window property
            while decreasing[0] - increasing[0] > limit:
                if nums[left] == decreasing[0]:
                    decreasing.popleft()
                if nums[left] == increasing[0]:
                    increasing.popleft()
                left += 1

            ans = max(ans, right - left + 1)

        return ans

    class StockSpanner:
        def __init__(self):
            self.stack = []

        def next(self, price: int) -> int:
            span = 1
            last_span = 0
            while self.stack and price >= self.stack[-1][0]:
                value, last_span = (
                    self.stack.pop()
                )  # getting span of the element on top of the stack
                span += last_span
            self.stack.append(
                (price, span)
            )  # In the end we have the span for current price and we add it to the stack.
            return span

    def nextGreaterElement(self, nums1: list[int], nums2: list[int]) -> list[int]:
        ng = {}
        st = []

        for num in reversed(nums2):
            while st and st[-1] <= num:
                st.pop()
            ng[num] = st[-1] if st else -1
            st.append(num)

        return [ng[num] for num in nums1]
