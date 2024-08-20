from collections import defaultdict, Counter


#   Author: Laurent Lanteigne
#   Date: 2024-08-19
#   source: https://leetcode.com/explore/interview/card/leetcodes-interview-crash-course-data-structures-and-algorithms/705/hashing/4510/
#
class Hashing:
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        dic = {}
        for i in range(len(nums)):
            num = nums[i]
            complement = target - num
            if complement in dic:  # This operation is O(1)!
                return [i, dic[complement]]

            dic[num] = i

        return [-1, -1]

    def checkIfPangram(self, sentence: str) -> bool:
        hash_map = set()
        for character in sentence.lower():
            if ord(character) >= ord("a") and ord(character) <= ord("z"):
                hash_map.add(character)

        return len(hash_map) == 26

    def missingNumber(self, nums: list[int]) -> int:
        unique_numbers = set(nums)
        for num in range(len(nums) + 1):
            if num not in unique_numbers:
                return num

    def countElements(self, arr: List[int]) -> int:
        answer = 0
        unique_vals = set(arr)
        for number in arr:
            if number + 1 in unique_vals:
                answer += 1

        return answer

    def find_longest_substring(self, s, k):
        counts = defaultdict(int)
        left = ans = 0
        for right in range(len(s)):
            counts[s[right]] += 1
            while len(counts) > k:
                counts[s[left]] -= 1
                if counts[s[left]] == 0:
                    del counts[s[left]]
                left += 1

            ans = max(ans, right - left + 1)

        return ans

    def intersection(self, nums: list[list[int]]) -> list[int]:
        counts = defaultdict(int)
        for arr in nums:
            for x in arr:
                counts[x] += 1

        n = len(nums)
        ans = []
        for key in counts:
            if counts[key] == n:
                ans.append(key)

        return sorted(ans)

    def areOccurrencesEqual(self, s: str) -> bool:
        counts = defaultdict(int)
        for c in s:
            counts[c] += 1

        frequencies = counts.values()
        return len(set(frequencies)) == 1

    def areOccurrencesEqual_(self, s: str) -> bool:
        return len(set(Counter(s).values())) == 1

    def subarraySum(self, nums: List[int], k: int) -> int:
        counts = defaultdict(int)
        counts[0] = 1
        ans = curr = 0

        for num in nums:
            curr += num
            ans += counts[curr - k]
            counts[curr] += 1

        return ans

    def findWinners(self, matches: list[list[int]]) -> list[list[int]]:
        hash_map = defaultdict(int)

        for match in matches:
            hash_map[match[0]] += 0
            hash_map[match[1]] += 1

        loss_none = []
        loss_one = []
        for key, val in sorted(hash_map.items()):
            if val == 0:
                loss_none += [key]
            elif val == 1:
                loss_one += [key]

        return [loss_none, loss_one]

    def largestUniqueNumber(self, nums: list[int]) -> int:
        #     if len(nums) == 1:
        #         return nums[0]
        nums = sorted(nums, reverse=True)
        hash_map = defaultdict(int)
        prev_num = nums[0]
        for num in nums:
            hash_map[num] += 1
            if num != prev_num and hash_map[prev_num] == 1:
                return prev_num
            prev_num = num
        if hash_map[num] == 1:
            return num
        return -1

    def maxNumberOfBalloons(self, text: str) -> int:
        hash_map = defaultdict(int)

        for character in text:
            hash_map[character] += 1

        answer = float("inf")

        for letter in "balloon":
            if letter == "l" or letter == "o":
                val = hash_map[letter] // 2
            else:
                val = hash_map[letter]
            answer = min(answer, val)

        return answer

    def findMaxLength(self, nums: list[int]) -> int:
        max_length = 0
        count = 0
        hash_map = {0: -1}
        for i, num in enumerate(nums):
            if num == 1:
                count += 1
            else:
                count -= 1

            if count in hash_map:
                max_length = max(max_length, i - hash_map[count])

            else:
                hash_map[count] = i

        return max_length

    def groupAnagrams(self, strs: list[str]) -> list[list[str]]:
        groups = defaultdict(list)
        for s in strs:
            key = "".join(sorted(s))
            groups[key].append(s)

        return groups.values()

    def minimumCardPickup(self, cards: list[int]) -> int:
        hash_map = defaultdict(list)

        for idx, card in enumerate(cards):
            hash_map[card].append(idx)

        res = float("inf")
        for card, idxs in hash_map.values():
            for i in range(len(idxs) - 1):
                res = min(res, idxs[i + 1] - idxs[i] + 1)

        return res if res < float("inf") else -1

    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        ammunitions = defaultdict(int)
        for s in magazine:
            ammunitions[s] += 1

        for s in ransomNote:
            ammunitions[s] -= 1
            if ammunitions[s] < 0:
                return False

        return True

    def lengthOfLongestSubstring(self, s: str) -> int:
        n = len(s)
        maxLength = 0
        charMap = {}
        left = 0

        for right in range(n):
            if s[right] not in charMap or charMap[s[right]] < left:
                charMap[s[right]] = right
                maxLength = max(maxLength, right - left + 1)
            else:
                left = charMap[s[right]] + 1
                charMap[s[right]] = right

        return maxLength
