from typing import Optional


#   Author: Laurent Lanteigne
#   Date: 2024-08-21
#   source: https://leetcode.com/explore/interview/card/leetcodes-interview-crash-course-data-structures-and-algorithms/707/traversals-trees-graphs/4722/
#
class TreeNode:
    def __init__(self, val: int):
        self.val = val
        self.left = None
        self.right = None


class TreeDFS:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0

        left = self.maxDepth(root.left)
        right = self.maxDepth(root.right)
        return max(left, right) + 1

    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        def dfs(node: Optional[TreeNode], curr: int) -> bool:
            if not node:
                return False 

            if node.left == None and node.right == None:
                return (curr + node.val) == targetSum 

            curr += node.val 
            left = dfs(node.left, curr=curr)
            right = dfs(node.right, curr=curr)

            return left or right 

        return dfs(root, 0)

    def goodNodes(self, root: Optional[TreeNode]) -> int: 
        def dfs(node: Optional[TreeNode], val: int) -> int:
            if not node:
                return 0 
           
            left = dfs(node.left, max(max_so_far, node.val))
            right = dfs(node.right, max(max_so_far, node.val))
            ans = left + right 
            if node.val >= max_so_far
                ans+=1 
            
            return ans

        return dfs(root, max_so_far=float(-'inf'))

    def isSameTree(self, p:TreeNode, q:TreeNode) -> bool:
        if p is None and q is None:
            return True 
        
        if p is None or q is None: 
            return False 

        if p.val != q.val:
            return False 

        left = self.isSameTree(p.left, q.left)
        right = self.isSameTree(p.right, q.right)

        return left and right 

    def lowestCommonAncestor(self, root: Optional[TreeNode], p:TreeNode, q:TreeNode) -> TreeNode:
        if not root:
            return None 

        if root == p or root == q:
            return root 

        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        if left and right:
            return root 

        if left:
            return left

        return right

    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0 

        if not root.left and not root.right:
            return 1 

        left = self.minDepth(root.left) if root.left else float('inf')
        right = self.minDepth(root.right) if root.right else float('inf')

        return 1 + min(left, right)

    def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0 
        
        self.result = 0 

        def dfs(node: Optional[TreeNode], cur_min: int, cur_max: int) -> None: 

            if not node:
                return 

            self.result = max(
                self.result,
                abs(cur_min - node.val),
                abs(cur_max - node.val)
            )
            
            cur_min = min(cur_min, node.val)
            cur_max = max(cur_max, node.val)
            dfs(node.left, cur_min, cur_max)
            dfs(node.right, cur_min, cur_max)

        dfs(root, root.val, root.val)
        return self.result

    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        class DiameterData:
            def __init__(self, diameter, height):
                self.diameter = diameter
                self.height = height

        def calculateDiameterAndHeight(root: Optional[TreeNode]) -> DiameterData:
            if not root:
                return DiameterData(0, 0)

            leftData = calculateDiameterAndHeight(root.left)
            rightData = calculateDiameterAndHeight(root.right)

            currentDiameter = max(leftData.height + rightData.height,
                                  max(leftData.diameter, rightData.diameter))
            currentHeight = max(leftData.height, rightData.height) + 1

            return DiameterData(currentDiameter, currentHeight)

        data = calculateDiameterAndHeight(root)
        return data.diameter
