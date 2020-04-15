package com.company;

import java.util.*;

public class Leetcode100to200 {

//    101. Symmetric Tree
//    Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).
//
//    For example, this binary tree [1,2,2,3,4,4,3] is symmetric:
//
//               1
//              / \
//             2   2
//            / \ / \
//           3  4 4  3
//    But the following [1,2,2,null,3,null,3] is not:
//             1
//            / \
//           2   2
//            \   \
//             3   3


    public boolean isSymmetric(TreeNode root) {
        if (root == null)
            return true;
        return isSymmetric(root.left, root.right);
    }

    public boolean isSymmetric(TreeNode l, TreeNode r) {
        if (l == null && r == null) {
            return true;
        } else if (r == null || l == null) {
            return false;
        }

        if (l.val != r.val)
            return false;

        if (!isSymmetric(l.left, r.right))
            return false;
        if (!isSymmetric(l.right, r.left))
            return false;

        return true;
    }



//    102. Binary Tree Level Order Traversal
//    Given a binary tree, return the level order traversal of its nodes' values.
// (ie, from left to right, level by level).
//
//    For example:
//    Given binary tree [3,9,20,null,null,15,7],
//            3
//            / \
//            9  20
//            /  \
//            15   7
//            return its level order traversal as:
//            [
//            [3],
//            [9,20],
//            [15,7]
//            ]


    public ArrayList<ArrayList<Integer>> levelOrder(TreeNode root) {
        ArrayList<ArrayList<Integer>> result = new ArrayList();
        if (root == null) {
            return result;
        }

        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            ArrayList<Integer> level = new ArrayList<Integer>();
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode head = queue.poll();
                level.add(head.val);
                if (head.left != null) {
                    queue.offer(head.left);
                }
                if (head.right != null) {
                    queue.offer(head.right);
                }
            }
            result.add(level);
        }

        return result;
    }



//    103. Binary Tree Zigzag Level Order Traversal
//    Given a binary tree, return the zigzag level order traversal of its nodes' values.
// (ie, from left to right, then right to left for the next level and alternate between).
//
//    For example:
//    Given binary tree [3,9,20,null,null,15,7],
//            3
//            / \
//            9  20
//            /  \
//            15   7
//            return its zigzag level order traversal as:
//            [
//            [3],
//            [20,9],
//            [15,7]
//            ]


    public ArrayList<ArrayList<Integer>> zigzagLevelOrder(TreeNode root) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();

        if (root == null) {
            return result;
        }

        Queue<TreeNode> q = new LinkedList<>();
        q.add(root);
        boolean order = true;

        while(!q.isEmpty()) {
            int size = q.size();
            ArrayList<Integer> tmp = new ArrayList<>();
            for(int i = 0; i < size; ++i) {
                TreeNode n = q.poll();
                if(order) {
                    tmp.add(n.val);
                } else {
                    tmp.add(0, n.val);
                }
                if(n.left != null) q.add(n.left);
                if(n.right != null) q.add(n.right);
            }
            result.add(tmp);

            order = order ? false : true;
        }
        return result;

    }


//    104. Maximum Depth of Binary Tree
//    Given a binary tree, find its maximum depth.
//
//    The maximum depth is the number of nodes along
// the longest path from the root node down to the farthest leaf node.


    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }

        int left = maxDepth(root.left);
        int right = maxDepth(root.right);
        return Math.max(left, right) + 1;
    }


    //    105. Construct Binary Tree from Preorder and Inorder Traversal
//    Given preorder and inorder traversal of a tree, construct the binary tree.

    private int findPosition(int[] arr, int start, int end, int key) {
        for ( int i = start; i <= end; i++) {
            if (arr[i] == key) {
                return i;
            }
        }
        return -1;
    }

    private TreeNode myBuildTree(int[] inorder, int instart, int inend,
                                 int[] preorder, int prestart, int preend) {
        if (instart > inend) {
            return null;
        }

        TreeNode root = new TreeNode(preorder[prestart]);
        int position = findPosition(inorder, instart, inend, preorder[prestart]);

        root.left = myBuildTree(inorder, instart, position - 1,
                preorder, prestart + 1, prestart + position - instart);
        root.right = myBuildTree(inorder, position + 1, inend,
                preorder, position - inend + preend + 1, preend);
        return root;
    }

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if (inorder.length != preorder.length) {
            return null;
        }
        return myBuildTree(inorder, 0, inorder.length - 1, preorder, 0, preorder.length - 1);
    }



//    106. Construct Binary Tree from Inorder and Postorder Traversal
//    Given inorder and postorder traversal of a tree, construct the binary tree.

    public TreeNode buildTree106(int[] inorder, int[] postorder) {
        return buildTree106(inorder, inorder.length-1, 0, postorder, postorder.length-1);
    }

    private TreeNode buildTree106(int[] inorder, int inStart, int inEnd, int[] postorder,
                               int postStart) {
        if (postStart < 0 || inStart < inEnd)
            return null;

        //The last element in postorder is the root.
        TreeNode root = new TreeNode(postorder[postStart]);

        //find the index of the root from inorder. Iterating from the end.
        int rIndex = inStart;
        for (int i = inStart; i >= inEnd; i--) {
            if (inorder[i] == postorder[postStart]) {
                rIndex = i;
                break;
            }
        }
        //build right and left subtrees. Again, scanning from the end to find the sections.
        root.right = buildTree106(inorder, inStart, rIndex + 1, postorder, postStart-1);
        root.left = buildTree106(inorder, rIndex - 1, inEnd, postorder, postStart - (inStart - rIndex) -1);
        return root;
    }

    /**107. Binary Tree Level Order Traversal II
     Given a binary tree, return the bottom-up level order traversal of its nodes' values.
     (ie, from left to right, level by level from leaf to root).
     For example:
     Given binary tree [3,9,20,null,null,15,7],
     3
     / \
     9  20
     /  \
     15   7
     return its bottom-up level order traversal as:
     [
     [15,7],
     [9,20],
     [3]
     ]
     */

            public List<List<Integer>> levelOrder3(TreeNode root) {
                List<List<Integer>> result = new ArrayList();
                if (root == null) {
                    return result;
                }

                Queue<TreeNode> q = new LinkedList();
                q.offer(root);
                while (!q.isEmpty()) {
                    List<Integer> thisLevel = new ArrayList<Integer>();
                    int qSize = q.size();
                    for (int i = 0; i < qSize; i++) {
                        TreeNode curr = q.poll();
                        thisLevel.add(curr.val);
                        if (curr.left != null) {
                            q.offer(curr.left);
                        }
                        if (curr.right != null) {
                            q.offer(curr.right);
                        }
                    }
                    result.add(0, thisLevel);
                }
//                Collections.reverse(result);
                return result;
            }

    /**
     * 108. Convert Sorted Array to Binary Search Tree
     *
     * Given an array where elements are sorted in ascending order,
     * convert it to a height balanced BST.
     */

            public TreeNode sortedArrayToBST2(int[] num) {

                return rec(num, 0, num.length - 1);
            }

            public TreeNode rec(int[] num, int low, int high) {
                if (low > high) {
                    return null;
                }
                int mid = low + (high - low) / 2;
                TreeNode root = new TreeNode(num[mid]);
                root.left = rec(num, low, mid - 1);
                root.right = rec(num, mid + 1, high);
                return root;
            }


    /**
     * 109. Convert Sorted List to Binary Search Tree
     *
     * Given a singly linked list where elements are sorted in ascending order,
     * convert it to a height balanced BST.
     */

            public TreeNode sortedListToBST2(ListNode head) {

                return toBstRecursively(head, null);
            }

            public TreeNode toBstRecursively(ListNode start, ListNode end) {
                if (start == end) {
                    return null;
                } else {
                    ListNode mid = start;
                    ListNode fast = start;
                    while (fast != end && fast.next != end) {
                        mid = mid.next;
                        fast = fast.next.next;
                    }

                    TreeNode root = new TreeNode(mid.val);
                    root.left = toBstRecursively(start, mid);
                    root.right = toBstRecursively(mid.next, end);
                    return root;
                }
            }


    /**
     * 110. Balanced Binary Tree
     *
     * Given a binary tree, determine if it is height-balanced.
     * For this problem, a height-balanced binary tree is defined as a binary tree
     * in which the depth of the two subtrees of every node never differ by more than 1.
     Example 1:
     Given the following tree [3,9,20,null,null,15,7]:
     3
     / \
     9  20
     /  \
     15  7
     Return true.
     Example 2:
     Given the following tree [1,2,2,3,3,null,null,4,4]:
     1
     / \
     2   2
     / \
     3   3
     / \
     4 4
     Return false.
     */

            //recursively get the height of each subtree of each node,
    // compare their difference, if greater than 1, then return false
            //although this is working, but it's not efficient,
    // since it repeatedly computes the heights of each node every time
            //Its time complexity is O(n^2).

    public class Solution110 {
        private boolean result = true;

        public boolean isBalanced(TreeNode root) {
            maxDepth(root);
            return result;
        }

        public int maxDepth(TreeNode root) {
            if (root == null)
                return 0;
            int l = maxDepth(root.left);
            int r = maxDepth(root.right);
            if (Math.abs(l - r) > 1)
                result = false;
            return 1 + Math.max(l, r);
        }
    }




    /**
     * 111. Minimum Depth of Binary Tree
     *
     * Given a binary tree, find its minimum depth.
     * The minimum depth is the number of nodes along the shortest path
     * from the root node down to the nearest leaf node.
     * */

            /**DFS*/
            public int minDepth1(TreeNode root) {
                if (root == null) {
                    return 0;
                }
                int left = minDepth1(root.left);
                int right = minDepth1(root.right);
                if (left == 0) {
                    return right + 1;
                }
                if (right == 0) {
                    return left + 1;
                }
                return Math.min(left, right) + 1;
            }

//    124. Binary Tree Maximum Path Sum

//    112. Path Sum
//    Given a binary tree and a sum, determine if the tree has
// a root-to-leaf path such that adding up all the values along the path equals the given sum.
//
//    For example:
//    Given the below binary tree and sum = 22,
//            5
//            / \
//            4   8
//            /   / \
//            11  13  4
//            /  \      \
//            7    2      1
//            return true, as there exist a root-to-leaf path 5->4->11->2 which sum is 22.

    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null) {
            return false;
        }
        if (root.left == null && root.right == null) {
            return sum == root.val;
        }
        return hasPathSum (root.left, sum - root.val)
                || hasPathSum(root.right, sum - root.val);
    }

//    113. Path Sum II
//    Given a binary tree and a sum, find all root-to-leaf paths where each path's sum equals the given sum.
//
//    For example:
//    Given the below binary tree and sum = 22,
//            5
//            / \
//            4   8
//            /   / \
//            11  13  4
//            /  \    / \
//            7    2  5   1
//            return
//            [
//            [5,4,11,2],
//            [5,8,4,5]
//            ]


    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<List<Integer>> rst = new ArrayList<>();
        List<Integer> solution = new ArrayList<>();

        findSum(rst, solution, root, sum);
        return rst;
    }

    private void findSum(List<List<Integer>>result, List<Integer> solution, TreeNode root, int sum){
        if (root == null) {
            return;
        }

        if (root.left == null && root.right == null) {
            if (sum == root.val){
                solution.add(root.val);
                result.add(new ArrayList<Integer>(solution));
                solution.remove(solution.size()-1);
            }
            return;
        }

        solution.add(root.val);
        findSum(result, solution, root.left, sum-root.val);
        findSum(result, solution, root.right, sum-root.val);
        solution.remove(solution.size()-1);
    }


//    114. Flatten Binary Tree to Linked List
//    Given a binary tree, flatten it to a linked list in-place.
//
//            For example,
//            Given
//
//              1
//             / \
//            2   5
//           / \   \
//          3   4   6
//    The flattened tree should look like:
//            1
//            \
//            2
//            \
//            3
//            \
//            4
//            \
//            5
//            \
//            6
//

    private TreeNode prevNode = null;

    public void flatten(TreeNode root) {
        if (root == null) {
            return;
        }

        if (prevNode != null) {
            prevNode.left = null;
            prevNode.right = root;
        }

        prevNode = root;
        flatten(root.left);
        flatten(root.right);
    }


//    115. Distinct Subsequences
//    Given a string S and a string T, count the number of distinct subsequences of T in S.
//
//    A subsequence of a string is a new string which is formed from the original string
// by deleting some (can be none) of the characters without disturbing the relative positions of
// the remaining characters. (ie, "ACE" is a subsequence of "ABCDE" while "AEC" is not).
//
//    Here is an example:
//    S = "rabbbit", T = "rabbit"
//
//    Return 3.

    /*

    As a typical way to implement a dynamic programming algorithm, we construct a matrix dp, where each cell dp[i][j] represents the number of solutions of aligning substring T[0..i] with S[0..j];

Rule 1). dp[0][j] = 1, since aligning T = "" with any substring of S would have only ONE solution which is to delete all characters in S.

Rule 2). when i > 0, dp[i][j] can be derived by two cases:

case 1). if T[i] != S[j], then the solution would be to ignore the character S[j] and align substring T[0..i] with S[0..(j-1)]. Therefore, dp[i][j] = dp[i][j-1].

case 2). if T[i] == S[j], then first we could adopt the solution in case 1), but also we could match the characters T[i] and S[j] and align the rest of them (i.e. T[0..(i-1)] and S[0..(j-1)]. As a result, dp[i][j] = dp[i][j-1] + d[i-1][j-1]

e.g. T = B, S = ABC

dp[1][2]=1: Align T'=B and S'=AB, only one solution, which is to remove character A in S'.
     */

    public int numDistinct(String S, String T) {
        if (S == null || T == null) {
            return 0;
        }

        int[][] nums = new int[S.length() + 1][T.length() + 1];

        for (int i = 0; i <= S.length(); i++) {
            nums[i][0] = 1;
        }
        for (int i = 1; i <= S.length(); i++) {
            for (int j = 1; j <= T.length(); j++) {
                //S.charAt(i - 1) != T.charAt(j - 1), and we must have all T, so it is still j
                nums[i][j] = nums[i - 1][j];

                if (S.charAt(i - 1) == T.charAt(j - 1)) {
                    nums[i][j] += nums[i - 1][j - 1];
                }
            }
        }
        return nums[S.length()][T.length()];
    }


//    116. Populating Next Right Pointers in Each Node
//    Given a binary tree
//
//    struct TreeLinkNode {
//        TreeLinkNode *left;
//        TreeLinkNode *right;
//        TreeLinkNode *next;
//    }
//    Populate each next pointer to point to its next right node. If there is no next right node,
// the next pointer should be set to NULL.
//
//    Initially, all next pointers are set to NULL.
//
//            Note:
//
//    You may only use constant extra space.
//    You may assume that it is a perfect binary tree
// (ie, all leaves are at the same level, and every parent has two children).
//    For example,
//    Given the following perfect binary tree,
//              1
//            /  \
//            2    3
//            / \  / \
//            4  5  6  7
//    After calling your function, the tree should look like:
//            1 -> NULL
//    /  \
//            2 -> 3 -> NULL
//    / \  / \
//            4->5->6->7 -> NULL

    public class TreeLinkNode {
        int val;
        TreeLinkNode left, right, next;
        TreeLinkNode(int x) { val = x; }
    }

    public void connect(TreeLinkNode root) {
        TreeLinkNode level_start=root;
        while(level_start!=null){

            TreeLinkNode cur=level_start;
            level_start=level_start.left;

            while(cur!=null){
                if(cur.left!=null) cur.left.next=cur.right;
                if(cur.right!=null && cur.next!=null) cur.right.next=cur.next.left;

                cur=cur.next;
            }
        }
    }


//    117. Populating Next Right Pointers in Each Node II
//    Follow up for problem "Populating Next Right Pointers in Each Node".
//
//    What if the given tree could be any binary tree? Would your previous solution still work?
//
//    Note:
//
//    You may only use constant extra space.
//            For example,
//    Given the following binary tree,
//            1
//            /  \
//            2    3
//            / \    \
//            4   5    7
//    After calling your function, the tree should look like:
//            1 -> NULL
//    /  \
//            2 -> 3 -> NULL
//    / \    \
//            4-> 5 -> 7 -> NULL

    public void connect2(TreeLinkNode root) {

        while(root != null){//root is the first node in upper level
            TreeLinkNode dumpChild = new TreeLinkNode(0);
            TreeLinkNode currentChild = dumpChild;
            while(root!=null){
                if(root.left != null) {
                    currentChild.next = root.left;
                    currentChild = currentChild.next;}
                if(root.right != null) {
                    currentChild.next = root.right;
                    currentChild = currentChild.next;}
                root = root.next;
            }
            root = dumpChild.next;
        }
    }


//    118. Pascal's Triangle
//    Given numRows, generate the first numRows of Pascal's triangle.
//
//    For example, given numRows = 5,
//    Return
//
//    [
//            [1],                       [1],
//            [1,1],                   [1,1]
//            [1,2,1],      =>       [1,2,1]
//            [1,3,3,1],           [1,3,3,1]
//            [1,4,6,4,1]
//            ]

    //1,2,1 => 1,1,2,1 => 1,3,3,1

    public List<List<Integer>> generate(int numRows)
    {
        List<List<Integer>> allrows = new ArrayList<List<Integer>>();
        ArrayList<Integer> row = new ArrayList<Integer>();
        for(int i=0;i<numRows;i++)
        {
            //both side are ok, we can add the into head or tail,
            //the key is we don't calculate the first and last items in the new list
            row.add(0, 1);
            for(int j=1;j<row.size()-1;j++)
                row.set(j, row.get(j)+row.get(j+1));
            allrows.add(new ArrayList<Integer>(row));
        }
        return allrows;

    }


//    119. Pascal's Triangle II
//    Given an index k, return the kth row of the Pascal's triangle.
//
//    For example, given k = 3,
//    Return [1,3,3,1].

    public List<Integer> getRow(int rowIndex) {
        List<Integer> list = new ArrayList<Integer>();
        if (rowIndex < 0)
            return list;

        for (int i = 0; i < rowIndex + 1; i++) {
            list.add(0, 1);
            for (int j = 1; j < list.size() - 1; j++) {
                list.set(j, list.get(j) + list.get(j + 1));
            }
        }
        return list;
    }

//    120. Triangle
//    Given a triangle, find the minimum path sum from top to bottom.
// Each step you may move to adjacent numbers on the row below.
//
//    For example, given the following triangle
//    [
//             [2],
//            [3,4],
//           [6,5,7],
//          [4,1,8,3]
//            ]
//    The minimum path sum from top to bottom is 11 (i.e., 2 + 3 + 5 + 1 = 11).


    public int minimumTotal(int[][] triangle) {
        if (triangle == null || triangle.length == 0) {
            return -1;
        }
        if (triangle[0] == null || triangle[0].length == 0) {
            return -1;
        }

        // state: f[x][y] = minimum path value from 0,0 to x,y
        int n = triangle.length;
        int[][] f = new int[n][n];

        // initialize the first and last elements
        f[0][0] = triangle[0][0];
        for (int i = 1; i < n; i++) {
            f[i][0] = f[i - 1][0] + triangle[i][0];
            f[i][i] = f[i - 1][i - 1] + triangle[i][i];
        }

        // top down
        for (int i = 1; i < n; i++) {
            for (int j = 1; j < i; j++) {
                f[i][j] = Math.min(f[i - 1][j - 1], f[i - 1][j]) + triangle[i][j];
            }
        }

        // answer
        int best = f[n - 1][0];
        for (int i = 1; i < n; i++) {
            best = Math.min(best, f[n - 1][i]);
        }
        return best;
    }

//    121. Best Time to Buy and Sell Stock
//    Say you have an array for which the ith element is the price of a given stock on day i.
//
//    If you were only permitted to complete at most one transaction
// (ie, buy one and sell one share of the stock), design an algorithm to find the maximum profit.
//
//    Example 1:
//    Input: [7, 1, 5, 3, 6, 4]
//    Output: 5
//
//    max. difference = 6-1 = 5 (not 7-1 = 6, as selling price needs to be larger than buying price)
//    Example 2:
//    Input: [7, 6, 4, 3, 1]
//    Output: 0
//
//    In this case, no transaction is done, i.e. max profit = 0.

    public int maxProfit(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        //min max doesn't work, because it is possible max is in the front of min
        int min = Integer.MAX_VALUE;  //just remember the smallest price
        int profit = 0;
        for (int i : prices) {
            min = i < min ? i : min;
            profit = (i - min) > profit ? i - min : profit;
        }

        return profit;
    }

//    122. Best Time to Buy and Sell Stock II
//    Say you have an array for which the ith element is the price of a given stock on day i.
//
//    Design an algorithm to find the maximum profit.
// You may complete as many transactions as you like
// (ie, buy one and sell one share of the stock multiple times).
// However, you may not engage in multiple transactions at the same time
// (ie, you must sell the stock before you buy again).

    public int maxProfit2(int[] prices) {
        int profit = 0;
        for (int i = 0; i < prices.length - 1; i++) {
            int diff = prices[i+1] - prices[i];
            if (diff > 0) {
                profit += diff;
            }
        }
        return profit;
    }


//    123. Best Time to Buy and Sell Stock III
//    Say you have an array for which the ith element is the price of a given stock on day i.
//
//    Design an algorithm to find the maximum profit. You may complete at most two transactions.
//
//            Note:
//    You may not engage in multiple transactions at the same time
// (ie, you must sell the stock before you buy again).

    public int maxProfit3(int[] prices) {
        if (prices == null || prices.length <= 1) {
            return 0;
        }

        int[] left = new int[prices.length];
        int[] right = new int[prices.length];

        // DP from left to right;
        left[0] = 0;
        int min = prices[0];
        for (int i = 1; i < prices.length; i++) {
            min = Math.min(prices[i], min);
            left[i] = Math.max(left[i - 1], prices[i] - min);
        }

        //DP from right to left;
        right[prices.length - 1] = 0;
        int max = prices[prices.length - 1];
        for (int i = prices.length - 2; i >= 0; i--) {
            max = Math.max(prices[i], max);
            right[i] = Math.max(right[i + 1], max - prices[i]);
        }

        int profit = 0;
        for (int i = 0; i < prices.length; i++){
            profit = Math.max(left[i] + right[i], profit);
        }

        return profit;
    }


    //    188. Best Time to Buy and Sell Stock IV
//    Say you have an array for which the ith element is the price of a given stock on day i.
//
//    Design an algorithm to find the maximum profit. You may complete at most k transactions.
//
//            Note:
//    You may not engage in multiple transactions at the same time
// (ie, you must sell the stock before you buy again).

    public int maxProfit(int k, int[] prices) {
        // write your code here
        if (k == 0) {
            return 0;
        }
        if (k >= prices.length / 2) {
            int profit = 0;
            for (int i = 1; i < prices.length; i++) {
                if (prices[i] > prices[i - 1]) {
                    profit += prices[i] - prices[i - 1];
                }
            }
            return profit;
        }
        int n = prices.length;
        int[][] mustsell = new int[n + 1][k + 1];   // mustSell[i][j] 表示前i天，进行j次交易，第i天必须sell的最大获益
        int[][] globalbest = new int[n + 1][k + 1];  // globalbest[i][j] 表示前i天，进行j次交易，第i天可以buy不sell的最大获益

        mustsell[0][0] = globalbest[0][0] = 0;
        for (int i = 1; i <= k; i++) {
            mustsell[0][i] = globalbest[0][i] = 0;
        }


        for (int i = 1; i < n; i++) {
            int gainorlose = prices[i] - prices[i - 1];
            mustsell[i][0] = 0;

            for (int j = 1; j <= k; j++) {
                //mustsell[i][j] = mustsell[(i - 1)][j] + gainorlose means finish j transactions in (i-1)th or ith day, so the gain is different.
                //mustsell[i][j] = globalbest[(i - 1)][j - 1] + gainorlose means, finished j-1 transactions, (i-1)th day not sell, so sell it on ith day.
                mustsell[i][j] = Math.max(globalbest[(i - 1)][j - 1] + gainorlose,
                        mustsell[(i - 1)][j] + gainorlose);
                globalbest[i][j] = Math.max(globalbest[(i - 1)][j]/*do nothing*/, mustsell[i][j]);
            }
        }
        return globalbest[(n - 1)][k];
    }


    //    309. Best Time to Buy and Sell Stock with Cooldown
//    Say you have an array for which the ith element is the price of a given stock on day i.
//
//    Design an algorithm to find the maximum profit.
// You may complete as many transactions as you like
// (ie, buy one and sell one share of the stock multiple times)
// with the following restrictions:
//
//    You may not engage in multiple transactions at the same time
// (ie, you must sell the stock before you buy again).
//    After you sell your stock, you cannot buy stock on next day. (ie, cooldown 1 day)
//    Example:
//
//    prices = [1, 2, 3, 0, 2]
//    maxProfit = 3
//    transactions = [buy, sell, cooldown, buy, sell]


//    The series of problems are typical dp. The key for dp is to find the variables
// to represent the states and deduce the transition function.
//
//    Of course one may come up with a O(1) space solution directly,
// but I think it is better to be generous when you think and be greedy when you implement.
//
//    The natural states for this problem is the 3 possible transactions :
// buy, sell, rest. Here rest means no transaction on that day (aka cooldown).
//
//    Then the transaction sequences can end with any of these three states.
//
//    For each of them we make an array, buy[n], sell[n] and rest[n].
//
//    buy[i] means before day i what is the maxProfit for any sequence end with buy.
//
//            sell[i] means before day i what is the maxProfit for any sequence end with sell.
//
//            rest[i] means before day i what is the maxProfit for any sequence end with rest.
//
//    Then we want to deduce the transition functions for buy sell and rest. By definition we have:
//
//    buy[i]  = max(rest[i-1]-price, buy[i-1])
//    sell[i] = max(buy[i-1]+price, sell[i-1])
//    rest[i] = max(sell[i-1], buy[i-1], rest[i-1])
//    Where price is the price of day i. All of these are very straightforward.
// They simply represents :
//
//            (1) We have to `rest` before we `buy` and
//            (2) we have to `buy` before we `sell`
//    One tricky point is how do you make sure you sell before you buy,
// since from the equations it seems that [buy, rest, buy] is entirely possible.
//
//            Well, the answer lies within the fact that buy[i] <= rest[i]
// which means rest[i] = max(sell[i-1], rest[i-1]).
// That made sure [buy, rest, buy] is never occurred.
//
//    A further observation is that and rest[i] <= sell[i] is also true therefore
//
//    rest[i] = sell[i-1]
//    Substitute this in to buy[i] we now have 2 functions instead of 3:
//
//    buy[i] = max(sell[i-2]-price, buy[i-1])
//    sell[i] = max(buy[i-1]+price, sell[i-1])
//    This is better than 3, but
//
//    we can do even better
//
//    Since states of day i relies only on i-1 and i-2 we can reduce the O(n) space to O(1).
// And here we are at our final solution:
//
//    Java

    public int maxProfit4(int[] prices) {
        int sell = 0, prev_sell = 0, buy = Integer.MIN_VALUE, prev_buy;
        for (int price : prices) {
            prev_buy = buy;
            buy = Math.max(prev_sell - price, prev_buy);

            prev_sell = sell;
            sell = Math.max(prev_buy + price, prev_sell);
        }
        return sell;
    }

    //714. Best Time to Buy and Sell Stock with Transaction Fee

    /*

    Your are given an array of integers prices,
    for which the i-th element is the price of a given stock on day i;
    and a non-negative integer fee representing a transaction fee.

You may complete as many transactions as you like,
but you need to pay the transaction fee for each transaction.
You may not buy more than 1 share of a stock at a time
(ie. you must sell the stock share before you buy again.)

Return the maximum profit you can make.

Example 1:

Input: prices = [1, 3, 2, 8, 4, 9], fee = 2
Output: 8
Explanation: The maximum profit can be achieved by:
Buying at prices[0] = 1
Selling at prices[3] = 8
Buying at prices[4] = 4
Selling at prices[5] = 9
The total profit is ((8 - 1) - 2) + ((9 - 4) - 2) = 8.

Note:
0 < prices.length <= 50000.
0 < prices[i] < 50000.
0 <= fee < 50000.


     */

    /*
    Case VI: k = +Infinity but with transaction fee

Again this case resembles Case II very much as they have the same k value,
except now the recurrence relations need to be modified slightly to account
for the "transaction fee" requirement. The original recurrence relations for Case II are given by

T[i][k][0] = max(T[i-1][k][0], T[i-1][k][1] + prices[i])
T[i][k][1] = max(T[i-1][k][1], T[i-1][k][0] - prices[i])

Since now we need to pay some fee (denoted as fee) for each transaction made,
the profit after buying or selling the stock on the i-th day should be subtracted
by this amount, therefore the new recurrence relations will be either

T[i][k][0] = max(T[i-1][k][0], T[i-1][k][1] + prices[i])
T[i][k][1] = max(T[i-1][k][1], T[i-1][k][0] - prices[i] - fee)

or

T[i][k][0] = max(T[i-1][k][0], T[i-1][k][1] + prices[i] - fee)
T[i][k][1] = max(T[i-1][k][1], T[i-1][k][0] - prices[i])

Note we have two options as for when to subtract the fee. This is because
(as I mentioned above) each transaction is characterized by two actions
coming as a pair - - buy and sell. The fee can be paid either when we
buy the stock (corresponds to the first set of equations) or when we sell it
(corresponds to the second set of equations). The following are the O(n)
time and O(1) space solutions corresponding to these two options,
where for the second solution we need to pay attention to possible overflows.
     */

    public int maxProfit(int[] prices, int fee) {
        int T_sell = 0, T_buy = Integer.MIN_VALUE;

        for (int price : prices) {
            int T_sell_old = T_sell;
            int T_buy_old = T_buy;

            T_sell = Math.max(T_sell, T_buy_old + price - fee);  //selling
            T_buy = Math.max(T_buy, T_sell_old - price);   //buy
        }

        return T_sell;
    }


    //    112. Path Sum
//    113. Path Sum


//    124. Binary Tree Maximum Path Sum
//            Hard
//
//    Given a non-empty binary tree, find the maximum path sum.
//
//            For this problem, a path is defined as any sequence of nodes from some starting node to any node in the tree along the parent-child connections. The path must contain at least one node and does not need to go through the root.
//
//            Example 1:
//
//    Input: [1,2,3]
//
//            1
//            / \
//            2   3
//
//    Output: 6
//
//    Example 2:
//
//    Input: [-10,9,20,null,null,15,7]
//
//            -10
//            / \
//            9  20
//              /  \
//             15   7
//
//    Output: 42



    class Solution124 {
        int maxValue;

        public int maxPathSum(TreeNode root) {
            maxValue = Integer.MIN_VALUE;
            maxPathDown(root);
            return maxValue;
        }

        private int maxPathDown(TreeNode node) {
            if (node == null) return 0;
            int left = Math.max(0, maxPathDown(node.left));
            int right = Math.max(0, maxPathDown(node.right));
            maxValue = Math.max(maxValue, left + right + node.val);
            return Math.max(left, right) + node.val;
        }
    }


//    125. Valid Palindrome
//    Given a string, determine if it is a palindrome, considering only
// alphanumeric characters and ignoring cases.
//
//    For example,
//    "A man, a plan, a canal: Panama" is a palindrome.
//    "race a car" is not a palindrome.
//
//    Note:
//    Have you consider that the string might be empty? This is a good question
// to ask during an interview.
//
//    For the purpose of this problem, we define empty string as valid palindrome.


    public boolean isPalindrome(String s) {
        if (s == null || s.length() == 0 || s.trim().equals("")) {
            return true;
        }

        int front = 0;
        int end = s.length() - 1;
        while (front < end) {
            while (front < s.length() && !isvalid(s.charAt(front))){ // nead to check range of a/b
                front++;
            }

            while (end >= 0 && ! isvalid(s.charAt(end))) { // same here, need to check border of a,b
                end--;
            }

            if (Character.toLowerCase(s.charAt(front)) != Character.toLowerCase(s.charAt(end))) {
                break;
            } else {
                front++;
                end--;
            }
        }

        return end <= front;
    }

    private boolean isvalid (char c) {

        return Character.isLetter(c) || Character.isDigit(c);
    }


//    127. Word Ladder
//    Given two words (beginWord and endWord), and a dictionary's word list,
// find the length of shortest transformation sequence from beginWord to endWord, such that:
//
//    Only one letter can be changed at a time.
//    Each transformed word must exist in the word list. Note that beginWord
// is not a transformed word.
//    For example,
//
//    Given:
//    beginWord = "hit"
//    endWord = "cog"
//    wordList = ["hot","dot","dog","lot","log","cog"]
//    As one shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog",
//            return its length 5.
//
//    Note:
//    Return 0 if there is no such transformation sequence.
//    All words have the same length.
//    All words contain only lowercase alphabetic characters.
//    You may assume no duplicates in the word list.
//    You may assume beginWord and endWord are non-empty and are not the same.


    public int ladderLength(String start, String end, List<String> wordList) {
        Set<String> dict = new HashSet<>(wordList);

        if (start.equals(end)) {
            return 1;
        }

        HashSet<String> seen = new HashSet<String>();
        Queue<String> queue = new LinkedList<String>();
        queue.offer(start);
        seen.add(start);
        int length = 1;

        while (!queue.isEmpty()) {
            length++;
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                String word = queue.poll();
                for (String nextWord: getNextWords(word, dict)) {
                    if (seen.contains(nextWord)) {
                        continue;
                    }
                    if (nextWord.equals(end)) {
                        return length;
                    }

                    seen.add(nextWord);
                    queue.offer(nextWord);
                }
            }
        }

        return 0;
    }

    // replace character of a string at given index to a given character
    // return a new string
    private String replace(String s, int index, char c) {
        char[] chars = s.toCharArray();
        chars[index] = c;
        return new String(chars);
    }

    // get connections with given word.
    // for example, given word = 'hot', dict = {'hot', 'hit', 'hog'}
    // it will return ['hit', 'hog']
    private ArrayList<String> getNextWords(String word, Set<String> dict) {
        ArrayList<String> nextWords = new ArrayList<String>();
        for (char c = 'a'; c <= 'z'; c++) {
            for (int i = 0; i < word.length(); i++) {
                if (c == word.charAt(i)) {
                    continue;
                }
                String nextWord = replace(word, i, c);
                if (dict.contains(nextWord)) {
                    nextWords.add(nextWord);
                }
            }
        }
        return nextWords;
    }


//    126. Word Ladder II
//    Given two words (beginWord and endWord), and a dictionary's word list,
// find all shortest transformation sequence(s) from beginWord to endWord, such that:
//
//    Only one letter can be changed at a time
//    Each transformed word must exist in the word list.
// Note that beginWord is not a transformed word.
//    For example,
//
//    Given:
//    beginWord = "hit"
//    endWord = "cog"
//    wordList = ["hot","dot","dog","lot","log","cog"]
//    Return
//    [
//            ["hit","hot","dot","dog","cog"],
//            ["hit","hot","lot","log","cog"]
//            ]
//    Note:
//    Return an empty list if there is no such transformation sequence.
//    All words have the same length.
//    All words contain only lowercase alphabetic characters.
//    You may assume no duplicates in the word list.
//    You may assume beginWord and endWord are non-empty and are not the same.

    public List<List<String>> findLadders(String start, String end, List<String> wordList) {
        HashSet<String> dict = new HashSet<String>(wordList);

        HashMap<String, ArrayList<String>> nodeNeighbors = new HashMap<String, ArrayList<String>>();// Neighbors for every node
        HashMap<String, Integer> distance = new HashMap<String, Integer>();// Distance of every node from the start node
        dict.add(start);
        bfs(start, end, dict, nodeNeighbors, distance);


        List<List<String>> res = new ArrayList<List<String>>();
        ArrayList<String> solution = new ArrayList<String>();
        dfs(start, end, dict, nodeNeighbors, distance, solution, res);
        return res;
    }

    // BFS: Trace every node's distance from the start node (level by level).
    private void bfs(String start, String end, Set<String> dict, HashMap<String, ArrayList<String>> nodeNeighbors, HashMap<String, Integer> distance) {
        for (String str : dict)
            nodeNeighbors.put(str, new ArrayList<String>());

        Queue<String> queue = new LinkedList<String>();
        queue.offer(start);
        distance.put(start, 0);

        while (!queue.isEmpty()) {
            int count = queue.size();
            boolean foundEnd = false;
            for (int i = 0; i < count; i++) {
                String cur = queue.poll();
                int curDistance = distance.get(cur);
                ArrayList<String> neighbors = getNeighbors(cur, dict);

                for (String neighbor : neighbors) {
                    nodeNeighbors.get(cur).add(neighbor);
                    if (!distance.containsKey(neighbor)) {// Check if visited
                        distance.put(neighbor, curDistance + 1);
                        if (end.equals(neighbor))// Found the shortest path
                            foundEnd = true; //can't break here because we want to get all paths
                        else
                            queue.offer(neighbor);
                    }
                }
            }

            if (foundEnd)
                break;
        }
    }

    // Find all next level nodes.
    private ArrayList<String> getNeighbors(String node, Set<String> dict) {
        ArrayList<String> res = new ArrayList<String>();
        char chs[] = node.toCharArray();

        for (char ch ='a'; ch <= 'z'; ch++) {
            for (int i = 0; i < chs.length; i++) {
                if (chs[i] == ch) continue;
                char old_ch = chs[i];
                chs[i] = ch;
                if (dict.contains(String.valueOf(chs))) {
                    res.add(String.valueOf(chs));
                }
                chs[i] = old_ch;
            }

        }
        return res;
    }

    // DFS: output all paths with the shortest distance.
    private void dfs(String cur, String end, Set<String> dict,
                     HashMap<String, ArrayList<String>> nodeNeighbors,
                     HashMap<String, Integer> distance,
                     ArrayList<String> solution,
                     List<List<String>> res) {
        solution.add(cur);
        if (end.equals(cur)) {
            res.add(new ArrayList<String>(solution));
        } else {
            for (String next : nodeNeighbors.get(cur)) {
                if (distance.get(next) == distance.get(cur) + 1) {
                    dfs(next, end, dict, nodeNeighbors, distance, solution, res);
                }
            }
        }
        solution.remove(solution.size() - 1);
    }



//    128. Longest Consecutive Sequence
//    Given an unsorted array of integers, find the length of the
// longest consecutive elements sequence.
//
//    For example,
//    Given [100, 4, 200, 1, 3, 2],
//    The longest consecutive elements sequence is [1, 2, 3, 4]. Return its length: 4.
//
//    Your algorithm should run in O(n) complexity.


    public int longestConsecutive(int[] nums) {
        HashSet<Integer> set = new HashSet<>();
        for (int i = 0; i < nums.length; i++) {
            set.add(nums[i]);
        }

        int longest = 0;
        for (int i = 0; i < nums.length; i++) {
            int down = nums[i] - 1;
            while (set.contains(down)) {
                set.remove(down);
                down--;
            }

            int up = nums[i] + 1;
            while (set.contains(up)) {
                set.remove(up);
                up++;
            }

            longest = Math.max(longest, up - down - 1);
        }

        return longest;
    }


//    129. Sum Root to Leaf Numbers
//    Given a binary tree containing digits from 0-9 only,
// each root-to-leaf path could represent a number.
//
//    An example is the root-to-leaf path 1->2->3 which represents the number 123.
//
//    Find the total sum of all root-to-leaf numbers.
//
//    For example,
//
//             1
//            / \
//           2   3
//    The root-to-leaf path 1->2 represents the number 12.
//    The root-to-leaf path 1->3 represents the number 13.
//
//    Return the sum = 12 + 13 = 25.


    public int sumNumbers(TreeNode root) {

        return dfs(root, 0);
    }

    private int dfs(TreeNode root, int prev){
        if(root == null) {
            return 0;
        }

        int sum = root.val + prev * 10;
        if(root.left == null && root.right == null) {
            return sum;
        }

        return dfs(root.left, sum) + dfs(root.right, sum);
    }



//    130. Surrounded Regions
//    Given a 2D board containing 'X' and 'O' (the letter O), capture all regions surrounded by 'X'.
//
//    A region is captured by flipping all 'O's into 'X's in that surrounded region.
//
//            For example,
//    X X X X
//    X O O X
//    X X O X
//    X O X X
//    After running your function, the board should be:
//
//    X X X X
//    X X X X
//    X X X X
//    X O X X


    public void solve(char[][] board) {
        if (board.length == 0 || board[0].length == 0)
            return;
        if (board.length < 2 || board[0].length < 2)
            return;
        int m = board.length, n = board[0].length;
        //Any 'O' connected to a boundary can't be turned to 'X', so ...
        //Start from first and last column, turn 'O' to '*'.
        for (int i = 0; i < m; i++) {
            if (board[i][0] == 'O')
                boundaryDFS(board, i, 0);
            if (board[i][n-1] == 'O')
                boundaryDFS(board, i, n-1);
        }
        //Start from first and last row, turn '0' to '*'
        for (int j = 0; j < n; j++) {
            if (board[0][j] == 'O')
                boundaryDFS(board, 0, j);
            if (board[m-1][j] == 'O')
                boundaryDFS(board, m-1, j);
        }
        //post-prcessing, turn 'O' to 'X', '*' back to 'O', keep 'X' intact.
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == 'O')
                    board[i][j] = 'X';
                else if (board[i][j] == '*')
                    board[i][j] = 'O';
            }
        }
    }
    //Use DFS algo to turn internal however boundary-connected 'O' to '*';
    private void boundaryDFS(char[][] board, int i, int j) {
        if (i < 0 || i > board.length - 1 || j <0 || j > board[0].length - 1 || board[i][j] != 'O')
            return;

        if (board[i][j] == 'O')
            board[i][j] = '*';

        boundaryDFS(board, i-1, j);
        boundaryDFS(board, i+1, j);
        boundaryDFS(board, i, j-1);
        boundaryDFS(board, i, j+1);
    }


//    131. Palindrome Partitioning
//    Given a string s, partition s such that every substring of the partition is a palindrome.
//
//    Return all possible palindrome partitioning of s.
//
//    For example, given s = "aab",
//    Return
//
//    [
//            ["aa","b"],
//            ["a","a","b"]
//            ]


    public List<List<String>> partition(String s) {
        List<List<String>> results = new ArrayList<>();
        if (s == null || s.length() == 0) {
            return results;
        }

        List<String> partition = new ArrayList<String>();
        helper(s, 0, partition, results);

        return results;
    }

    private void helper(String s,
                        int startIndex,
                        List<String> partition,
                        List<List<String>> results) {
        if (startIndex == s.length()) {
            results.add(new ArrayList<>(partition));
            return;
        }

        for (int i = startIndex; i < s.length(); i++) {
            String subString = s.substring(startIndex, i + 1);
            if (!isPalindrome2(subString)) {
                continue;
            }
            partition.add(subString);
            helper(s, i + 1, partition, results);
            partition.remove(partition.size() - 1);
        }
    }

    private boolean isPalindrome2(String s) {
        for (int i = 0, j = s.length() - 1; i < j; i++, j--) {
            if (s.charAt(i) != s.charAt(j)) {
                return false;
            }
        }
        return true;
    }


//    134. Gas Station
//    There are N gas stations along a circular route, where the amount of gas at station i is gas[i].
//
//    You have a car with an unlimited gas tank and it costs cost[i] of gas to
// travel from station i to its next station (i+1). You begin the journey
// with an empty tank at one of the gas stations.
//
//    Return the starting gas station's index if you can travel around the circuit once,
// otherwise return -1.


    public int canCompleteCircuit(int[] gas, int[] cost) {
        if (gas == null || cost == null || gas.length == 0 || cost.length == 0) {
            return -1;
        }

        int sum = 0;
        int total = 0;
        int index = -1;

        for(int i = 0; i<gas.length; i++) {
            sum += gas[i] - cost[i];
            total += gas[i] - cost[i];
            if(sum < 0) {
                index = i;
                sum = 0;
            }
        }
        return total < 0 ? -1 : index + 1;
        // index should be updated here for cases ([5], [4]);
        // total < 0 is for case [2], [2]
    }


//    135. Candy
//
//    There are N children standing in a line. Each child is assigned a rating value.
//
//    You are giving candies to these children subjected to the following requirements:
//
//    Each child must have at least one candy.
//    Children with a higher rating get more candies than their neighbors.
//    What is the minimum candies you must give?

    public int candy(int[] ratings) {
        if(ratings == null || ratings.length == 0) {
            return 0;
        }

        int[] count = new int[ratings.length];
        Arrays.fill(count, 1);
        int sum = 0;
        for(int i = 1; i < ratings.length; i++) {
            if(ratings[i] > ratings[i - 1]) {
                count[i] = count[i - 1] + 1;
            }
        }

        for(int i = ratings.length - 1; i >= 1; i--) {
            sum += count[i];
            if(ratings[i - 1] > ratings[i] && count[i - 1] <= count[i]) {  // second round has two conditions
                count[i-1] = count[i] + 1;
            }
        }
        sum += count[0];
        return sum;
    }


//    136. Single Number
//    Given an array of integers, every element appears twice except for one. Find that single one.
//
//    Note:
//    Your algorithm should have a linear runtime complexity.
// Could you implement it without using extra memory?

    public int singleNumber(int[] A) {
        if(A == null || A.length == 0) {
            return -1;
        }
        int rst = 0;
        for (int i = 0; i < A.length; i++) {
            rst ^= A[i];
        }
        return rst;
    }

//    137. Single Number II
//    Given an array of integers, every element appears three times except for one,
// which appears exactly once. Find that single one.

    public int singleNumber2(int[] A) {
        if (A == null || A.length == 0) {
            return -1;
        }
        int result=0;
        int[] bits=new int[32];
        for (int i = 0; i < 32; i++) {
            for(int j = 0; j < A.length; j++) {
                bits[i] += A[j] >> i & 1;
                bits[i] %= 3;
            }

            result |= (bits[i] << i);
        }
        return result;
    }

//    139. Word Break
//    Given a non-empty string s and a dictionary wordDict containing a list of non-empty words,
// determine if s can be segmented into a space-separated sequence of one or more dictionary words.
// You may assume the dictionary does not contain duplicate words.
//
//    For example, given
//    s = "leetcode",
//    dict = ["leet", "code"].
//
//    Return true because "leetcode" can be segmented as "leet code".

    public boolean wordBreak(String s, Set<String> dict) {
        if (s == null || s.length() == 0) return false;

        int n = s.length();

        // dp[i] represents whether s[0...i] can be formed by dict
        boolean[] dp = new boolean[n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                String sub = s.substring(j, i + 1);

                if (dict.contains(sub) && (j == 0 || dp[j - 1])) {
                    dp[i] = true;
                    break;
                }
            }
        }

        return dp[n - 1];
    }

    /**
    140. Word Break II
Hard

Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, add spaces in s to construct a sentence where each word is a valid dictionary word. Return all such possible sentences.

Note:

    The same word in the dictionary may be reused multiple times in the segmentation.
    You may assume the dictionary does not contain duplicate words.

Example 1:

Input:
s = "catsanddog"
wordDict = ["cat", "cats", "and", "sand", "dog"]
Output:
[
  "cats and dog",
  "cat sand dog"
]

Example 2:

Input:
s = "pineapplepenapple"
wordDict = ["apple", "pen", "applepen", "pine", "pineapple"]
Output:
[
  "pine apple pen apple",
  "pineapple pen apple",
  "pine applepen apple"
]
Explanation: Note that you are allowed to reuse a dictionary word.

Example 3:

Input:
s = "catsandog"
wordDict = ["cats", "dog", "sand", "and", "cat"]
Output:
[]

     */

    HashMap<Integer, List<String>> dp = new HashMap<>();

    public List<String> wordBreak140(String s, Set<String> wordDict) {
        int maxLength = -1;
        for(String ss : wordDict) maxLength = Math.max(maxLength, ss.length());
        return addSpaces(s, wordDict, 0, maxLength);
    }

    private List<String> addSpaces(String s, Set<String> wordDict, int start, int max){
        List<String> words = new ArrayList<>();
        if(start == s.length()) {
            words.add("");
            return words;
        }
        for(int i = start + 1; i <= max + start && i <= s.length(); i++){
            String temp = s.substring(start, i);
            if(wordDict.contains(temp)){
                List<String> ll;
                if(dp.containsKey(i)) ll = dp.get(i);
                else ll = addSpaces(s, wordDict, i, max);
                for(String ss : ll) words.add(temp + (ss.equals("") ? "" : " ") + ss);
            }

        }
        dp.put(start, words);
        return words;
    }

//    private final Map<String, List<String>> cache = new HashMap<>();
//
//    public List<String> wordBreak140(String s, Set<String> dict) {
//        if (cache.containsKey(s)) return cache.get(s);
//        List<String> result = new LinkedList<>();
//        if (dict.contains(s)) result.add(s);
//        for (int i = 1; i < s.length(); i++) {
//            String left = s.substring(0,i), right = s.substring(i);
//            if (dict.contains(left) && containsSuffix(dict, right)) {
//                for (String ss : wordBreak140(right, dict)) {
//                    result.add(left + " " + ss);
//                }
//            }
//        }
//        cache.put(s, result);
//        return result;
//    }
//
//    private boolean containsSuffix(Set<String> dict, String str) {
//        for (int i = 0; i < str.length(); i++) {
//            if (dict.contains(str.substring(i))) return true;
//        }
//        return false;
//    }

//    141. Linked List Cycle
//    Given a linked list, determine if it has a cycle in it.
//
//    Follow up:
//    Can you solve it without using extra space?

    public Boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }

        ListNode fast, slow;
        fast = head.next;
        slow = head;
        while (fast != slow) {
            if(fast==null || fast.next==null)
                return false;
            fast = fast.next.next;
            slow = slow.next;
        }
        return true;
    }


//    142. Linked List Cycle II
//    Given a linked list, return the node where the cycle begins. If there is no cycle, return null.
//
//    Note: Do not modify the linked list.

    public ListNode detectCycle(ListNode head) {
        if (head == null || head.next==null) {
            return null;
        }

        ListNode fast, slow;
        fast = head.next;
        slow = head;
        while (fast != slow) {
            if(fast==null || fast.next==null)
                return null;
            fast = fast.next.next;
            slow = slow.next;
        }

        while (head != slow.next) {
            head = head.next;
            slow = slow.next;
        }
        return head;
    }



//    143. Reorder List
//    Given a singly linked list L: L0→L1→…→Ln-1→Ln,
//    reorder it to: L0→Ln→L1→Ln-1→L2→Ln-2→…
//
//    You must do this in-place without altering the nodes' values.
//
//    For example,
//    Given {1,2,3,4}, reorder it to {1,4,2,3}.

    private ListNode reverse2(ListNode head) {
        ListNode newHead = null;
        while (head != null) {
            ListNode temp = head.next;
            head.next = newHead;
            newHead = head;
            head = temp;
        }
        return newHead;
    }

    private void merge2(ListNode head1, ListNode head2) {
        int index = 0;
        ListNode dummy = new ListNode(0);
        while (head1 != null && head2 != null) {
            if (index % 2 == 0) {
                dummy.next = head1;
                head1 = head1.next;
            } else {
                dummy.next = head2;
                head2 = head2.next;
            }
            dummy = dummy.next;
            index ++;
        }
        if (head1 != null) {
            dummy.next = head1;
        } else {
            dummy.next = head2;
        }
    }

    private ListNode findMiddle(ListNode head) {
        ListNode slow = head, fast = head.next;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }

    public void reorderList(ListNode head) {
        if (head == null || head.next == null) {
            return;
        }

        ListNode mid = findMiddle(head);
        ListNode tail = reverse2(mid.next);
        mid.next = null;

        merge2(head, tail);
    }

//    144. Binary Tree Preorder Traversal

//    public List<Integer> preorderTraversal(TreeNode root) {
//        Stack<TreeNode> stack = new Stack<TreeNode>();
//        List<Integer> preorder = new ArrayList<Integer>();
//
//        if (root == null) {
//            return preorder;
//        }
//
//        stack.push(root);
//        while (!stack.empty()) {
//            TreeNode node = stack.pop();
//            preorder.add(node.val);
//            if (node.right != null) {
//                stack.push(node.right);
//            }
//            if (node.left != null) {
//                stack.push(node.left);
//            }
//        }
//
//        return preorder;
//    }

    public ArrayList<Integer> preorderTraversal2(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<Integer>();
        traverse(root, result);
        return result;
    }
    private void traverse(TreeNode root, ArrayList<Integer> result) {
        if (root == null) {
            return;
        }

        result.add(root.val);
        traverse(root.left, result);
        traverse(root.right, result);
    }


    //    145. Binary Tree Postorder Traversal
    //Recursive
    public ArrayList<Integer> postorderTraversal(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<Integer>();
        traverse2(root,result);
        return result;
    }

    private void traverse2(TreeNode root, ArrayList<Integer> result) {
        if (root == null) {
            return;
        }
        traverse2(root.left, result);
        traverse2(root.right, result);
        result.add(root.val);
    }

    //Iterative
    public ArrayList<Integer> postorderTraversal2(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<Integer>();
        Stack<TreeNode> stack = new Stack<TreeNode>();
        TreeNode prev = null; // previously traversed node
        TreeNode curr = root;

        if (root == null) {
            return result;
        }

        stack.push(root);
        while (!stack.empty()) {
            curr = stack.peek();
            if (prev == null || prev.left == curr || prev.right == curr) { // traverse down the tree
                if (curr.left != null) {
                    stack.push(curr.left);
                } else if (curr.right != null) {
                    stack.push(curr.right);
                }
            } else if (curr.left == prev) { // traverse up the tree from the left
                if (curr.right != null) {
                    stack.push(curr.right);
                }
            } else { // traverse up the tree from the right
                result.add(curr.val);
                stack.pop();
            }
            prev = curr;
        }

        return result;
    }
//    147. Insertion Sort List
//    Sort a linked list using insertion sort.

    public ListNode insertionSortList(ListNode head) {
        ListNode dummy = new ListNode(0);
        // 这个dummy的作用是，把head开头的链表一个个的插入到dummy开头的链表里
        // 所以这里不需要dummy.next = head;

        while (head != null) {
            ListNode node = dummy;
            while (node.next != null && node.next.val < head.val) {
                node = node.next;
            }
            ListNode temp = head.next;
            //insert head into [node, node.next]
            head.next = node.next;
            node.next = head;

            head = temp;
        }

        return dummy.next;
    }

//    148. Sort List
//    Sort a linked list in O(n log n) time using constant space complexity.

    private ListNode findMiddle2(ListNode head) {
        ListNode slow = head, fast = head.next;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }

    private ListNode merge3(ListNode head1, ListNode head2) {
        ListNode dummy = new ListNode(0);
        ListNode tail = dummy;
        while (head1 != null && head2 != null) {
            if (head1.val < head2.val) {
                tail.next = head1;
                head1 = head1.next;
            } else {
                tail.next = head2;
                head2 = head2.next;
            }
            tail = tail.next;
        }
        if (head1 != null) {
            tail.next = head1;
        } else {
            tail.next = head2;
        }

        return dummy.next;
    }

    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }

        ListNode mid = findMiddle2(head);

        ListNode right = sortList(mid.next);
        mid.next = null;
        ListNode left = sortList(head);

        return merge3(left, right);
    }


//    150. Evaluate Reverse Polish Notation
//    Evaluate the value of an arithmetic expression in Reverse Polish Notation.
//
//    Valid operators are +, -, *, /. Each operand may be an integer or another expression.
//
//    Some examples:
//            ["2", "1", "+", "3", "*"] -> ((2 + 1) * 3) -> 9
//            ["4", "13", "5", "/", "+"] -> (4 + (13 / 5)) -> 6

    public int evalRPN(String[] tokens) {
        Stack<Integer> s = new Stack<Integer>();
        String operators = "+-*/";
        for(String token : tokens){
            if(!operators.contains(token)){
                s.push(Integer.valueOf(token));
                continue;
            }

            int a = s.pop();
            int b = s.pop();
            if(token.equals("+")) {
                s.push(b + a);
            } else if(token.equals("-")) {
                s.push(b - a);
            } else if(token.equals("*")) {
                s.push(b * a);
            } else {
                s.push(b / a);
            }
        }
        return s.pop();
    }


//    151. Reverse Words in a String
    //    186 Reverse Words in a String II
//    Given an input string, reverse the string word by word.
//
//            For example,
//    Given s = "the sky is blue",
//    return "blue is sky the".
//
//    Update (2015-02-12):
//    For C programmers: Try to solve it in-place in O(1) space.

    public String reverseWords(String s) {
        if (s == null || s.length() == 0) {
            return "";
        }
        String[] array = s.split(" ");
        StringBuilder sb = new StringBuilder();

        for (int i = array.length - 1; i >= 0; --i) {
            if (!array[i].equals("")) {
                sb.append(array[i]).append(" ");
            }
        }

        //remove the last " "
        return sb.length() == 0 ? "" : sb.substring(0, sb.length() - 1);
    }

//    152. Maximum Product Subarray
//    Find the contiguous subarray within an array (containing at least one number)
// which has the largest product.
//
//    For example, given the array [2,3,-2,4],
//    the contiguous subarray [2,3] has the largest product = 6.

    public int maxProduct(int[] a) {
        if (a == null || a.length == 0)
            return 0;

        int res = a[0], min = res, max = res;

        for (int i = 1; i < a.length; i++) {
            if (a[i] >= 0) {
                max = Math.max(a[i], max * a[i]);
                min = Math.min(a[i], min * a[i]);
            } else {
                int tmp = max;
                max = Math.max(a[i], min * a[i]);
                min = Math.min(a[i], tmp * a[i]);
            }

            res = Math.max(res, max);
        }

        return res;
    }


//    153. Find Minimum in Rotated Sorted Array
//    Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.
//
//            (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
//
//    Find the minimum element.
//
//    You may assume no duplicate exists in the array.

    public int findMin(int[] num) {
        if (num == null || num.length == 0) {
            return 0;
        }
        if (num.length == 1) {
            return num[0];
        }
        int start = 0, end = num.length - 1;
        while (start < end) {
            int mid = (start + end) / 2;
            if (mid > 0 && num[mid] < num[mid - 1]) {
                return num[mid];
            }
            if (num[start] <= num[mid] && num[mid] > num[end]) {
                start = mid + 1;
            } else {
                end = mid - 1;
            }
        }
        return num[start];
    }


//    154. Find Minimum in Rotated Sorted Array II
//    Follow up for "Find Minimum in Rotated Sorted Array":
//    What if duplicates are allowed?
//
//    Would this affect the run-time complexity? How and why?
//    Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.
//
//            (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
//
//    Find the minimum element.
//
//    The array may contain duplicates.


    //O(n)
    public int findMin2(int[] nums) {
        int l = 0, r = nums.length-1;
        while (l < r) {
            int mid = (l + r) / 2;
            if (nums[mid] < nums[r]) {
                r = mid;
            } else if (nums[mid] > nums[r]){
                l = mid + 1;
            } else {
                r--;  //nums[mid]=nums[r] no idea, but we can eliminate nums[r];
            }
        }
        return nums[l];
    }


//    155. Min Stack
//    Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.
//
//            push(x) -- Push element x onto stack.
//            pop() -- Removes the element on top of the stack.
//    top() -- Get the top element.
//    getMin() -- Retrieve the minimum element in the stack.
//            Example:
//    MinStack minStack = new MinStack();
//    minStack.push(-2);
//    minStack.push(0);
//    minStack.push(-3);
//    minStack.getMin();   --> Returns -3.
//    minStack.pop();
//    minStack.top();      --> Returns 0.
//    minStack.getMin();   --> Returns -2.

    public class MinStack {
        private Stack<Integer> stack;
        private Stack<Integer> minStack;

        public MinStack() {
            stack = new Stack<Integer>();
            minStack = new Stack<Integer>();
        }

        public void push(int number) {
            stack.push(number);
            if (minStack.isEmpty()) {
                minStack.push(number);
            } else {
                minStack.push(Math.min(number, minStack.peek()));
            }
        }

        public int pop() {
            minStack.pop();
            return stack.pop();
        }

        public int min() {
            return minStack.peek();
        }
    }




//    157. Read N Characters Given Read4
//    Leetcode: Read N Characters Given Read4
//    The API: int read4(char *buf) reads 4 characters at a time from a file.
//            The return value is the actual number of characters read.
// For example, it returns 3 if there is only 3 characters left in the file.
//    By using the read4 API, implement the function int read(char *buf, int n)
// that reads n characters from the file.
//    Note:
//    The read function will only be called once for each test case.
//    Understand the problem:
//    This seemingly easy coding question has some tricky edge cases. When read4 returns
//    less than 4, we know it must reached the end of file. However, take note that read4
//    returning 4 could mean the last 4 bytes of the file.
//
//    To make sure that the buffer is not copied more than n bytes, copy the remaining bytes
//            (n – readBytes) or the number of bytes read, whichever is smaller.

    public int read4(char[] buf){return 0;}
    public int read(char[] buf, int n) {
        char[] buf4 = new char[4];
        int offset = 0;

        while (true) {
            int size = read4(buf4);
            for (int i = 0; i < size && offset < n; i++) {
                buf[offset++] = buf4[i];
            }
            if (size == 0 || offset == n) {
                return offset;
            }
        }
    }



//    158 Read N Characters Given Read4 II - Call multiple times
//    The API: int read4(char *buf) reads 4 characters at a time from a file.
//
//            The return value is the actual number of characters read.
// For example, it returns 3 if there is only 3 characters left in the file.
//
//    By using the read4 API, implement the function int read(char *buf, int n)
// that reads n characters from the file.
//
//    Note:
//    The read function may be called multiple times.


    private char[] buf4 = new char[4];
    private int buf4Index = 4;
    private int buf4Size = 4;

    private boolean readNext(char[] buf, int index) {
        if (buf4Index >= buf4Size) {
            buf4Size = read4(buf4);
            buf4Index = 0;
            if (buf4Size == 0) {
                return false;
            }
        }

        buf[index] = buf4[buf4Index++];
        return true;
    }


    /**
     * @param buf Destination buffer
     * @param n   Maximum number of characters to read
     * @return    The number of characters read
     */
    public int read2(char[] buf, int n) {
        for (int i = 0; i < n; i++) {
            if (!readNext(buf, i)) {
                return i;
            }
        }

        return n;
    }

    /**
     * 159. Longest Substring with At Most Two Distinct Characters
     Given a string s , find the length of the longest substring t
     that contains at most 2 distinct characters.
     Example 1:
     Input: "eceba"
     Output: 3
     Explanation: t is "ece" which its length is 3.
     Example 2:
     Input: "ccaabbb"
     Output: 5
     Explanation: t is "aabbb" which its length is 5.
     */

    public int lengthOfLongestSubstringTwoDistinct(String s) {
        if (s.length() < 1) {
            return 0;
        }
        Map<Character, Integer> index = new HashMap<>();
        int lo = 0;
        int hi = 0;
        int maxLength = 0;
        while (hi < s.length()) {
            if (index.size() <= 2) {
                char c = s.charAt(hi); // keep the last position of a char
                index.put(c, hi);
                hi++;
            }
            if (index.size() > 2) {
                //int leftMost = s.length();
                int leftMost  = Collections.min(index.values());
//                for (int i : index.values()) {
//                    leftMost = Math.min(leftMost, i);
//                }
//                char c = s.charAt(leftMost);
                index.remove(s.charAt(leftMost));
                lo = leftMost + 1;
            }
            maxLength = Math.max(maxLength, hi - lo);
        }
        return maxLength;
    }


//    160. Intersection of Two Linked Lists
//    Write a program to find the node at which the intersection of two singly linked lists begins.
//
//
//    For example, the following two linked lists:
//
//    A:          a1 → a2
//                        ↘
//                        c1 → c2 → c3
//                        ↗
//    B:     b1 → b2 → b3
//    begin to intersect at node c1.
//
//
//    Notes:
//
//    If the two linked lists have no intersection at all, return null.
//    The linked lists must retain their original structure after the function returns.
//    You may assume there are no cycles anywhere in the entire linked structure.
//    Your code should preferably run in O(n) time and use only O(1) memory.

    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) {
            return null;
        }

        // get the tail of list A.
        ListNode node = headA;
        while (node.next != null) {
            node = node.next;
        }
        node.next = headB;

        ListNode result = listCycleII(headA);
        node.next = null;
        return result;
    }

    private ListNode listCycleII(ListNode head) {
        ListNode slow = head, fast = head.next;

        while (slow != fast) {
            if (fast == null || fast.next == null) {
                return null;
            }
            slow = slow.next;
            fast = fast.next.next;
        }

        while (head != slow.next) {
            head = head.next;
            slow = slow.next;
        }

        return head;
    }


    //    161 One Edit Distance
    /*
 * There're 3 possibilities to satisfy one edit distance apart:
 *
 * 1) Replace 1 char:
 	  s: a B c
 	  t: a D c
 * 2) Delete 1 char from s:
	  s: a D  b c
	  t: a    b c
 * 3) Delete 1 char from t
	  s: a   b c
	  t: a D b c
 */
    public boolean isOneEditDistance(String s, String t) {
        for (int i = 0; i < Math.min(s.length(), t.length()); i++) {
            if (s.charAt(i) != t.charAt(i)) {
                if (s.length() == t.length()) // s has the same length as t, so the only possibility is replacing one char in s and t
                    return s.substring(i + 1).equals(t.substring(i + 1));
                else if (s.length() < t.length()) // t is longer than s, so the only possibility is deleting one char from t
                    return s.substring(i).equals(t.substring(i + 1));
                else // s is longer than t, so the only possibility is deleting one char from s
                    return t.substring(i).equals(s.substring(i + 1));
            }
        }
        //All previous chars are the same, the only possibility is deleting the end char in the longer one of s and t
        return Math.abs(s.length() - t.length()) == 1;
    }


//    162. Find Peak Element
//    A peak element is an element that is greater than its neighbors.
//
//    Given an input array where num[i] ≠ num[i+1], find a peak element and return its index.
//
//    The array may contain multiple peaks, in that case return the index to
// any one of the peaks is fine.
//
//    You may imagine that num[-1] = num[n] = -∞.
//
//    For example, in array [1, 2, 3, 1], 3 is a peak element and your function
// should return the index number 2.


    public int findPeak(int[] A)
        {
            int low = 0;
            int high = A.length-1;

            while(low < high)
            {
                int mid1 = (low+high)/2;
                int mid2 = mid1+1;
                if(A[mid1] < A[mid2])
                    low = mid2;
                else
                    high = mid1;
            }
            return low;
        }


    //    163 Missing Ranges
//Given a sorted integer array where the range of elements are [lower, upper] inclusive,
// return its missing ranges.
//
//    For example, given [0, 1, 3, 50, 75], lower = 0 and upper = 99,
// return ["2", "4->49", "51->74", "76->99"].

    //a[i]-a[i-1], plus lower and upper, then it is OK
    public List<String> findMissingRanges(int[] A, int lower, int upper) {
        List<String> result = new ArrayList<String>();
        int pre = lower - 1;

        for(int i = 0 ; i <= A.length; i++){// good when i <= A.length
            int after = i == A.length ? upper + 1 : A[i];
            if(pre + 2 == after){
                result.add(String.valueOf(pre + 1));
            }else if(pre + 2 < after){
                result.add(String.valueOf(pre + 1) + "->" + String.valueOf(after - 1));
            }
            pre = after;
        }
        return result;
    }


//    164. Maximum Gap

    /**
    Given an unsorted array, find the maximum difference
    between the successive elements in its sorted form.

Return 0 if the array contains less than 2 elements.

Example 1:

Input: [3,6,9,1]
Output: 3
Explanation: The sorted form of the array is [1,3,6,9], either
             (3,6) or (6,9) has the maximum difference 3.

Example 2:

Input: [10]
Output: 0
Explanation: The array contains less than 2 elements, therefore return 0.

Note:

    You may assume all elements in the array are non-negative integers and
    fit in the 32-bit signed integer range.
    Try to solve it in linear time/space.
     */

//    Suppose there are N elements in the array,
// the min value is min and the max value is max.
// Then the maximum gap will be no smaller than ceiling[(max - min ) / (N - 1)].
//
//    Let gap = ceiling[(max - min ) / (N - 1)].
// We divide all numbers in the array into n-1 buckets,
// where k-th bucket contains all numbers in [min + (k-1)gap, min + k*gap).
// Since there are n-2 numbers that are not equal min or max and there are n-1 buckets,
// at least one of the buckets are empty. We only need to store the largest number
// and the smallest number in each bucket.
//
//    After we put all the numbers into the buckets.
// We can scan the buckets sequentially and get the max gap.
//    my blog for this problem

    public int maximumGap(int[] num) {
        if (num == null || num.length < 2)
            return 0;
        // get the max and min value of the array
        int min = num[0];
        int max = num[0];
        for (int i:num) {
            min = Math.min(min, i);
            max = Math.max(max, i);
        }
        // the minimum possibale gap, ceiling of the integer division
        int gap = Math.max(1, (max - min) / (num.length  - 1));
//        int gap = (int)Math.ceil((double)(max - min)/(num.length - 1));
        int[] bucketsMIN = new int[num.length - 1]; // store the min value in that bucket
        int[] bucketsMAX = new int[num.length - 1]; // store the max value in that bucket
        Arrays.fill(bucketsMIN, Integer.MAX_VALUE);
        Arrays.fill(bucketsMAX, Integer.MIN_VALUE);
        // put numbers into buckets
        for (int i:num) {
            if (i == min || i == max)
                continue;
            int idx = (i - min) / gap; // index of the right position in the buckets
            bucketsMIN[idx] = Math.min(i, bucketsMIN[idx]);
            bucketsMAX[idx] = Math.max(i, bucketsMAX[idx]);
        }
        // scan the buckets for the max gap
        int maxGap = Integer.MIN_VALUE;
        int previous = min;
        for (int i = 0; i < num.length - 1; i++) {
            if (bucketsMIN[i] == Integer.MAX_VALUE && bucketsMAX[i] == Integer.MIN_VALUE)
                // empty bucket
                continue;
            // min value minus the previous value is the current gap
            maxGap = Math.max(maxGap, bucketsMIN[i] - previous);
            // update previous bucket value
            previous = bucketsMAX[i];
        }
        maxGap = Math.max(maxGap, max - previous); // updata the final max value gap
        return maxGap;
    }



//    165. Compare Version Numbers
//    Compare two version numbers version1 and version2.
//    If version1 > version2 return 1, if version1 < version2 return -1, otherwise return 0.
//
//    You may assume that the version strings are non-empty and contain only digits and the . character.
//    The . character does not represent a decimal point and is used to separate number sequences.
//    For instance, 2.5 is not "two and a half" or "half way to version three",
// it is the fifth second-level revision of the second first-level revision.
//
//    Here is an example of version numbers ordering:
//
//            0.1 < 1.1 < 1.2 < 13.37

    public int compareVersion(String version1, String version2) {
        String[] levels1 = version1.split("\\.");
        String[] levels2 = version2.split("\\.");

        int length = Math.max(levels1.length, levels2.length);
        for (int i=0; i<length; i++) {
            Integer v1 = i < levels1.length ? Integer.parseInt(levels1[i]) : 0;
            Integer v2 = i < levels2.length ? Integer.parseInt(levels2[i]) : 0;
            int compare = v1.compareTo(v2);
            if (compare != 0) {
                return compare;
            }
        }

        return 0;
    }


//    166. Fraction to Recurring Decimal
//    Given two integers representing the numerator and denominator of a fraction,
// return the fraction in string format.
//
//    If the fractional part is repeating, enclose the repeating part in parentheses.
//
//            For example,
//
//    Given numerator = 1, denominator = 2, return "0.5".
//    Given numerator = 2, denominator = 1, return "2".
//    Given numerator = 2, denominator = 3, return "0.(6)".


    public String fractionToDecimal(int numerator, int denominator) {
        if (numerator == 0)
            return "0";
        if (denominator == 0)
            return "";

        String result = "";

        // is result is negative
        if ((numerator < 0) ^ (denominator < 0)) {
            result += "-";
        }

        // convert int to long
        long num = numerator, den = denominator;
        num = Math.abs(num);
        den = Math.abs(den);

        // quotient
        long res = num / den;
        result += String.valueOf(res);

        // if remainder is 0, return result
        long remainder = (num % den) * 10;
        if (remainder == 0)
            return result;

        // right-hand side of decimal point
        HashMap<Long, Integer> map = new HashMap<Long, Integer>();
        result += ".";
        while (remainder != 0) {
            // if digits repeat
            if (map.containsKey(remainder)) {
                int beg = map.get(remainder);
                String part1 = result.substring(0, beg);
                String part2 = result.substring(beg);
                result = part1 + "(" + part2 + ")";
                return result;
            }

            // continue
            map.put(remainder, result.length());
            res = remainder / den;
            result += String.valueOf(res);
            remainder = (remainder % den) * 10;
        }

        return result;
    }


//    167. Two Sum II - Input array is sorted
//    Given an array of integers that is already sorted in ascending order,
// find two numbers such that they add up to a specific target number.
//
//    The function twoSum should return indices of the two numbers
// such that they add up to the target, where index1 must be less than index2.
// Please note that your returned answers (both index1 and index2) are not zero-based.
//
//    You may assume that each input would have exactly one solution
// and you may not use the same element twice.
//
//    Input: numbers={2, 7, 11, 15}, target=9
//    Output: index1=1, index2=2

    public int[] twoSum2(int[] numbers, int target) {
        if (numbers == null || numbers.length == 0)
            return null;

        int i = 0;
        int j = numbers.length - 1;

        while (i < j) {
            int x = numbers[i] + numbers[j];
            if (x < target) {
                ++i;
            } else if (x > target) {
                j--;
            } else {
                return new int[] { i + 1, j + 1 };
            }
        }

        return null;
    }

//    168. Excel Sheet Column Title
//    Given a positive integer, return its corresponding column title as appear in an Excel sheet.
//
//    For example:
//
//            1 -> A
//    2 -> B
//    3 -> C
//    ...
//            26 -> Z
//    27 -> AA
//    28 -> AB

    String convertToTitle(int n) {
        if (n == 0) {
            return "";
        }
        return convertToTitle((n - 1) / 26) + (char)((n - 1) % 26 + 'A');
    }


//    169. Majority Element
//    Given an array of size n, find the majority element.
// The majority element is the element that appears more than ⌊ n/2 ⌋ times.
//
//    You may assume that the array is non-empty and the majority element always exist in the array.

    public int majorityElement(int[] nums) {
        int result = 0, count = 0;

        for(int i = 0; i<nums.length; i++ ) {
            if(count == 0){
                result = nums[ i ];
                count = 1;
            }else if(result == nums[i]){
                count++;
            }else{
                count--;
            }
        }

        return result;
    }

//    170. Two Sum III – Data structure design (Java)
//
//    Design and implement a TwoSum class. It should support the following operations: add and find.
//
//            add - Add the number to an internal data structure.
//    find - Find if there exists any pair of numbers which sum is equal to the value.
//
//            For example,
//
//    add(1);
//    add(3);
//    add(5);
//    find(4) -> true
//    find(7) -> false

    public class TwoSum {
        private HashMap<Integer, Integer> elements = new HashMap<Integer, Integer>();

        public void add(int number) {
            if (elements.containsKey(number)) {
                elements.put(number, elements.get(number) + 1);
            } else {
                elements.put(number, 1);
            }
        }

        public boolean find(int value) {
            for (Integer i : elements.keySet()) {
                int target = value - i;
                if (elements.containsKey(target)) {
                    if (i == target && elements.get(target) < 2) {
                        continue;
                    }
                    return true;
                }
            }
            return false;
        }
    }


//    171. Excel Sheet Column Number
//    Given a column title as appear in an Excel sheet, return its corresponding column number.
//
//    For example:
//
//    A -> 1
//    B -> 2
//    C -> 3
//            ...
//    Z -> 26
//    AA -> 27
//    AB -> 28

    public int titleToNumber(String s) {
        int result = 0;
        for(int i = 0 ; i < s.length(); i++) {
            result = result * 26 + (s.charAt(i) - 'A' + 1);
        }
        return result;
    }


//    172. Factorial Trailing Zeroes
//    Given an integer n, return the number of trailing zeroes in n!.
//
//    Note: Your solution should be in logarithmic time complexity.

    public int trailingZeroes(int n) {
        if (n < 0)
            return -1;

        int count = 0;
        while(n>1){
            count+=n/5;
            n/=5;
        }

        return count;
    }


//    173. Binary Search Tree Iterator
//    Implement an iterator over a binary search tree (BST).
// Your iterator will be initialized with the root node of a BST.
//
//    Calling next() will return the next smallest number in the BST.
//
//            Note: next() and hasNext() should run in average O(1) time
// and uses O(h) memory, where h is the height of the tree.


    private Stack<TreeNode> stack = new Stack<>();
    private TreeNode curt;

    //@return: True if there has next node, or false
    public boolean hasNext() {
        return (curt != null || !stack.isEmpty());
    }

    //@return: return next node
    public TreeNode next() {
        while (curt != null) {
            stack.push(curt);
            curt = curt.left;
        }

        curt = stack.pop();
        TreeNode node = curt;
        curt = curt.right;

        return node;
    }


    /**
     * 174. Dungeon Game
     The demons had captured the princess (P) and imprisoned her
     in the bottom-right corner of a dungeon.
     The dungeon consists of M x N rooms laid out in a 2D grid.
     Our valiant knight (K) was initially positioned in the top-left room
     and must fight his way through the dungeon to rescue the princess.
     The knight has an initial health point represented by a positive integer.
     If at any point his health point drops to 0 or below, he dies immediately.
     Some of the rooms are guarded by demons, so the knight loses health (negative integers)
     upon entering these rooms; other rooms are either empty (0's)
     or contain magic orbs that increase the knight's health (positive integers).
     In order to reach the princess as quickly as possible, the knight decides
     to move only rightward or downward in each step.
     Write a function to determine the knight's minimum initial health
     so that he is able to rescue the princess.
     For example, given the dungeon below, the initial health of the knight
     must be at least 7 if he follows the optimal path RIGHT-> RIGHT -> DOWN -> DOWN.
     -2 (K) 	-3 	3
     -5 	-10 	1
     10 	30  	-5 (P)
     Note:
     The knight's health has no upper bound.
     Any room can contain threats or power-ups, even the first room the knight enters and the bottom-right room where the princess is imprisoned.
     */
    /** This problem should fill the dp matrix from bottom right. */
    public int calculateMinimumHP(int[][] dungeon) {
        if (dungeon == null || dungeon.length == 0) {
            return 0;
        }

        int height = dungeon.length;
        int width = dungeon[0].length;
        int[][] dp = new int[height][width];
        dp[height - 1][width - 1] =
                (dungeon[height - 1][width - 1] > 0) ? 1 : 1 - dungeon[height - 1][width - 1];

        //fill the last column
        for (int i = height - 2; i >= 0; i--) {
            int temp = dp[i + 1][width - 1] - dungeon[i][width - 1];
            dp[i][width - 1] = Math.max(1, temp);
        }

        //fill the last row
        for (int j = width - 2; j >= 0; j--) {
            int temp = dp[height - 1][j + 1] - dungeon[height - 1][j];
            dp[height - 1][j] = Math.max(temp, 1);
        }

        for (int i = height - 2; i >= 0; i--) {
            for (int j = width - 2; j >= 0; j--) {
                int down = Math.max(1, dp[i + 1][j] - dungeon[i][j]);
                int right = Math.max(1, dp[i][j + 1] - dungeon[i][j]);
                dp[i][j] = Math.min(down, right);
            }
        }

//        System.out.println(dp);
        return dp[0][0];
    }






//    179. Largest Number
//    Given a list of non negative integers, arrange them such that they form the largest number.
//
//    For example, given [3, 30, 34, 5, 9], the largest formed number is 9534330.
//
//    Note: The result may be very large, so you need to return a string instead of an integer.


    public String largestNumber(int[] nums) {
        String[] strs = new String[nums.length];
        for (int i = 0; i < nums.length; i++) {
            strs[i] = Integer.toString(nums[i]);
        }
        // 5, 51   "5"<"51"  we need 551 not 515
        Arrays.sort(strs, (s1, s2)-> (s2+s1).compareTo(s1+s2));
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < strs.length; i++) {
            sb.append(strs[i]);
        }
        String result = sb.toString();
        int index = 0;
        //remove 0 at the beginning of a string.
        while (index < result.length() && result.charAt(index) == '0') {
            index++;
        }
        if (index == result.length()) {
            return "0";
        }
        return result.substring(index);
    }



//    186 Reverse Words in a String II
    //    151. Reverse Words in a String

//    Given an input string, reverse the string word by word.
// A word is defined as a sequence of non-space characters.
//
//    The input string does not contain leading or trailing spaces
// and the words are always separated by a single space.
//
//    For example,
//    Given s = "the sky is blue",
//    return "blue is sky the".
//
//    Could you do it in-place without allocating extra space?

    public void reverseWords(char[] s) {
        int i=0;
        for(int j=0; j<s.length; j++){
            if(s[j]==' '){
                reverse(s, i, j-1);
                i=j+1;
            }
        }

        reverse(s, i, s.length-1);

        reverse(s, 0, s.length-1);
    }

    public void reverse(char[] s, int i, int j){
        while(i<j){
            char temp = s[i];
            s[i]=s[j];
            s[j]=temp;
            i++;
            j--;
        }
    }


//    187. Repeated DNA Sequences
//    All DNA is composed of a series of nucleotides abbreviated as A, C, G, and T,
// for example: "ACGAATTCCG". When studying DNA, it is sometimes useful
// to identify repeated sequences within the DNA.
//
//    Write a function to find all the 10-letter-long sequences (substrings)
// that occur more than once in a DNA molecule.
//
//    For example,
//
//    Given s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT",
//
//    Return:
//            ["AAAAACCCCC", "CCCCCAAAAA"].

    public int encode(String s) {
        int sum = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == 'A') {
                sum = sum * 4;
            } else if (s.charAt(i) == 'C') {
                sum = sum * 4 + 1;
            } else if (s.charAt(i) == 'G') {
                sum = sum * 4 + 2;
            } else {
                sum = sum * 4 + 3;
            }
        }
        return sum;
    }
    public List<String> findRepeatedDnaSequences(String s) {
        HashSet<Integer> hash = new HashSet<Integer>();
        HashSet<String> dna = new HashSet<String>();
        for (int i = 9; i < s.length(); i++) {
            String subString = s.substring(i - 9, i + 1);
            int encoded = encode(subString);
            if (hash.contains(encoded)) {
                dna.add(subString);
            } else {
                hash.add(encoded);
            }
        }
        return new ArrayList<>(dna);
//        List<String> result = new ArrayList<String>(dna);
//        for (String d: dna) {
//            result.add(d);
//        }
//        return result;
    }




//    189. Rotate Array
//    Rotate an array of n elements to the right by k steps.
//
//    For example, with n = 7 and k = 3, the array [1,2,3,4,5,6,7] is rotated to [5,6,7,1,2,3,4].

    private void reverse2(int[] nums, int start, int end) {
        while (start < end) {
            int temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
            start++; end--;
        }
    }
    public void rotate(int[] nums, int k) {
        if (nums.length == 0) {
            return;
        }

        k = k % nums.length;
        reverse2(nums, 0, nums.length - k - 1);
        reverse2(nums, nums.length - k, nums.length - 1);
        reverse2(nums, 0, nums.length - 1);
    }

//    190. Reverse Bits// move tail to head
//    Reverse bits of a given 32 bits unsigned integer.
//
//    For example, given input 43261596 (represented in binary as
//                                            00000010100101000001111010011100),
// return 964176192 (represented in binary as 00111001011110000010100101000000).
//
//    Follow up:
//    If this function is called many times, how would you optimize it?

    public int reverseBits(int n) {
        int reversed = 0;
        for (int i = 0; i < 32; i++) {
            //(n & 1) get the last bit
            //(reversed << 1) empty last bit
            //merge new last bit into empty last bit. (reversed << 1) | (n & 1);
            reversed = (reversed << 1) | (n & 1);
            n = (n >> 1);
        }
        return reversed;
    }

//    191. Number of 1 Bits
//    Write a function that takes an unsigned integer and
// returns the number of ’1' bits it has (also known as the Hamming weight).
//
//    For example, the 32-bit integer ’11' has binary
// representation 00000000000000000000000000001011, so the function should return 3.


    //Integer.bitCount(n);

    public int hammingWeight(int n) {
        // sol1: n = n & (n-1): remove last 1 in n:
        int res = 0;
        while(n != 0) {
            n &= (n-1); // remove the last 1 of n
            res++;
        }
        return res;
    }

    public int hammingWeight2(int n) {
        //sol2: n & 1: get 1 from the tail
        int res = 0;
        for(int i = 0; i < 32; i++) {
            res += n & 1;
            n >>= 1;
        }
        return res;
    }


    /**
    192. Word Frequency
Write a bash script to calculate the frequency of each word in a text file words.txt.
For simplicity sake, you may assume:
words.txt contains only lowercase characters and space ' ' characters.
Each word must consist of lowercase characters only.
Words are separated by one or more whitespace characters.
Example:
Assume that words.txt has the following content:
the day is sunny the the
the sunny is is
Your script should output the following, sorted by descending frequency:
the 4
is 3
sunny 2
day 1
Note:
Don't worry about handling ties, it is guaranteed that each word's frequency count is unique.
Could you write it in one-line using Unix pipes?

cat words.txt | tr -s ' ' '\n' | sort | uniq -c | sort -r | awk '{ print $2, $1 }'

     */


//    198. House Robber
//    You are a professional robber planning to rob houses along a street.
// Each house has a certain amount of money stashed, the only constraint stopping you
// from robbing each of them is that adjacent houses have security system connected
// and it will automatically contact the police if two adjacent houses were broken
// into on the same night.
//
//    Given a list of non-negative integers representing the amount of money of each house,
// determine the maximum amount of money you can rob tonight without alerting the police.

    public static int rob(int[] nums)
    {
        int ifRobbedPrevious = 0; 	// max monney can get if rob current house
        int ifDidntRobPrevious = 0; // max money can get if not rob current house

        // We go through all the values, we maintain two counts, 1) if we rob this cell, 2) if we didn't rob this cell
        for(int i=0; i < nums.length; i++)
        {
            // If we rob current cell, previous cell shouldn't be robbed. So, add the current value to previous one.
            int currRobbed = ifDidntRobPrevious + nums[i];

            // If we don't rob current cell, then the count should be max of the previous cell robbed and not robbed
            int currNotRobbed = Math.max(ifDidntRobPrevious, ifRobbedPrevious);

            // Update values for the next round
            ifDidntRobPrevious  = currNotRobbed;
            ifRobbedPrevious = currRobbed;
        }

        return Math.max(ifRobbedPrevious, ifDidntRobPrevious);
    }


//    199. Binary Tree Right Side View
//
//    Given a binary tree, imagine yourself standing on the right side of it,
// return the values of the nodes you can see ordered from top to bottom.
//
//    For example:
//    Given the following binary tree,
//            1            <---
//            /   \
//            2     3         <---
//            \     \
//            5     4       <---
//    You should return [1, 3, 4].

    private void dfs(HashMap<Integer, Integer> depthToValue, TreeNode node, int depth) {
        if (node == null) {
            return;
        }

        depthToValue.put(depth, node.val);
        dfs(depthToValue, node.left, depth + 1);
        dfs(depthToValue, node.right, depth + 1);
    }

    public List<Integer> rightSideView(TreeNode root) {
        HashMap<Integer, Integer> depthToValue = new HashMap<Integer, Integer>();
        dfs(depthToValue, root, 1);

        int depth = 1;
        List<Integer> result = new ArrayList<Integer>();
        while (depthToValue.containsKey(depth)) {
            result.add(depthToValue.get(depth));
            depth++;
        }
        return result;
    }


    /*
    200. Number of Islands
Given a 2d grid map of '1's (land) and '0's (water), count the number of islands.
An island is surrounded by water and is formed by connecting adjacent lands
horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.
Example 1:
Input:
11110
11010
11000
00000

Output: 1
Example 2:
Input:
11000
11000
00100
00011

Output: 3
     */

    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }
        int count = 0;
        int m = grid.length;
        int n = grid[0].length;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == '1') {
                    count++;
                    dfs(grid, i, j, m, n);
                }
            }
        }
        return count;
    }

    void dfs(char[][] grid, int i, int j, int m, int n) {
        if (i < 0 || i >= m || j < 0 || j >= n || grid[i][j] == '0') {
            return;
        }
        grid[i][j] = '0';
        dfs(grid, i + 1, j, m, n);
        dfs(grid, i, j + 1, m, n);
        dfs(grid, i - 1, j, m, n);
        dfs(grid, i, j - 1, m, n);
    }
}
