package com.company;

import java.math.BigInteger;
import java.util.*;
import java.util.concurrent.LinkedBlockingQueue;

public class Leetcode0to100 {

    //1. Two Sum
//    Given an array of integers, return indices of the two numbers such
// that they add up to a specific target.
//
//    You may assume that each input would have exactly one solution.
//
//            Example:
//    Given nums = [2, 7, 11, 15], target = 9,
//
//    Because nums[0] + nums[1] = 2 + 7 = 9,
//            return [0, 1].

    public int[] twoSum(int[] numbers, int target) {
        int[] result = new int[2];
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < numbers.length; i++) {
            if (map.containsKey(target - numbers[i])) {
                result[1] = i;
                result[0] = map.get(target - numbers[i]);
                return result;
            }
            map.put(numbers[i], i);
        }
        return result;
    }


    //2. Add Two Numbers
//    You are given two non-empty linked lists representing two non-negative integers.
// The digits are stored in reverse order and each of their nodes contain a single digit.
// Add the two numbers and return it as a linked list.
//
//    You may assume the two numbers do not contain any leading zero, except the number 0 itself.
//
//            Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
//    Output: 7 -> 0 -> 8




    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode c1 = l1;
        ListNode c2 = l2;
        ListNode head = new ListNode(0);
        ListNode d = head;
        int sum = 0;
        while (c1 != null || c2 != null) {
            sum /= 10;
            if (c1 != null) {
                sum += c1.val;
                c1 = c1.next;
            }
            if (c2 != null) {
                sum += c2.val;
                c2 = c2.next;
            }
            d.next = new ListNode(sum % 10);
            d = d.next;
        }
        if (sum / 10 == 1)
            d.next = new ListNode(1);
        return head.next;
    }

    //3. Longest Substring Without Repeating Characters
    //    Given a string, find the length of the longest substring without repeating characters.
//
//    Examples:
//
//    Given "abcabcbb", the answer is "abc", which the length is 3.
//
//    Given "bbbbb", the answer is "b", with the length of 1.
//
//    Given "pwwkew", the answer is "wke", with the length of 3.
// Note that the answer must be a substring, "pwke" is a subsequence and not a substring.
    public int lengthOfLongestSubstring(String s) {
        if (s.length()==0 || s.length()==1) return s.length();

        HashMap<Character, Integer> map = new HashMap<>();
        int max=0;
        for (int i=0, j=0; i<s.length(); ++i){
            if (map.containsKey(s.charAt(i))){
                j = Math.max(j,map.get(s.charAt(i)));
            }
            map.put(s.charAt(i),i);
            max = Math.max(max,i-j);
        }
        return max;
    }


//4. Median of Two Sorted Arrays
//    There are two sorted arrays nums1 and nums2 of size m and n respectively.
//
//    Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).
//
//    Example 1:
//    nums1 = [1, 3]
//    nums2 = [2]
//
//    The median is 2.0
//    Example 2:
//    nums1 = [1, 2]
//    nums2 = [3, 4]
//
//    The median is (2 + 3)/2 = 2.5

    public double findMedianSortedArrays(int[] A, int[] B) {
        int m = A.length, n = B.length;
        int l = (m + n + 1) / 2;
        int r = (m + n + 2) / 2;
        return (getkth(A, 0, B, 0, l) + getkth(A, 0, B, 0, r)) / 2.0;
    }

    public double getkth(int[] A, int aStart, int[] B, int bStart, int k) {
        if (aStart > A.length - 1) return B[bStart + k - 1];
        if (bStart > B.length - 1) return A[aStart + k - 1];
        if (k == 1) return Math.min(A[aStart], B[bStart]);

        int aMid = Integer.MAX_VALUE, bMid = Integer.MAX_VALUE;
        if (aStart + k/2 - 1 < A.length) aMid = A[aStart + k/2 - 1];
        if (bStart + k/2 - 1 < B.length) bMid = B[bStart + k/2 - 1];

        if (aMid < bMid)
            return getkth(A, aStart + k/2, B, bStart,k - k/2);// Check: aRight + bLeft
        else
            return getkth(A, aStart,B, bStart + k/2, k - k/2);// Check: bRight + aLeft
    }


    //5. Longest Palindromic Substring
//    Given a string s, find the longest palindromic substring in s. You may assume that the maximum length of s is 1000.
//
//    Example:
//
//    Input: "babad"
//
//    Output: "bab"
//
//    Note: "aba" is also a valid answer.
//            Example:
//
//    Input: "cbbd"
//
//    Output: "bb"

    public String longestPalindrome(String s) {
        int len = s.length();
        if (s == null) return null;
        String longest = s.substring(0, 1);
        for (int i = 0; i < len-1; i++) {
            String palindrome = extendPalindrome(s, i, i);  //assume odd length, try to extend Palindrome as possible
            if (palindrome.length() > longest.length()) {
                longest = palindrome;
            }
            palindrome = extendPalindrome(s, i, i+1); //assume even length.
            if (palindrome.length() > longest.length()) {
                longest = palindrome;
            }
        }
        return longest;

    }

    private String extendPalindrome(String s, int left, int right) {
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            left--;
            right++;
        }
        return s.substring(left + 1, right);
    }


//6. ZigZag Conversion
//    The string "PAYPALISHIRING" is written in a zigzag pattern on
// a given number of rows like this: (you may want to display this pattern
// in a fixed font for better legibility)
//
//    P   A   H   N
//    A P L S I I G
//    Y   I   R
//    And then read line by line: "PAHNAPLSIIGYIR"
//    Write the code that will take a string and make this conversion given a number of rows:
//
//    string convert(string text, int nRows);
//    convert("PAYPALISHIRING", 3) should return "PAHNAPLSIIGYIR".

    public String convert(String s, int nRows) {
        if (s == null || s.length()==0) {
            return s;
        }

        int length = s.length();
        if (length <= nRows || nRows == 1) {
            return s;
        }

        StringBuilder[] sb = new StringBuilder[nRows];

        for (int i = 0; i < sb.length; i++) sb[i] = new StringBuilder();

        int i = 0;
        while (i < length) {
            for (int idx = 0; idx < nRows && i < length; idx++) // vertically down
                sb[idx].append(s.charAt(i++));
            for (int idx = nRows-2; idx >= 1 && i < length; idx--) // obliquely up
                sb[idx].append(s.charAt(i++));
        }
        for (int idx = 1; idx < sb.length; idx++)
            sb[0].append(sb[idx]);
        return sb[0].toString();
    }

//7. Reverse Integer
//    Reverse digits of an integer.
//
//            Example1: x = 123, return 321
//    Example2: x = -123, return -321

    public int reverse(int x) {
        long res= 0;
        while( x != 0){
            res= res*10 + x % 10;
            x= x/10;
            if( res > Integer.MAX_VALUE || res < Integer.MIN_VALUE)
                return 0;
        }
        return (int) res;
    }


//8. String to Integer (atoi)

//    Implement atoi to convert a string to an integer.
//
//            Hint: Carefully consider all possible input cases.
// If you want a challenge, please do not see below and ask yourself
// what are the possible input cases.
//
//    Notes: It is intended for this problem to be specified vaguely
// (ie, no given input specs). You are responsible to gather all the input requirements up front.

    public int myAtoi(String str) {
        if(str==null || str.trim().length()==0) return 0;
        str=str.trim();
        int sign = 1;
        //Handle signs
        if(str.charAt(0) == '+' || str.charAt(0) == '-'){
            sign = str.charAt(0) == '+' ? 1 : -1;
            str= str.substring(1);
        }
        //Convert number and avoid overflow
        int index = 0;
        int total = 0;
        while(index < str.length()){
            int digit = str.charAt(index) - '0';
            if(digit < 0 || digit > 9) break;

            //check if total will be overflow after 10 times and add digit
            if(total > Integer.MAX_VALUE/10 || total == Integer.MAX_VALUE/10  && digit > Integer.MAX_VALUE %10 )
                return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;

            total = 10 * total + digit;
            index ++;
        }
        return total * sign;
    }

//9. Palindrome Number
    //    Palindrome Number
//    Determine whether an integer is a palindrome. Do this without extra space.
    public boolean isPalindrome(int x) {
        if (x<0 || (x!=0 && x%10==0)) return false;
        int rev = 0;
        while (x>rev){
            rev = rev*10 + x%10;
            x = x/10;
        }
        return (x==rev || x==rev/10);
    }


//    Longest Common Substring
//    In computer science, the longest common substring problem is
// to find the longest string that is a substring of two or more strings.

    public int getLongestCommonSubstringInt(String a, String b){

        if(a == null || b == null || a.length()==0 || b.length() == 0) return 0;

        int m = a.length();
        int n = b.length();

        int max = 0;
        int[][] dp = new int[m][n];

        for(int i=0; i<m; i++){
            for(int j=0; j<n; j++){
                if(a.charAt(i) == b.charAt(j)){
                    if(i==0 || j==0){
                        dp[i][j]=1;
                    }else{
                        dp[i][j] = dp[i-1][j-1]+1;
                    }

                    if(max < dp[i][j])
                        max = dp[i][j];
                }
            }
        }

        return max;
    }

    public String getLongestCommonSubstring(String a, String b){

        if (a == null || b == null) return null;
        if (a.length() == 0 || b.length() == 0) return "";

        int m = a.length();
        int n = b.length();

        int max = 0, endPoint=0;
        int[][] dp = new int[m][n];

        for(int i=0; i<m; i++){
            for(int j=0; j<n; j++){
                if(a.charAt(i) == b.charAt(j)){
                    if(i==0 || j==0){
                        dp[i][j]=1;
                    }else{
                        dp[i][j] = dp[i-1][j-1]+1;
                    }

                    if(max < dp[i][j]) {
                        max = dp[i][j];
                        endPoint=i;
                    }
                }
            }
        }

        return a.substring(endPoint-max+1,endPoint);
    }

//10. Regular Expression Matching

//    Implement regular expression matching with support for '.' and '*'.
//
//            '.' Matches any single character.
//            '*' Matches zero or more of the preceding element.
//
//    The matching should cover the entire input string (not partial).
//
//    Example 4:
//
//    Input:
//    s = "aab"
//    p = "c*a*b"
//    Output: true
//    Explanation: c can be repeated 0 times, a can be repeated 1 time. Therefore, it matches "aab".
//
//    Example 5:
//
//    Input:
//    s = "mississippi"
//    p = "mis*is*p*."
//    Output: false

//    The function prototype should be:
//    bool isMatch(const char *s, const char *p)
//
//    Some examples:
//    isMatch("aa","a") → false
//    isMatch("aa","aa") → true
//    isMatch("aaa","aa") → false
//    isMatch("aa", "a*") → true
//    isMatch("aa", ".*") → true
//    isMatch("ab", ".*") → true
//    isMatch("aab", "c*a*b") → true


    public boolean isMatch2(String s, String p) {// pattern covers input. It means patten contains input string.
        if (p.length() == 0) {
            return s.length() == 0;
        }
        if (p.length() > 1 && p.charAt(1) == '*') {  // second char is '*'
            if (isMatch2(s, p.substring(2))) {
                return true;
            }
            if(s.length() > 0 && (p.charAt(0) == '.' || s.charAt(0) == p.charAt(0))) {
                return isMatch2(s.substring(1), p);
            }
            return false;
        } else {                                     // second char is not '*'
            if(s.length() > 0 && (p.charAt(0) == '.' || s.charAt(0) == p.charAt(0))) {
                return isMatch2(s.substring(1), p.substring(1));
            }
            return false;
        }
    }

//    /*
//
//1, If p.charAt(j) == s.charAt(i) :  dp[i][j] = dp[i-1][j-1];
//2, If p.charAt(j) == '.' : dp[i][j] = dp[i-1][j-1];
//3, If p.charAt(j) == '*':
//   here are two sub conditions:
//       1   if p.charAt(j-1) != s.charAt(i) : dp[i][j] = dp[i][j-2]  //in this case, a* only counts as empty
//       2   if p.charAt(j-1) == s.charAt(i) or p.charAt(j-1) == '.':
//                      dp[i][j] = dp[i-1][j]    //in this case, a* counts as multiple a
//                   or dp[i][j] = dp[i][j-1]   // in this case, a* counts as single a
//                   or dp[i][j] = dp[i][j-2]   // in this case, a* counts as empty     */
//
//    public boolean isMatch(String s, String p) {
//        if(s == null || p == null || p.length()>0 && p.startsWith("*")) {
//            return false;
//        }
//
//        boolean[][] state = new boolean[s.length() + 1][p.length() + 1];
//        state[0][0] = true;
//        // no need to initialize state[i][0] as false
//        // initialize state[0][j]
//        for (int i = 0; i < p.length(); i++) {
//            if (p.charAt(i) == '*') {
//                state[0][i+1] = state[0][i-1];
//            }
//        }
//        for (int i = 1; i < state.length; i++) {
//            for (int j = 1; j < state[0].length; j++) {
//                if (s.charAt(i - 1) == p.charAt(j - 1) || p.charAt(j - 1) == '.') {
//                    state[i][j] = state[i - 1][j - 1];
//                }
//                if (p.charAt(j - 1) == '*') {
//                    if (s.charAt(i - 1) != p.charAt(j - 2) && p.charAt(j - 2) != '.') {
//                        state[i][j] = state[i][j - 2];
//                    } else {
//                        state[i][j] = state[i - 1][j] || state[i][j - 1] || state[i][j - 2];
//                    }
//                }
//            }
//        }
//        return state[s.length()][p.length()];
//    }

//11. Container With Most Water

//    Container With Most Water
//    Given n non-negative integers a1, a2, ..., an,
// where each represents a point at coordinate (i, ai).
// n vertical lines are drawn such that
// the two endpoints of line i is at (i, ai) and (i, 0).
// Find two lines, which together with x-axis forms a container,
// such that the container contains the most water.
//
//            Note: You may not slant the container and n is at least 2.

    public int maxArea(int[] height) {
        int left = 0, right = height.length - 1;
        int maxArea = 0;

        while (left < right) {
            maxArea = Math.max(maxArea, Math.min(height[left], height[right])
                    * (right - left));
            if (height[left] < height[right])
                left++;
            else
                right--;
        }

        return maxArea;
    }

//12. Integer to Roman

    //    Given an integer, convert it to a roman numeral.
//    Input is guaranteed to be within the range from 1 to 3999.
    public static String intToRoman(int num) {
        String M[] = {"", "M", "MM", "MMM"};
        String C[] = {"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"};
        String X[] = {"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"};
        String I[] = {"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"};
        return M[num/1000] + C[(num%1000)/100] + X[(num%100)/10] + I[num%10];
    }


//13. Roman to Integer

//    unordered_map<char, int> T =
//          { { 'I' , 1 },
//            { 'V' , 5 },
//            { 'X' , 10 },
//            { 'L' , 50 },
//            { 'C' , 100 },
//            { 'D' , 500 },
//            { 'M' , 1000 } };

    public int romanToInt(String s) {

        Map<Character, Integer> map= new HashMap<Character, Integer>();
        map.put('I',1);
        map.put('V',5);
        map.put('X',10);
        map.put('L',50);
        map.put('C',100);
        map.put('D',500);
        map.put('M',1000);

        int sum = map.get(s.charAt(s.length()));
        for (int i = s.length() - 2; i >= 0; --i)
        {
            if (map.get(s.charAt(i)) < map.get(s.charAt(i+1)))
            {
                sum -= map.get(s.charAt(i));
            }
            else
            {
                sum += map.get(s.charAt(i));
            }
        }

        return sum;
    }


    //14. Longest Common Prefix

//    Write a function to find the longest common prefix string among an array of strings.

    public String longestCommonPrefix(String[] strs) {
        if(strs == null || strs.length == 0)    return "";
        String pre = strs[0];
        int i = 1;
        while(i < strs.length){
            while(strs[i].indexOf(pre) != 0)
                pre = pre.substring(0,pre.length()-1);
            i++;
        }
        return pre;
    }

//15. 3Sum
//    3Sum
//    Given an array S of n integers, are there elements a, b, c in S
// such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.
//
//            Note: The solution set must not contain duplicate triplets.
//
//    For example, given array S = [-1, 0, 1, 2, -1, -4],
//
//    A solution set is:
//            [
//            [-1, 0, 1],
//            [-1, -1, 2]
//            ]

    public List<List<Integer>> threeSum(int[] num) {
        if(num == null) return null;
        Arrays.sort(num);
        List<List<Integer>> res = new LinkedList<List<Integer>>();
        for (int i = 0; i < num.length-2; i++) {
            if (i == 0 || (i > 0 && num[i] != num[i-1])) {
                int lo = i+1, hi = num.length-1, sum = 0 - num[i];
                while (lo < hi) {
                    if (num[lo] + num[hi] == sum) {
                        res.add(Arrays.asList(num[i], num[lo], num[hi]));
                        while (lo < hi && num[lo] == num[lo+1]) lo++;
                        while (lo < hi && num[hi] == num[hi-1]) hi--;
                        lo++; hi--;
                    } else if (num[lo] + num[hi] < sum) lo++;
                    else hi--;
                }
            }
        }
        return res;
    }

//16. 3Sum Closest

//    3Sum Closest
//Given an array S of n integers, find three integers in S such
// that the sum is closest to a given number, target.
// Return the sum of the three integers. You may assume that
// each input would have exactly one solution.
//
//    For example, given array S = {-1 2 1 -4}, and target = 1.
//
//    The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).


    public int threeSumClosest(int[] nums, int target) {
        int sum = 0;
        if (nums.length <= 3) {
            for (int i: nums) {
                sum += i;
            }
            return sum;
        }

        Arrays.sort(nums);
        sum = nums[0] + nums[1] + nums[nums.length - 1];
        int closestSum = sum;

        for(int i = 0; i < nums.length - 2; i++){
            if(i==0 || nums[i]!=nums[i-1]){
                int left = i + 1, right = nums.length - 1;
                while(left < right){
                    sum = nums[left] + nums[right] + nums[i];

                    if(sum==target){
                        return sum;
                    }

                    if(Math.abs(sum-target) <= Math.abs(closestSum-target)){
                        closestSum = sum;
                    }

                    if(sum < target){
                        left++;
                        //move closer to target sum.
                        while(left<right && nums[left] == nums[left+1]){
                            left++;
                        }

                    }else if(sum > target){
                        right--;
                        //move closer to target sum.
                        while(left<right && nums[right] == nums[right-1]){
                            right--;
                        }
                    }
                }
            }

        }
        return closestSum;
    }

//17. Letter Combinations of a Phone Number

//    Letter Combinations of a Phone Number
//    Given a digit string, return all possible letter combinations that the number could represent.
//
//    A mapping of digit to letters (just like on the telephone buttons) is given below.
//    Input:Digit string "23"
//    Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].

    public List<String> letterCombinations(String digits) {
        LinkedList<String> result = new LinkedList<String>();
        if (digits == null || digits.equals("")) {
            return result;
        }
        String[] mapping = new String[] {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        result.add("");
        int length=0;  //use queue.size
        for(int i =0; i<digits.length();i++){
            int x = digits.charAt(i)-'0';
            if(x>1 && x<10) {
                while (result.peek().length() == length) {
                    String t = result.poll();
                    for (char s : mapping[x].toCharArray())
                        result.offer(t + s);
                }
                length++;
            }
        }
        return result;
    }

//18. 4Sum
    //    4SUM
//    Given an array S of n integers, are there elements a, b, c, and d in S
// such that a + b + c + d = target? Find all unique quadruplets in the array
// which gives the sum of target.
//
//            Note: The solution set must not contain duplicate quadruplets.
//
//    For example, given array S = [1, 0, -1, 0, -2, 2], and target = 0.
//
//    A solution set is:
//            [
//            [-1,  0, 0, 1],
//            [-2, -1, 1, 2],
//            [-2,  0, 0, 2]
//            ]
    public List<List<Integer>> fourSum(int[] num, int target) {
        ArrayList<List<Integer>> ans = new ArrayList<>();
        if(num.length<4)return ans;
        Arrays.sort(num);
        for(int i=0; i<num.length-3; i++){
            if(i>0 && num[i]==num[i-1])continue;
            for(int j=i+1; j<num.length-2; j++){
                if(j>i+1 && num[j]==num[j-1])continue;
                int low=j+1, high=num.length-1;
                while(low<high){
                    int sum=num[i]+num[j]+num[low]+num[high];
                    if(sum==target){
                        ans.add(Arrays.asList(num[i], num[j], num[low], num[high]));
                        while(low<high&&num[low]==num[low+1])low++;
                        while(low<high&&num[high]==num[high-1])high--;
                        low++;
                        high--;
                    }
                    else if(sum<target)low++;
                    else high--;
                }
            }
        }
        return ans;
    }



//    19. Remove Nth Node From End of List
//Given a linked list, remove the nth node from the end of list and return its head.
//
//    For example,
//
//    Given linked list: 1->2->3->4->5, and n = 2.
//
//    After removing the second node from the end, the linked list becomes 1->2->3->5.

//    Note:
//    Given n will always be valid.
//    Try to do this in one pass.

    public ListNode removeNthFromEnd(ListNode head, int n) {

        if(head.next == null) return null;

        ListNode start = new ListNode(0);
        ListNode slow = start, fast = start;
        slow.next = head;

        //Move fast in front so that the gap between slow and fast becomes n
        for(int i=1; i<=n+1; i++)   {
            fast = fast.next;
        }
        //Move fast to the end, maintaining the gap
        while(fast != null) {
            slow = slow.next;
            fast = fast.next;
        }
        //Skip the desired node
        slow.next = slow.next.next;
        return start.next;
    }

    //20. Valid Parentheses

//    Valid Parentheses
//    Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
//
//    The brackets must close in the correct order, "()" and "()[]{}" are all valid but "(]" and "([)]" are not.

    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<Character>();
        for (char c : s.toCharArray()) {
            if (c == '(')
                stack.push(')');
            else if (c == '{')
                stack.push('}');
            else if (c == '[')
                stack.push(']');
            else if (stack.isEmpty() || stack.pop() != c)
                return false;
        }
        return stack.isEmpty();
    }

//21. Merge Two Sorted Lists

//    Merge Two Sorted Lists
//Merge two sorted linked lists and return it as a new list.
// The new list should be made by splicing together the nodes of the first two lists.

    public ListNode mergeTwoLists(ListNode l1, ListNode l2){
        if(l1 == null) return l2;
        if(l2 == null) return l1;
        if(l1.val < l2.val){
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        } else{
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }
    }

//22. Generate Parentheses

    //    Generate Parentheses
//Given n pairs of parentheses, write a function to generate all combinations of
// well-formed parentheses.
//
//    For example, given n = 3, a solution set is:
//
//            [
//            "((()))",
//            "(()())",
//            "(())()",
//            "()(())",
//            "()()()"
//            ]
    public List<String> generateParenthesis(int n) {
        List<String> list = new ArrayList<String>();
        backtrack(list, "", 0, 0, n);
        return list;
    }

    public void backtrack(List<String> list, String str, int open, int close, int max){

        if(str.length() == max*2){
            list.add(str);
            return;
        }

        if(open < max)
            backtrack(list, str+"(", open+1, close, max);
        if(close < open)
            backtrack(list, str+")", open, close+1, max);
    }


//    23 Merge k Sorted Lists
//  Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity.
    //Merge k Sorted Lists
    //Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity.
    //Input:
    //[
    //  1->4->5,
    //  1->3->4,
    //  2->6
    //]
    //Output: 1->1->2->3->4->4->5->6
    //
    //
    //class ListNode {
    //    int val;
    //    ListNode next;
    //    ListNode(int x) { val = x; }
    //}

    public ListNode mergeKLists(List<ListNode> lists) {
        if (lists==null||lists.size()==0) return null;

        PriorityQueue<ListNode> queue= new PriorityQueue<>(lists.size(),(a, b)-> a.val- b.val);

        ListNode dummy = new ListNode(0);
        ListNode tail=dummy;

        for (ListNode node:lists)
            if (node!=null)
                queue.add(node);

        while (!queue.isEmpty()){
            tail.next=queue.poll();
            tail=tail.next;

            if (tail.next!=null)
                queue.add(tail.next);
        }
        return dummy.next;
    }


//    24 Swap Nodes in Pairs
//    Given a linked list, swap every two adjacent nodes and return its head.
//
//    For example,
//    Given 1->2->3->4, you should return the list as 2->1->4->3.
//
//    Your algorithm should use only constant space. You may not modify the values in the list, only nodes itself can be changed.


    public ListNode swapPairs(ListNode head) {
        if ((head == null)||(head.next == null))
            return head;
        ListNode n = head.next;
        head.next = swapPairs(head.next.next);
        n.next = head;
        return n;
    }


//    25 Reverse Nodes in k-Group
//    Given a linked list, reverse the nodes of
// a linked list k at a time and return its modified list.
//
//    k is a positive integer and is less than or equal to
// the length of the linked list. If the number of nodes is not
// a multiple of k then left-out nodes in the end should remain as it is.
//
//    You may not alter the values in the nodes, only nodes itself may be changed.
//
//    Only constant memory is allowed.
//
//            For example,
//    Given this linked list: 1->2->3->4->5
//
//    For k = 2, you should return: 2->1->4->3->5
//
//    For k = 3, you should return: 3->2->1->4->5

    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode curr = head;
        int count = 0;
        while (curr != null && count != k) { // find the k+1 node
            curr = curr.next;
            count++;
        }
        if (count == k) { // if k+1 node is found
            curr = reverseKGroup(curr, k); // reverse list with k+1 node as head
            // head - head-pointer to direct part,
            // curr - head-pointer to reversed part;
            while (count-- > 0) { // reverse current k-group:
                ListNode tmp = head.next; // tmp - next head in direct part
                head.next = curr; // preappending "direct" head to the reversed list
                curr = head; // move head of reversed part to a new node
                head = tmp; // move "direct" head to the next node in direct part
            }
            head = curr;
        }
        return head;
    }

    //80

//    26 Remove Duplicates from Sorted Array
//    Given a sorted array, remove the duplicates in place
// such that each element appear only once and return the new length.
//
//    Do not allocate extra space for another array,
// you must do this in place with constant memory.
//
//            For example,
//    Given input array nums = [1,1,2],
//
//    Your function should return length = 2,
// with the first two elements of nums being 1 and 2 respectively.
// It doesn't matter what you leave beyond the new length.

    public int removeDuplicates2(int[] nums) {
        int i = 0;
        for (int n : nums)
            if (i == 0 || n > nums[i-1])
                nums[i++] = n;
        return i;
    }



//    27 Remove Element
//    Given an array and a value, remove all instances of
// that value in place and return the new length.
//
//    Do not allocate extra space for another array,
// you must do this in place with constant memory.
//
//    The order of elements can be changed. It doesn't matter
// what you leave beyond the new length.
//
//    Example:
//    Given input array nums = [3,2,2,3], val = 3
//
//    Your function should return length = 2, with the first two elements of nums being 2.


    public int removeElement(int[] A, int elem) {
        int m = 0;
        for(int n: A){
            if(n != elem){
                A[m++] = n;
            }
        }
        return m;
    }


//    28 Implement strStr()
//    Implement strStr().
//
//    Returns the index of the first occurrence of needle in haystack,
// or -1 if needle is not part of haystack.

    public int strStr(String haystack, String needle) {
        for (int i = 0; ; i++) {
            for (int j = 0; ; j++) {
                if (j == needle.length()) return i;
                if (i + j == haystack.length()) return -1;
                if (needle.charAt(j) != haystack.charAt(i + j)) break;
            }
        }
    }


//    29 Divide Two Integers
//    Divide two integers without using multiplication, division and mod operator.
//
//    If it is overflow, return MAX_INT.


    public int divide(int dividend, int divisor) {
        if (divisor == 0) {
            return dividend >= 0? Integer.MAX_VALUE : Integer.MIN_VALUE;
        }

        if (dividend == 0) {
            return 0;
        }

        if (dividend == Integer.MIN_VALUE && divisor == -1) {
            return Integer.MAX_VALUE;
        }

        boolean isNegative = (dividend < 0 && divisor > 0) ||
                (dividend > 0 && divisor < 0);

        long a = Math.abs((long)dividend);
        long b = Math.abs((long)divisor);
        int result = 0;
        while(a >= b){
            int shift = 0;
            //b << shift   ==  b*pow(2,shift)
            //get the pow of B, substract it from A,
            // so it is pow times of B
            while(a >= (b << shift)){
                shift++;
            }
            a -= b << (shift - 1);
            result += 1 << (shift - 1);
        }
        return isNegative? -result: result;
    }

//    30. Substring with Concatenation of All Words
//    You are given a string, s, and a list of words, words, that are all of the same length.
// Find all starting indices of substring(s) in s
// that is a concatenation of each word in words exactly once and
// without any intervening characters.
//    Example 1:
//    Input:
//    s = "barfoothefoobarman",
//    words = ["foo","bar"]
//    Output: [0,9]
//    Explanation: Substrings starting at index 0 and 9 are "barfoor" and "foobar" respectively.
//    The output order does not matter, returning [9,0] is fine too.
//            Example 2:
//    Input:
//    s = "wordgoodstudentgoodword",
//    words = ["word","student"]
//    Output: []

    public List<Integer> findSubstring(String s, String[] words) {
        Map<String, Integer> counts = new HashMap<>();
        for ( String word : words) {
            counts.put(word, counts.getOrDefault(word, 0) + 1);
        }
         List<Integer> indexes = new ArrayList<>();
         int n = s.length(), num = words.length, len = words[0].length();

        for (int i = 0; i < n - num * len + 1; i++) {

            Map<String, Integer> seen = new HashMap<>();
            int j = 0;
            while (j < num) {
                 String word = s.substring(i + j * len, i + (j + 1) * len);
                if (counts.containsKey(word)) {
                    seen.put(word, seen.getOrDefault(word, 0) + 1);
                    if (seen.get(word) > counts.getOrDefault(word, 0)) {
                        break;
                    }
                } else {
                    break;
                }
                j++;
            }
            if (j == num) {
                indexes.add(i);
            }
        }
        return indexes;
    }



//    31 Implement next permutation, which rearranges numbers into
// the lexicographically next greater permutation of
// numbers. If such arrangement is not possible, it must rearrange it as
// the lowest possible order (ie, sorted in ascending order).
// The replacement must be in-place, do not allocate extra memory.
// Here are some examples. Inputs are in the left-hand column and
// its corresponding outputs are in the right-hand column.
// 1,2,3 → 1,3,2 3,2,1 → 1,2,3 1,1,5 → 1,5,1

    /*

     i+1
     /\
    /  \
   i    \
         \
          \
           length()-1
     */

    public void nextPermutation(int[] num) {
        if (num == null) {
            return;
        }

        int len = num.length;
        for (int i = len - 2; i >= 0; i--) {
            // find the adjacent two numbers
            if (num[i + 1] > num[i]) {
                int j;
                //find the minimum larger number for i
                for (j = len - 1; j > i - 1; j--) {
                    if (num[j] > num[i]) {
                        break;
                    }
                }
                swap(num, i, j);
                reverse(num, i + 1, len-1);
                return;
            }
        }
        reverse(num, 0, len-1);
    }

    void swap(int[] num, int i, int j) {
        int tmp = num[i];
        num[i] = num[j];
        num[j] = tmp;
    }

    void reverse(int[] num, int beg, int end) {
        for (int i = beg, j = end; i < j; i ++, j --) {
            swap(num, i, j);
        }
    }

    /**
    32. Longest Valid Parentheses
Given a string containing just the characters '(' and ')',
find the length of the longest valid (well-formed) parentheses substring.
Example 1:
Input: "(()"
Output: 2
Explanation: The longest valid parentheses substring is "()"
Example 2:
Input: ")()())"
Output: 4
Explanation: The longest valid parentheses substring is "()()"
     */

    public int longestValidParentheses(String s) {
        Stack<Integer> stack = new Stack<Integer>();
        int max=0;
        int left = -1;
        for(int j=0;j<s.length();j++){
            if(s.charAt(j)=='(') stack.push(j);
            else {
                if (stack.isEmpty()) left=j;
                else{
                    stack.pop();
                    if(stack.isEmpty()) max=Math.max(max,j-left);
                    else max=Math.max(max,j-stack.peek());
                }
            }
        }
        return max;
    }


    //    81. Search in Rotated Sorted Array II

//   33  Search in Rotated Sorted Array



//    Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.
//
//            (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
//
//    You are given a target value to search. If found in the array return its index, otherwise return -1.
//
//    You may assume no duplicate exists in the array.
//
//    Subscribe to see which companies asked this question.


    public int search2(int[] nums, int target) {
        int start = 0;
        int end = nums.length - 1;
        while (start <= end){
            int mid = (start + end) / 2;
            if (nums[mid] == target)
                return mid;

            if (nums[start] <= nums[mid]){
                if (target < nums[mid] && target >= nums[start])
                    end = mid - 1;
                else
                    start = mid + 1;
            }

            if (nums[mid] <= nums[end]){
                if (target > nums[mid] && target <= nums[end])
                    start = mid + 1;
                else
                    end = mid - 1;
            }
        }
        return -1;
    }


//   34 Search for a Range

//    Given an array of integers sorted in ascending order,
// find the starting and ending position of a given target value.
//
//    Your algorithm's runtime complexity must be in the order of O(log n).
//
//    If the target is not found in the array, return [-1, -1].
//
//    For example,
//    Given [5, 7, 7, 8, 8, 10] and target value 8,
//            return [3, 4].

    public int[] searchRange(int[] A, int target) {
        if (A.length == 0) {
            return new int[]{-1, -1};
        }

        int[] bound = new int[2];

        // search for left bound
        int left = 0;
        int right = A.length - 1;

        while(left <= right){
            int mid = left + (right - left) / 2;
            if(A[mid] == target){
                bound[0] = mid;
                right = mid - 1;
            }
            else if(A[mid] > target) right = mid -1;
            else left = mid + 1;
        }

        // search for right bound
        left = 0;
        right = A.length - 1;

        while(left <= right){
            int mid = left + (right - left) / 2;
            if(A[mid] == target){
                bound[1] = mid;
                left = mid + 1;
            }
            else if(A[mid] > target) right = mid -1;
            else left = mid + 1;
        }

        return bound;
    }

//   35 Search Insert Position
//
//    Given a sorted array and a target value, return the index if the target is found.
// If not, return the index where it would be if it were inserted in order.
//
//    You may assume no duplicates in the array.
//
//    Here are few examples.
//            [1,3,5,6], 5 → 2
//            [1,3,5,6], 2 → 1
//            [1,3,5,6], 7 → 4
//            [1,3,5,6], 0 → 0

    public int searchInsert(int[] a, int key) {
        if (a == null || a.length == 0) {
            return 0;
        }
        int low = 0;
        int high = a.length-1;

        while (low <= high) {
            int mid = (low + high) >>> 1;
            long midVal = a[mid];

            if (midVal < key)
                low = mid + 1;
            else if (midVal > key)
                high = mid - 1;
            else
                return mid; // key found
        }
        return low + 1;  // key not found.
    }

//    36 Valid Sudoku
//    Determine if a Sudoku is valid, according to: Sudoku Puzzles - The Rules.
//Sudoku ( digit-single) (/suːˈdoʊkuː/, /-ˈdɒk-/, /sə-/, originally called Number Place)[1] is a logic-based,[2][3] combinatorial[4] number-placement puzzle. The objective is to fill a 9×9 grid with digits so that each column, each row, and each of the nine 3×3 subgrids that compose the grid (also called "boxes", "blocks", or "regions") contain all of the digits from 1 to 9. The puzzle setter provides a partially completed grid, which for a well-posed puzzle has a single solution.
//
//Completed games are always a type of Latin square with an additional constraint on the contents of individual regions. For example, the same single integer may not appear twice in the same row, column, or any of the nine 3×3 subregions of the 9×9 playing board.
//    The Sudoku board could be partially filled,
// where empty cells are filled with the character '.'.

    public boolean isValidSudoku(char[][] board) {
        boolean[] visited = new boolean[9];

        // row
        for(int i = 0; i<9; i++){
            Arrays.fill(visited, false);
            for(int j = 0; j<9; j++){
                if(!process(visited, board[i][j]))
                    return false;
            }
        }

        //col
        for(int i = 0; i<9; i++){
            Arrays.fill(visited, false);
            for(int j = 0; j<9; j++){
                if(!process(visited, board[j][i]))
                    return false;
            }
        }

        // sub matrix
        for(int i = 0; i<9; i+= 3){
            for(int j = 0; j<9; j+= 3){
                Arrays.fill(visited, false);
                for(int k = 0; k<9; k++){
                    if(!process(visited, board[i + k/3][ j + k%3]))
                        return false;
                }
            }
        }
        return true;

    }

    private boolean process(boolean[] visited, char digit){
        if(digit == '.'){
            return true;
        }

        int num = digit - '0';
        if ( num < 1 || num > 9 || visited[num-1]){
            return false;
        }

        visited[num-1] = true;
        return true;
    }

    /**

    37. Sudoku Solver
Write a program to solve a Sudoku puzzle by filling the empty cells.
A sudoku solution must satisfy all of the following rules:
Each of the digits 1-9 must occur exactly once in each row.
Each of the digits 1-9 must occur exactly once in each column.
Each of the the digits 1-9 must occur exactly once in each of the 9 3x3 sub-boxes of the grid.
Empty cells are indicated by the character '.'.


Try 1 through 9 for each cell. The time complexity should be 9 ^ m
(m represents the number of blanks to be filled in),
since each blank can have 9 choices. Details see comments inside code.

     */

    public void solveSudoku(char[][] board) {
        if(board == null || board.length == 0)
            return;
        solve(board);
    }

    public boolean solve(char[][] board){
        for(int i = 0; i < board.length; i++){
            for(int j = 0; j < board[0].length; j++){
                if(board[i][j] == '.'){
                    for(char c = '1'; c <= '9'; c++){//trial. Try 1 through 9
                        if(isValid(board, i, j, c)){
                            board[i][j] = c; //Put c for this cell

                            if(solve(board))
                                return true; //If it's the solution return true
                            else
                                board[i][j] = '.'; //Otherwise go back
                        }
                    }

                    return false;
                }
            }
        }
        return true;
    }

    private boolean isValid(char[][] board, int row, int col, char c){
        for(int i = 0; i < 9; i++) {
            if(board[i][col] != '.' && board[i][col] == c) return false; //check row
            if(board[row][i] != '.' && board[row][i] == c) return false; //check column
            if(board[3 * (row / 3) + i / 3][ 3 * (col / 3) + i % 3] != '.' &&
                    board[3 * (row / 3) + i / 3][3 * (col / 3) + i % 3] == c) return false; //check 3*3 block
        }
        return true;
    }


//    38 Count and Say
//    The count-and-say sequence is the sequence of integers beginning as follows:
//            1, 11, 21, 1211, 111221, ...
//
//            1 is read off as "one 1" or 11.
//            11 is read off as "two 1s" or 21.
//            21 is read off as "one 2, then one 1" or 1211.
//    Given an integer n, generate the nth sequence.
//
//            Note: The sequence of integers will be represented as a string.

    public String countAndSay(int n) {
        String oldString = "1";
        while (--n > 0) {
            StringBuilder sb = new StringBuilder();
            char [] oldChars = oldString.toCharArray();

            for (int i = 0; i < oldChars.length; i++) {
                int count = 1;
                while ((i+1) < oldChars.length && oldChars[i] == oldChars[i+1]) {
                    count++;
                    i++;
                }
                sb.append(String.valueOf(count) + String.valueOf(oldChars[i]));
            }
            oldString = sb.toString();
        }

        return oldString;
    }


//  39  Combination Sum

//    Given a set of candidate numbers (C) (without duplicates) and a target number (T),
// find all unique combinations in C where the candidate numbers sums to T.
//
//    The same repeated number may be chosen from C unlimited number of times.
//
//            Note:
//    All numbers (including target) will be positive integers.
//    The solution set must not contain duplicate combinations.
//    For example, given candidate set [2, 3, 6, 7] and target 7,
//    A solution set is:
//            [
//            [7],
//            [2, 2, 3]
//            ]

    public  List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> result = new ArrayList<>();
        if (candidates == null) {
            return result;
        }
        List<Integer> combination = new ArrayList<>();
        Arrays.sort(candidates);
        helper(candidates, 0, target, combination, result);

        return result;
    }

    void helper(int[] candidates,
                int index,
                int target,
                List<Integer> combination,
                List<List<Integer>> result) {
        if (target == 0) {
            result.add(new ArrayList<>(combination));
            return;
        }

        for (int i = index; i < candidates.length; i++) {
            if (candidates[i] > target) {
                break;
            }

            //The solution set must not contain duplicate combinations.
            if (i != index && candidates[i] == candidates[i - 1]) {
                continue;
            }

            combination.add(candidates[i]);
            helper(candidates, i, target - candidates[i], combination, result);
            combination.remove(combination.size() - 1);
        }
    }


//  40  Combination Sum II

//    Given a collection of candidate numbers (C) and a target number (T),
// find all unique combinations in C where the candidate numbers sums to T.
//
//    Each number in C may only be used once in the combination.
//
//    Note:
//    All numbers (including target) will be positive integers.
//    The solution set must not contain duplicate combinations.
//    For example, given candidate set [10, 1, 2, 7, 6, 1, 5] and target 8,
//    A solution set is:
//            [
//            [1, 7],
//            [1, 2, 5],
//            [2, 6],
//            [1, 1, 6]
//            ]

    public List<List<Integer>> combinationSum2(int[] candidates,
                                               int target) {
        List<List<Integer>> results = new ArrayList<>();
        if (candidates == null || candidates.length == 0) {
            return results;
        }

        Arrays.sort(candidates);
        List<Integer> combination = new ArrayList<Integer>();
        helper(candidates, 0, combination, target, results);

        return results;
    }

    private void helper(int[] candidates,
                        int startIndex,
                        List<Integer> combination,
                        int target,
                        List<List<Integer>> results) {
        if (target == 0) {
            results.add(new ArrayList<>(combination));
            return;
        }

        for (int i = startIndex; i < candidates.length; i++) {
            //The solution set must not contain duplicate combinations.
            if (i != startIndex && candidates[i] == candidates[i - 1]) {
                continue;
            }
            if (target < candidates[i]) {
                break;
            }
            combination.add(candidates[i]);
            helper(candidates, i + 1, combination, target - candidates[i], results);
            combination.remove(combination.size() - 1);
        }
    }

//   41 First Missing Positive
//Given an unsorted integer array, find the first missing positive integer.
//
//    For example,
//    Given [1,2,0] return 3,
//    and [3,4,-1,1] return 2.
//
//    Your algorithm should run in O(n) time and uses constant space.

    public int firstMissingPositive(int[] A) {
        if (A == null) {
            return 1;
        }

        for (int i = 0; i < A.length; i++) {
            while (A[i] > 0 && A[i] <= A.length && A[i] != (i+1)) {
                //move current A[i] to correct position
                int tmp = A[A[i]-1];
                A[A[i]-1] = A[i];
                A[i] = tmp;
            }
        }

        for (int i = 0; i < A.length; i ++) {
            if (A[i] != i + 1) {
                return i + 1;
            }
        }

        return A.length + 1;
    }

    //407 Trapping Rain Water II

//42. Trapping Rain Water
//    Given n non-negative integers representing an elevation map
//  where the width of each bar is 1, compute how much water it is able to trap after raining.
//
//            For example,
//    Given [0,1,0,2,1,0,1,3,2,1,2,1], return 6.

    public int trapRainWater(int[] heights) {
        int res = 0;
        if(heights == null || heights.length ==0 || heights.length==1)
            return res;

        int left = 0, right = heights.length - 1;
        int leftheight = heights[left];
        int rightheight = heights[right];

        while(left < right) {
            if(leftheight < rightheight) {
                left ++;
                if(leftheight > heights[left]) {
                    res += (leftheight - heights[left]);
                } else {
                    leftheight = heights[left];
                }
            } else {
                right --;
                if(rightheight > heights[right]) {
                    res += (rightheight - heights[right]);
                } else {
                    rightheight = heights[right];
                }
            }
        }
        return res;
    }

//    43. Multiply Strings
//    Given two non-negative integers num1 and num2 represented as strings,
// return the product of num1 and num2.
//
//    Note:
//
//    The length of both num1 and num2 is < 110.
//    Both num1 and num2 contains only digits 0-9.
//    Both num1 and num2 does not contain any leading zero.
//    You must not use any built-in BigInteger library or convert the inputs to integer directly.

    public String multiply(String num1, String num2) {
//        new BigInteger(num1).multiply(new BigInteger(num2)).toString()
        if (num1 == null || num2 == null) {
            return null;
        }

        int len1 = num1.length(), len2 = num2.length();
        int len3 = len1 + len2;
        int i, j, product, carry;

        int[] num3 = new int[len3];
        for (i = len1 - 1; i >= 0; i--) {
            carry = 0;
            for (j = len2 - 1; j >= 0; j--) {
                product = carry + num3[i + j + 1] +
                        (num1.charAt(i)-'0') * (num2.charAt(j)-'0');
                num3[i + j + 1] = product % 10;
                carry = product / 10;
            }
            num3[i + j + 1] = carry;
        }

        StringBuilder sb = new StringBuilder();
        i = 0;

        while (i < len3 - 1 && num3[i] == 0) {
            i++;
        }

        while (i < len3) {
            sb.append(num3[i++]);
        }

        return sb.toString();
    }

/**

    44. Wildcard Matching
    Given an input string (s) and a pattern (p), implement wildcard pattern matching
    with support for '?' and '*'.
    '?' Matches any single character.
    '*' Matches any sequence of characters (including the empty sequence).
    The matching should cover the entire input string (not partial).
    Note:
    s could be empty and contains only lowercase letters a-z.
    p could be empty and contains only lowercase letters a-z, and characters like ? or *.
    Example 1:
    Input:
    s = "aa"
    p = "a"
    Output: false
    Explanation: "a" does not match the entire string "aa".
    Example 2:
    Input:
    s = "aa"
    p = "*"
    Output: true
    Explanation: '*' matches any sequence.
    Example 3:
    Input:
    s = "cb"
    p = "?a"
    Output: false
    Explanation: '?' matches 'c', but the second letter is 'a', which does not match 'b'.
    Example 4:
    Input:
    s = "adceb"
    p = "*a*b"
    Output: true
    Explanation: The first '*' matches the empty sequence, while the second '*' matches
    the substring "dce".
    Example 5:
    Input:
    s = "acdcb"
    p = "a*c?b"
    Output: false

 */

    public boolean isMatch1(String s, String p) {

        int m = s.length(), n = p.length();
        char[] ws = s.toCharArray();
        char[] wp = p.toCharArray();
        boolean[][] dp = new boolean[m+1][n+1];
        dp[0][0] = true;
        for (int j = 1; j <= n; j++)
            dp[0][j] = dp[0][j-1] && wp[j-1] == '*';
//        for (int i = 1; i <= m; i++)
//            dp[i][0] = false;
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (wp[j-1] == '?' || ws[i-1] == wp[j-1])
                    dp[i][j] = dp[i-1][j-1];
                else if (wp[j-1] == '*')
                    dp[i][j] = dp[i-1][j] || dp[i][j-1];
            }
        }
        return dp[m][n];
    }
//        boolean[][] match = new boolean[s.length() + 1][p.length() + 1];
//        match[s.length()][p.length()] = true;
//        for (int i = p.length() - 1; i >= 0; i--) {
//            if (p.charAt(i) != '*') {
//                break;
//            } else {
//                match[s.length()][i] = true;
//            }
//        }
//
//        for (int i = s.length() - 1; i >= 0; i--) {
//            for (int j = p.length() - 1; j >= 0; j--) {
//                if (s.charAt(i) == p.charAt(j) || p.charAt(j) == '?') {
//                    match[i][j] = match[i + 1][j + 1];
//                } else if (p.charAt(j) == '*') {
//                    //  a adfsafd  ==>  adfsafd  or  a adfsafd
//                    //  * sdfasfd     * sdfasfd        sdfasfd
//                    match[i][j] = match[i + 1][j] || match[i][j + 1];
//                } else {
//                    match[i][j] = false;
//                }
//            }
//        }
//        return match[0][0];
//    }

    //871. Minimum Number of Refueling Stops
    //    55. Jump Game

//    45. Jump Game II
//    Given an array of non-negative integers,
// you are initially positioned at the first index of the array.
//
//    Each element in the array represents your maximum jump length at that position.
//
//    Your goal is to reach the last index in the minimum number of jumps.
//
//    For example:
//    Given array A = [2,3,1,1,4]

    public int jump(int[] A) {
        int jumps = 0, curEnd = 0, curFarthest = 0;
        for (int i = 0; i < A.length - 1; i++) {
            curFarthest = Math.max(curFarthest, i + A[i]);
            if (i == curEnd) {
                jumps++;
                curEnd = curFarthest;
            }
        }
        return jumps;
    }
//
//    The minimum number of jumps to reach the last index is 2.
// (Jump 1 step from index 0 to 1, then 3 steps to the last index.)

//    public int jump(int[] A) {
//        if (A == null || A.length == 0) {
//            return -1;
//        }
//        int start = 0, end = 0, jumps = 0;
//        while (end < A.length - 1) {
//            jumps++;
//            int farthest = end;
//            for (int i = start; i <= end; i++) {
//                if (A[i] + i > farthest) {
//                    farthest = A[i] + i;
//                }
//            }
//            start = end + 1;
//            end = farthest;
//        }
//        return jumps;
//    }




//    46. Permutations
//    Given a collection of distinct numbers, return all possible permutations.
//
//            For example,
//    [1,2,3] have the following permutations:
//            [
//            [1,2,3],
//            [1,3,2],
//            [2,1,3],
//            [2,3,1],
//            [3,1,2],
//            [3,2,1]
//            ]

    public List<List<Integer>> permute(int[] nums) {
        ArrayList<List<Integer>> rst = new ArrayList<List<Integer>>();
        if (nums == null) {
            return rst;
        }

        if (nums.length == 0) {
            rst.add(new ArrayList<Integer>());
            return rst;
        }

        ArrayList<Integer> list = new ArrayList<Integer>();
        helper(rst, list, nums);
        return rst;
    }

    public void helper(ArrayList<List<Integer>> rst, ArrayList<Integer> list, int[] nums){
        if(list.size() == nums.length) {
            rst.add(new ArrayList<Integer>(list));
            return;
        }

        for(int i = 0; i < nums.length; i++){
            if(list.contains(nums[i])){
                continue;
            }
            list.add(nums[i]);
            helper(rst, list, nums);
            list.remove(list.size() - 1);
        }

    }

    //    47. Permutations II
    //
    //    Given a collection of numbers that might contain duplicates,
    // return all possible unique permutations.
    //
    //    For example,
    //    [1,1,2] have the following unique permutations:
    //            [
    //            [1,1,2],
    //            [1,2,1],
    //            [2,1,1]
    //            ]


    public List<List<Integer>> permuteUnique(int[] nums) {

        ArrayList<List<Integer>> results = new ArrayList<List<Integer>>();

        if (nums == null) {
            return results;
        }

        if(nums.length == 0) {
            results.add(new ArrayList<Integer>());
            return results;
        }

        Arrays.sort(nums);
        ArrayList<Integer> list = new ArrayList<Integer>();
        int[] visited = new int[nums.length];
        for ( int i = 0; i < visited.length; i++){
            visited[i] = 0;
        }

        helper(results, list, visited, nums);
        return results;
    }


    public void helper(ArrayList<List<Integer>> results,
                       ArrayList<Integer> list, int[] visited, int[] nums) {

        if(list.size() == nums.length) {
            results.add(new ArrayList<Integer>(list));
            return;
        }

        for(int i = 0; i < nums.length; i++) {
            if ( visited[i] == 1 ||
                    ( i != 0 && visited[i-1] == 0 && nums[i] == nums[i - 1])){
                continue;
            }

            visited[i] = 1;
            list.add(nums[i]);
            helper(results, list, visited, nums);
            list.remove(list.size() - 1);
            visited[i] = 0;
        }
    }


//    48. Rotate Image
//    You are given an n x n 2D matrix representing an image.
//
//    Rotate the image by 90 degrees (clockwise).
//
//    Follow up:
//    Could you do this in-place?

    public void rotate(int[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return;
        }

        int length = matrix.length;

        for (int i = 0; i < length / 2; i++) {
            for (int j = 0; j < (length + 1) / 2; j++){
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[length - j - 1][i];
                matrix[length -j - 1][i] = matrix[length - i - 1][length - j - 1];
                matrix[length - i - 1][length - j - 1] = matrix[j][length - i - 1];
                matrix[j][length - i - 1] = tmp;
            }
        }
    }


//    49. Group Anagrams
//    Given an array of strings, group anagrams together.
//
//    For example, given: ["eat", "tea", "tan", "ate", "nat", "bat"],
//    Return:
//
//            [
//            ["ate", "eat","tea"],
//            ["nat","tan"],
//            ["bat"]
//            ]

    public List<List<String>> groupAnagrams(String[] strs) {
        List<List<String>> result = new ArrayList<List<String>>();

        HashMap<String, ArrayList<String>> map = new HashMap<String, ArrayList<String>>();
        for(String str: strs){
            char[] arr = new char[26];
            for(int i=0; i<str.length(); i++){
                arr[str.charAt(i)-'a']++;
            }
            //String ns = new String(arr);
            String ns = Arrays.toString(arr);

            if(map.containsKey(ns)){
                map.get(ns).add(str);
            }else{
                ArrayList<String> al = new ArrayList<String>();
                al.add(str);
                map.put(ns, al);
            }
        }

        result.addAll(map.values());

        return result;
    }


//    50. Pow(x, n)

    public double pow(double x, int n) {
        if(n == 0)
            return 1;
        if(n<0){
            n = -n;
            x = 1/x;
        }

        return (n%2 == 0) ? pow(x*x, n/2) : x*pow(x*x, n/2);
    }

    /**
     * 51. N-Queens
     *
     * The n-queens puzzle is the problem of placing n queens on
     * an n×n chessboard such that no two queens attack each other.
     Given an integer n, return all distinct solutions to the n-queens puzzle.
     Each solution contains a distinct board configuration of
     the n-queens' placement, where 'Q' and '.' both indicate a queen
     and an empty space respectively.
     For example,
     There exist two distinct solutions to the 4-queens puzzle:
     [
     [".Q..",  // Solution 1
     "...Q",
     "Q...",
     "..Q."],
     ["..Q.",  // Solution 2
     "Q...",
     "...Q",
     ".Q.."]
     ]
     */

    public List<List<String>> solveNQueens(int n) {
        List<List<String>> result = new ArrayList<>();
        if (n <= 0) {
            return result;
        }
        helper(n, new ArrayList<>(), result);
        return result;
    }

    private void helper(int n, ArrayList<Integer> col, List<List<String>> result) {
        if (col.size() == n) {
            result.add(drawChessBoard(col));
            return;
        }

        for (int i = 0; i < n; i++) {
            if (!isValid(col, i)) {
                continue;
            }
            col.add(i);
            helper(n, col, result);
            col.remove(col.size() - 1);
        }
    }

    private boolean isValid(ArrayList<Integer> col, int next) {
        int row = col.size();
        for (int i = 0; i < row; i++) {
            if (next == col.get(i)) {
                return false;
            }
            // x1-y1=x2-y2
            if (i - col.get(i)  == row - next) {
                return false;
            }
            //x1+y1=x2+y2
            if (i + col.get(i)  == row + next) {
                return false;
            }
        }
        return true;
    }

    private ArrayList<String> drawChessBoard(ArrayList<Integer> col) {
        ArrayList<String> chessBoard = new ArrayList<>();

        for (int i = 0; i < col.size(); i++) {
            String row = "";
            for (int j = 0; j < col.size(); j++) {
                if (col.get(j) == i) {
                    row += "Q";
                } else {
                    row += ".";
                }
            }
            chessBoard.add(row);
        }
        return chessBoard;
    }


    /**
     * 52. N-Queens II
     *
     * Follow up for N-Queens problem.
     * Now, instead outputting board configurations, return the total number of distinct solutions.
     */
            /**credit: https://discuss.leetcode.com/topic/29626/easiest-java-solution-1ms-98-22*/
    int count = 0;

    public int totalNQueens(int n) {
        boolean[] cols = new boolean[n];
        boolean[] diagnol = new boolean[2 * n];
        boolean[] antiDiagnol = new boolean[2 * n];
        backtracking(0, cols, diagnol, antiDiagnol, n);
        return count;
    }

    private void backtracking(int row, boolean[] cols, boolean[] diagnol, boolean[] antiDiagnol, int n) {
        if (row == n) {
            count++;
        }
        //for each row, try different col
        for (int col = 0; col < n; col++) {
            // function of diagnol: y=-x+ a
            //x and y is used to calculate diagnol
            int x = col - row + n; //add n in order to make result positive
            int y = col + row;
            if (cols[col] || diagnol[x] || antiDiagnol[y]) {
                continue;
            }
            cols[col] = true;
            diagnol[x] = true;
            antiDiagnol[y] = true;
            backtracking(row + 1, cols, diagnol, antiDiagnol, n);
            cols[col] = false;
            diagnol[x] = false;
            antiDiagnol[y] = false;
        }
    }

//    53. Maximum Subarray
//    Find the contiguous subarray within an array (containing at least one number)
// which has the largest sum.
//
//    For example, given the array [-2,1,-3,4,-1,2,1,-5,4],
//    the contiguous subarray [4,-1,2,1] has the largest sum = 6.


    public int maxSubArray(int[] A) {
        if (A == null || A.length == 0){
            return 0;
        }
        int max = Integer.MIN_VALUE, sum = 0;

        for (int i = 0; i < A.length; i++) {
            sum = Math.max(sum + A[i], A[i]);//start from positive, or keep positive when meet new positive
            max = Math.max(max, sum);
        }

        return max;
    }


    /**
     * 918. Maximum Sum Circular Subarray
     * Medium
     *
     * Given a circular array C of integers represented by A, find the maximum possible sum of a non-empty subarray of C.
     *
     * Here, a circular array means the end of the array connects to the beginning of the array.  (Formally, C[i] = A[i] when 0 <= i < A.length, and C[i+A.length] = C[i] when i >= 0.)
     *
     * Also, a subarray may only include each element of the fixed buffer A at most once.  (Formally, for a subarray C[i], C[i+1], ..., C[j], there does not exist i <= k1, k2 <= j with k1 % A.length = k2 % A.length.)
     *
     *
     *
     * Example 1:
     *
     * Input: [1,-2,3,-2]
     * Output: 3
     * Explanation: Subarray [3] has maximum sum 3
     *
     * Example 2:
     *
     * Input: [5,-3,5]
     * Output: 10
     * Explanation: Subarray [5,5] has maximum sum 5 + 5 = 10
     *
     * Example 3:
     *
     * Input: [3,-1,2,-1]
     * Output: 4
     * Explanation: Subarray [2,-1,3] has maximum sum 2 + (-1) + 3 = 4
     *
     * Example 4:
     *
     * Input: [3,-2,2,-3]
     * Output: 3
     * Explanation: Subarray [3] and [3,-2,2] both have maximum sum 3
     *
     * Example 5:
     *
     * Input: [-2,-3,-1]
     * Output: -1
     * Explanation: Subarray [-1] has maximum sum -1
     *
     *
     *
     * Note:
     *
     *     -30000 <= A[i] <= 30000
     *     1 <= A.length <= 30000
     */

    /**
     * I guess you know how to solve max subarray sum (without circular).
     * If not, you can have a reference here: 53. Maximum Subarray
     *
     * So there are two case.
     *
     *     The first is that the subarray take only a middle part, and we know how to find the max subarray sum.
     *     The second is that the subarray take a part of head array and a part of tail array.
     *     We can transfer this case to the first one.
     *     The maximum result equals to the total sum minus the minimum subarray sum.
     *
     * Here is a diagram by @motorix:
     * image
     *
     * So the max subarray circular sum equals to
     * max(the max subarray sum, the total sum - the min subarray sum)
     *
     * One** corner case** to pay attention:
     * If all number are negative,
     * return the maximum one,
     * (which equals to the max subarray sum)
     */

    public int maxSubarraySumCircular(int[] A) {
        int total = 0, maxSum = -30000, curMax = 0;
        int minSum = 30000, curMin = 0;
        for (int a : A) {
            curMax = Math.max(curMax + a, a);
            maxSum = Math.max(maxSum, curMax);
            curMin = Math.min(curMin + a, a);
            minSum = Math.min(minSum, curMin);
            total += a;
        }
        return maxSum > 0 ? Math.max(maxSum, total - minSum) : maxSum;
    }

    // suggest to use 59
//    54. Spiral Matrix
//    Given a matrix of m x n elements (m rows, n columns),
// return all elements of the matrix in spiral order.
//
//    For example,
//    Given the following matrix:
//
//            [
//            [ 1, 2, 3 ],
//            [ 4, 5, 6 ],
//            [ 7, 8, 9 ]
//            ]
//    You should return [1,2,3,6,9,8,7,4,5].

    public ArrayList<Integer> spiralOrder(int[][] matrix) {
        ArrayList<Integer> rst = new ArrayList<Integer>();
        if(matrix == null || matrix.length == 0)
            return rst;

        int rows = matrix.length;
        int cols = matrix[0].length;
        int count = 0;
        while(count * 2 < rows && count * 2 < cols){
            for(int i = count; i < cols-count; i++)
                rst.add(matrix[count][i]);


            for(int i = count+1; i< rows-count; i++)
                rst.add(matrix[i][cols-count-1]);

            if(rows - 2 * count == 1 || cols - 2 * count == 1)  // if only one row /col remains
                break;

            for(int i = cols-count-2; i>=count; i--)
                rst.add(matrix[rows-count-1][i]);

            for(int i = rows-count-2; i>= count+1; i--)
                rst.add(matrix[i][count]);

            count++;
        }
        return rst;
    }

//    45. Jump Game II

//    55. Jump Game
//    Given an array of non-negative integers, you are initially positioned at
// the first index of the array.
//
//    Each element in the array represents your maximum jump length at that position.
//
//            Determine if you are able to reach the last index.
//
//    For example:
//    A = [2,3,1,1,4], return true.
//
//    A = [3,2,1,0,4], return false.

    public boolean canJump(int[] A) {
        // think it as merging n intervals
        if (A == null || A.length == 0) {
            return false;
        }
        int farthest = A[0];
        for (int i = 1; i < A.length; i++) {
            if (i <= farthest && A[i] + i >= farthest) {
                farthest = A[i] + i;
            }
        }
        return farthest >= A.length - 1;
    }

//    56. Merge Intervals
//    Given a collection of intervals, merge all overlapping intervals.
//
//    For example,
//    Given [1,3],[2,6],[8,10],[15,18],
//            return [1,6],[8,10],[15,18].


//    //    Definition for an interval.
//    public class Interval {
//        int start;
//        int end;
//        Interval() { start = 0; end = 0; }
//        Interval(int s, int e) { start = s; end = e; }
//    }


    public ArrayList<Interval> merge(ArrayList<Interval> intervals) {
        if (intervals == null || intervals.size() <= 1) {
            return intervals;
        }

        Collections.sort(intervals, (a, b) -> a.start - b.start);

        ArrayList<Interval> result = new ArrayList<Interval>();
        Interval last = intervals.get(0);
        for (int i = 1; i < intervals.size(); i++) {
            Interval curt = intervals.get(i);
            if (curt.start <= last.end ){
                last.end = Math.max(last.end, curt.end);
            }else{
                result.add(last);
                last = curt;
            }
        }

        result.add(last);
        return result;
    }


//    57. Insert Interval
//    Given a set of non-overlapping intervals, insert a new interval into
// the intervals (merge if necessary).
//
//    You may assume that the intervals were initially sorted according to their start times.
//
//    Example 1:
//    Given intervals [1,3],[6,9], insert and merge [2,5] in as [1,5],[6,9].
//
//    Example 2:
//    Given [1,2],[3,5],[6,7],[8,10],[12,16], insert and merge [4,9] in as [1,2],[3,10],[12,16].
//
//    This is because the new interval [4,9] overlaps with [3,5],[6,7],[8,10].

    public ArrayList<Interval> insert(ArrayList<Interval> intervals, Interval newInterval) {
        if (newInterval == null || intervals == null) {
            return intervals;
        }

        ArrayList<Interval> results = new ArrayList<Interval>();
        int insertPos = 0;

        for (Interval interval : intervals) {
            if (interval.end < newInterval.start) {
                results.add(interval);
                insertPos++;
            } else if (interval.start > newInterval.end) {
                results.add(interval);
            } else {
                newInterval.start = Math.min(interval.start, newInterval.start);
                newInterval.end = Math.max(interval.end, newInterval.end);
            }
        }

        results.add(insertPos, newInterval);

        return results;
    }

//    58. Length of Last Word
//    Given a string s consists of upper/lower-case alphabets and empty space characters ' ',
// return the length of last word in the string.
//
//    If the last word does not exist, return 0.
//
//    Note: A word is defined as a character sequence consists of non-space characters only.
//
//            For example,
//    Given s = "Hello World",
//    return 5.

    public int lengthOfLastWord(String s) {
        int len=s.length(), lastLength=0;

        while(len > 0 && s.charAt(len-1)==' '){
            len--;
        }

        while(len > 0 && s.charAt(len-1)!=' '){
            lastLength++;
            len--;
        }

        return lastLength;
    }

//    59. Spiral Matrix II
//    Given an integer n, generate a square matrix filled with elements from 1 to n2 in spiral order.
//
//            For example,
//    Given n = 3,
//
//    You should return the following matrix:
//            [
//            [ 1, 2, 3 ],
//            [ 8, 9, 4 ],
//            [ 7, 6, 5 ]
//            ]

    public int[][] generateMatrix(int n) {
        // Declaration
        int[][] matrix = new int[n][n];

        // Edge Case
        if (n == 0) {
            return matrix;
        }

        // Normal Case
        int rowStart = 0;
        int rowEnd = n-1;
        int colStart = 0;
        int colEnd = n-1;
        int num = 1; //change

        while (rowStart <= rowEnd && colStart <= colEnd) {
            for (int i = colStart; i <= colEnd; i ++) {
                matrix[rowStart][i] = num ++; //change
            }
            rowStart ++;

            for (int i = rowStart; i <= rowEnd; i ++) {
                matrix[i][colEnd] = num ++; //change
            }
            colEnd --;

            for (int i = colEnd; i >= colStart; i --) {
                if (rowStart <= rowEnd)
                    matrix[rowEnd][i] = num ++; //change
            }
            rowEnd --;

            for (int i = rowEnd; i >= rowStart; i --) {
                if (colStart <= colEnd)
                    matrix[i][colStart] = num ++; //change
            }
            colStart ++;
        }

        return matrix;
    }

//    60. Permutation Sequence
//    The set [1,2,3,…,n] contains a total of n! unique permutations.
//
//    By listing and labeling all of the permutations in order,
//    We get the following sequence (ie, for n = 3):
//
//            "123"
//            "132"
//            "213"
//            "231"
//            "312"
//            "321"
//    Given n and k, return the kth permutation sequence.
//
//    Note: Given n will be between 1 and 9 inclusive.



//    I'm sure somewhere can be simplified so it'd be nice
// if anyone can let me know. The pattern was that:
//
//    say n = 4, you have {1, 2, 3, 4}
//
//    If you were to list out all the permutations you have
//
//    1 + (permutations of 2, 3, 4)
//
//            2 + (permutations of 1, 3, 4)
//
//            3 + (permutations of 1, 2, 4)
//
//            4 + (permutations of 1, 2, 3)
//
//
//    We know how to calculate the number of permutations of n numbers... n!
// So each of those with permutations of 3 numbers means there are 6 possible permutations.
// Meaning there would be a total of 24 permutations in this particular one.
// So if you were to look for the (k = 14) 14th permutation, it would be in the
//
//    3 + (permutations of 1, 2, 4) subset.
//
//    To programmatically get that, you take k = 13
// (subtract 1 because of things always starting at 0) and
// divide that by the 6 we got from the factorial, which would give you the index of
// the number you want. In the array {1, 2, 3, 4}, k/(n-1)! = 13/(4-1)! = 13/3! = 13/6 = 2.
// The array {1, 2, 3, 4} has a value of 3 at index 2. So the first number is a 3.
//
//    Then the problem repeats with less numbers.
//
//    The permutations of {1, 2, 4} would be:
//
//            1 + (permutations of 2, 4)
//
//            2 + (permutations of 1, 4)
//
//            4 + (permutations of 1, 2)
//
//    But our k is no longer the 14th, because in the previous step,
// we've already eliminated the 12 4-number permutations starting with 1 and 2.
// So you subtract 12 from k.. which gives you 1. Programmatically that would be...
//
//    k = k - (index from previous) * (n-1)! = k - 2*(n-1)! = 13 - 2*(3)! = 1
//
//    In this second step, permutations of 2 numbers has only 2 possibilities,
// meaning each of the three permutations listed above a has two possibilities,
// giving a total of 6. We're looking for the first one,
// so that would be in the 1 + (permutations of 2, 4) subset.
//
//    Meaning: index to get number from is k / (n - 2)! = 1 / (4-2)! = 1 / 2! = 0..
// from {1, 2, 4}, index 0 is 1
//
//
//    so the numbers we have so far is 3, 1... and then repeating without explanations.
//
//
//    {2, 4}
//
//    k = k - (index from pervious) * (n-2)! = k - 0 * (n - 2)! = 1 - 0 = 1;
//
//    third number's index = k / (n - 3)! = 1 / (4-3)! = 1/ 1! = 1... from {2, 4}, index 1 has 4
//
//    Third number is 4
//
//
//    {2}
//
//    k = k - (index from pervious) * (n - 3)! = k - 1 * (4 - 3)! = 1 - 1 = 0;
//
//    third number's index = k / (n - 4)! = 0 / (4-4)! = 0/ 1 = 0... from {2}, index 0 has 2
//
//    Fourth number is 2

    public String getPermutation(int n, int k) {
        StringBuilder sb = new StringBuilder();
        boolean[] used = new boolean[n];

        k = k - 1;
        int factor = 1;
        for (int i = 1; i < n; i++) { //not include n
            factor *= i;
        }

        for (int i = 0; i < n; i++) {
            int index = k / factor;
            k = k % factor;

            for (int j = 0; j < n; j++) {// index is decided, then we try to find the previous used digit, then skip it and find the digit. in general, we try to find the index-ed digit after excluding used digit
                if (used[j] == false) {
                    if (index == 0) {
                        used[j] = true;
                        sb.append((char) ('0' + j + 1));
                        break;
                    } else {
                        index--;
                    }
                }
            }

            if (i < n - 1) {
                factor = factor / (n - 1 - i);
            }
        }

        return sb.toString();
    }


//    61. Rotate List
//    Given a list, rotate the list to the right by n places, where k is non-negative.
//
//    For example:
//    Given 1->2->3->4->5->NULL and k = 2,
//            return 4->5->1->2->3->NULL.


    public ListNode rotateRight(ListNode head, int n) {
        if (head==null||head.next==null) return head;
        ListNode dummy=new ListNode(0);
        dummy.next=head;
        ListNode fast=dummy,slow=dummy;

        int i;
        for (i=0;fast.next!=null;i++)//Get the total length
            fast=fast.next;

        for (int j=i-n%i;j>0;j--) //Get the i-n%i th node
            slow=slow.next;

        fast.next=dummy.next; //Do the rotation
        dummy.next=slow.next;
        slow.next=null;

        return dummy.next;
    }

//    62. Unique Paths
//    A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).
//
//    The robot can only move either down or right at any point in time.
// The robot is trying to reach the bottom-right corner of the grid
// (marked 'Finish' in the diagram below).
//
//    How many possible unique paths are there?

    public int uniquePaths(int m, int n) {
        if (m == 0 || n == 0) {
            return 1;
        }

        int[][] sum = new int[m][n];
        for (int i = 0; i < m; i++) {
            sum[i][0] = 1;
        }
        for (int i = 0; i < n; i++) {
            sum[0][i] = 1;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                sum[i][j] = sum[i - 1][j] + sum[i][j - 1];
            }
        }
        return sum[m - 1][n - 1];
    }


//    63. Unique Paths II
//    Follow up for "Unique Paths":
//
//    Now consider if some obstacles are added to the grids. How many unique paths would there be?
//
//    An obstacle and empty space is marked as 1 and 0 respectively in the grid.
//
//    For example,
//    There is one obstacle in the middle of a 3x3 grid as illustrated below.
//
//    [
//            [0,0,0],
//            [0,1,0],
//            [0,0,0]
//            ]
//    The total number of unique paths is 2.

    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        if (obstacleGrid == null || obstacleGrid.length == 0 || obstacleGrid[0].length == 0) {
            return 0;
        }

        int n = obstacleGrid.length;
        int m = obstacleGrid[0].length;
        int[][] paths = new int[n][m];

        for (int i = 0; i < n; i++) {
            if (obstacleGrid[i][0] != 1) {
                paths[i][0] = 1;
            } else {
                break;
            }
        }

        for (int i = 0; i < m; i++) {
            if (obstacleGrid[0][i] != 1) {
                paths[0][i] = 1;
            } else {
                break;
            }
        }

        for (int i = 1; i < n; i++) {
            for (int j = 1; j < m; j++) {
                if (obstacleGrid[i][j] != 1) {
                    paths[i][j] = paths[i - 1][j] + paths[i][j - 1];
                } else {
                    paths[i][j] = 0;
                }
            }
        }

        return paths[n - 1][m - 1];
    }


//    64. Minimum Path Sum
//    Given a m x n grid filled with non-negative numbers,
// find a path from top left to bottom right
// which minimizes the sum of all numbers along its path.
//
//    Note: You can only move either down or right at any point in time.

    public int minPathSum(int[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0) {
            return 0;
        }

        int M = grid.length;
        int N = grid[0].length;
        int[][] sum = new int[M][N];

        sum[0][0] = grid[0][0];

        for (int i = 1; i < M; i++) {
            sum[i][0] = sum[i - 1][0] + grid[i][0];
        }

        for (int i = 1; i < N; i++) {
            sum[0][i] = sum[0][i - 1] + grid[0][i];
        }

        for (int i = 1; i < M; i++) {
            for (int j = 1; j < N; j++) {
                sum[i][j] = Math.min(sum[i - 1][j], sum[i][j - 1]) + grid[i][j];
            }
        }

        return sum[M - 1][N - 1];
    }


//    65. Valid Number
//    Validate if a given string is numeric.
//
//    Some examples:
//            "0" => true
//            " 0.1 " => true
//            "abc" => false
//            "1 a" => false
//            "2e10" => true
//    Note: It is intended for the problem statement to be ambiguous.
// You should gather all requirements up front before implementing one.

    public boolean isNumber(String s) {
        s = s.trim();
        boolean pointSeen = false;
        boolean eSeen = false;
        boolean numberSeen = false;
        boolean numberAfterE = true;

        for(int i=0; i<s.length(); i++) {
            if('0' <= s.charAt(i) && s.charAt(i) <= '9') {
                numberSeen = true;
                numberAfterE = true;
            } else if(s.charAt(i) == '.') {
                if(eSeen || pointSeen) {
                    return false;
                }
                pointSeen = true;
            } else if(s.charAt(i) == 'e') {
                if(eSeen || !numberSeen) {
                    return false;
                }
                numberAfterE = false;
                eSeen = true;
            } else if(s.charAt(i) == '-' || s.charAt(i) == '+') {
                if(i != 0 && s.charAt(i-1) != 'e') {
                    return false;
                }
            } else {
                return false;
            }
        }

        return numberSeen && numberAfterE;
    }


//    66. Plus One
//    Given a non-negative integer represented as a non-empty array of digits, plus one to the integer.
//
//    You may assume the integer do not contain any leading zero, except the number 0 itself.
//
//    The digits are stored such that the most significant digit is at the head of the list.

    public int[] plusOne(int[] digits) {
        int carries = 1;
        for(int i = digits.length-1; i>=0 && carries > 0; i--){  // fast break when carries equals zero
            int sum = digits[i] + carries;
            digits[i] = sum % 10;
            carries = sum / 10;
        }
        if(carries == 0)
            return digits;

        int[] rst = new int[digits.length+1];
        rst[0] = 1;
        for(int i=1; i< rst.length; i++){
            rst[i] = digits[i-1];
        }
        return rst;
    }


//    67. Add Binary
//    Given two binary strings, return their sum (also a binary string).
//
//    For example,
//            a = "11"
//    b = "1"
//    Return "100".

    public String addBinary(String a, String b) {
        StringBuilder sb = new StringBuilder();
        int i = a.length() - 1, j = b.length() -1;
        int carry = 0;
        while (i >= 0 || j >= 0) {
            int sum = carry;
            if (j >= 0) sum += b.charAt(j--) - '0';
            if (i >= 0) sum += a.charAt(i--) - '0';
            sb.append(sum % 2);
            carry = sum / 2;
        }
        if (carry != 0) sb.append(carry);
        return sb.reverse().toString();
    }


    /**
     * 68. Text Justification
     Given an array of words and a length L, format the text such that
     each line has exactly L characters and is fully (left and right) justified.
     You should pack your words in a greedy approach; that is,
     pack as many words as you can in each line.
     Pad extra spaces ' ' when necessary so that each line has exactly L characters.
     Extra spaces between words should be distributed as evenly as possible.
     If the number of spaces on a line do not divide evenly between words,
     the empty slots on the left will be assigned more spaces than the slots on the right.
     For the last line of text, it should be left justified and no extra space
     is inserted between words.
     For example,
     words: ["This", "is", "an", "example", "of", "text", "justification."]
     L: 16.
     Return the formatted lines as:
     [
     "This    is    an",
     "example  of text",
     "justification.  "
     ]
     Note: Each word is guaranteed not to exceed L in length.
     Corner Cases:
     A line other than the last line might contain only one word. What should you do in this case?
     In this case, that line should be left-justified.
     */
    public List<String> fullJustify(String[] words, int L) {
        List<String> result = new ArrayList();
        if (words == null || words.length == 0) {
            return result;
        }
        int count = 0;
        int last = 0;
        for (int i = 0; i < words.length; i++) {
            if (count + words[i].length() + (i - last) > L) {
                int spaceNum = 0;
                int extraNum = 0;
                if (i - last - 1 > 0) {
                    spaceNum = (L - count) / (i - last - 1);
                    extraNum = (L - count) % (i - last - 1);
                }
                StringBuilder sb = new StringBuilder();
                for (int j = last; j < i; j++) {
                    sb.append(words[j]);
                    if (j < i - 1) {
                        for (int k = 0; k < spaceNum; k++) {
                            sb.append(" ");
                        }
                        if (extraNum > 0) {
                            sb.append(" ");
                        }
                        extraNum--;
                    }
                }
                //seems we don't need this line
//                for (int j = sb.length(); j < L; j++) {
//                    sb.append(" ");
//                }
                result.add(sb.toString());
                count = 0;
                last = i;
            }
            count += words[i].length();
        }
        StringBuilder sb = new StringBuilder();
        for (int i = last; i < words.length; i++) {
            sb.append(words[i]);
            if (sb.length() < L) {
                sb.append(" ");
            }
        }
        for (int i = sb.length(); i < L; i++) {
            sb.append(" ");
        }
        result.add(sb.toString());
        return result;
    }

//    69. Sqrt(x)
//    Implement int sqrt(int x).
//
//    Compute and return the square root of x.

    public int sqrt(int x) {
        // find the last number which square of it <= x
        if (x == 0) return 0;
        int start = 1, end = x;
        while (start < end) {
            int mid = start + (end - start) / 2;
            if (mid <= x / mid && (mid + 1) > x / (mid + 1))// Found the result
                return mid;
            else if (mid > x / mid)// Keep checking the left part
                end = mid;
            else
                start = mid + 1;// Keep checking the right part
        }
        return start;
    }


//    70. Climbing Stairs
//    You are climbing a stair case. It takes n steps to reach to the top.
//
//    Each time you can either climb 1 or 2 steps.
// In how many distinct ways can you climb to the top?

//    The Fibonacci numbers are generated by setting F0=0, F1=1,
// and then using the recursive formula
//
//
//            Fn = Fn-1 + Fn-2
//
//    to get the rest. Thus the sequence begins: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...
// This sequence of Fibonacci numbers arises all over mathematics and also in nature.
//
//    The problem seems to be a dynamic programming one. Hint: the tag also suggests that!
//    Here are the steps to get the solution incrementally.
//
//    Base cases:
//            if n <= 0, then the number of ways should be zero.
//            if n == 1, then there is only way to climb the stair.
//    if n == 2, then there are two ways to climb the stairs.
// One solution is one step by another; the other one is two steps at one time.
//
//    The key intuition to solve the problem is that given a number of stairs n,
// if we know the number ways to get to the points [n-1] and [n-2] respectively,
// denoted as n1 and n2 , then the total ways to get to the point [n] is n1 + n2.
// Because from the [n-1] point, we can take one single step to reach [n].
// And from the [n-2] point, we could take two steps to get there.
// There is NO overlapping between these two solution sets, because we differ in the final step.
//
//    Now given the above intuition, one can construct an array
// where each node stores the solution for each number n.
// Or if we look at it closer, it is clear that this is basically a fibonacci number,
// with the starting numbers as 1 and 2, instead of 1 and 1.

    public int climbStairs(int n) {
        if (n <= 1) {
            return 1;
        }
        if(n==2) return 2;
        int second = 2, first = 1;
        int now = 0;
        for (int i = 3; i <= n; i++) {
            now = first + second;
            first = second;
            second = now;
        }
        return now;
    }


//    71. Simplify Path
//    Total Accepted: 76384
//    Total Submissions: 314458
//    Difficulty: Medium
//    Contributors: Admin
//    Given an absolute path for a file (Unix-style), simplify it.
//
//    For example,
//            path = "/home/", => "/home"
//    path = "/a/./b/../../c/", => "/c"

    public String simplifyPath(String path) {
        StringBuilder result = new StringBuilder();
        String[] stubs = path.split("/+");
        ArrayList<String> paths = new ArrayList<String>();
        for (String s : stubs){
            if(s.equals("..")){
                if(paths.size() > 0){
                    paths.remove(paths.size() - 1);
                }
            }
            else if (!s.equals(".") && !s.equals("")){
                paths.add(s);
            }
        }
        for (String s : paths){
            result.append("/"+s);
        }
        return result.length()==0?"/":result.toString();
    }


//    72. Edit Distance
//    Given two words word1 and word2, find the minimum number of
// steps required to convert word1 to word2. (each operation is counted as 1 step.)
//
//    You have the following 3 operations permitted on a word:
//
//    a) Insert a character
//    b) Delete a character
//    c) Replace a character

    public int minDistance(String word1, String word2) {
        int n = word1.length();
        int m = word2.length();

        int[][] dp = new int[n+1][m+1];
        for(int i=0; i< m+1; i++){
            dp[0][i] = i;
        }
        for(int i=0; i<n+1; i++){
            dp[i][0] = i;
        }

        for(int i = 1; i<n+1; i++){
            for(int j=1; j<m+1; j++){
                if(word1.charAt(i-1) == word2.charAt(j-1)){
                    dp[i][j] = dp[i-1][j-1];
                }else{                     //Replace a character = dp[i-1][j-1] +1
                    dp[i][j] = 1 + Math.min(dp[i-1][j-1], Math.min(dp[i-1][j],dp[i][j-1]));
                }
            }
        }
        return dp[n][m];
    }



//    73. Set Matrix Zeroes
//    Given a m x n matrix, if an element is 0, set its entire row and column to 0. Do it in place.

    public void setZeroes(int[][] matrix) {
        if(matrix == null || matrix.length == 0)
            return;

        int rows = matrix.length;
        int cols = matrix[0].length;

        boolean empty_row0 = false;
        boolean empty_col0 = false;
        for(int i = 0; i < cols; i++){
            if(matrix[0][i] == 0){
                empty_row0 = true;
                break;
            }
        }

        for(int i = 0; i < rows; i++){
            if(matrix[i][0] == 0){
                empty_col0 = true;
                break;
            }
        }

        for(int i = 1; i < rows; i++) {
            for(int j =1; j<cols; j++){
                if(matrix[i][j] == 0){
                    matrix[0][j] = 0;
                    matrix[i][0] = 0;
                }
            }
        }

        for(int i = 1; i<rows; i++) {
            for (int j=1; j< cols; j++) {
                if(matrix[0][j] == 0 || matrix[i][0] == 0)
                    matrix[i][j] = 0;
            }
        }

        if(empty_row0){
            for(int i = 0; i < cols; i++){
                matrix[0][i] = 0;
            }
        }

        if(empty_col0){
            for(int i = 0; i < rows; i++){
                matrix[i][0] = 0;
            }
        }

    }



//    74. Search a 2D Matrix
//    Write an efficient algorithm that searches for a value in an m x n matrix.
// This matrix has the following properties:
//
//    Integers in each row are sorted from left to right.
//    The first integer of each row is greater than the last integer of the previous row.
//    For example,
//
//    Consider the following matrix:
//
//            [
//            [1,   3,  5,  7],
//            [10, 11, 16, 20],
//            [23, 30, 34, 50]
//            ]
//    Given target = 3, return true.

    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0) {
            return false;
        }
        if (matrix[0] == null || matrix[0].length == 0) {
            return false;
        }

        int row = matrix.length, column = matrix[0].length;
        int start = 0, end = row * column - 1;

        while (start + 1 < end) {
            int mid = start + (end - start) / 2;
            int number = matrix[mid / column][mid % column];
            if (number == target) {
                return true;
            } else if (number < target) {
                start = mid;
            } else {
                end = mid;
            }
        }

        if (matrix[start / column][start % column] == target) {
            return true;
        } else if (matrix[end / column][end % column] == target) {
            return true;
        }

        return false;
    }


//    75. Sort Colors
//    Given an array with n objects colored red, white or blue,
// sort them so that objects of the same color are adjacent,
// with the colors in the order red, white and blue.
//
//    Here, we will use the integers 0, 1, and 2 to represent
// the color red, white, and blue respectively.

    public void sortColors(int[] a) {
        if (a == null || a.length <= 1) {
            return;
        }

        int pl = 0;
        int pr = a.length - 1;
        int i = 0;
        while (i <= pr) {
            if (a[i] == 0) {
                swap1(a, pl, i);
                pl++;
                i++;
            } else if(a[i] == 1) {
                i++;
            } else {
                swap1(a, pr, i);
                pr--;
            }
        }
    }

     void swap1(int[] a, int i, int j) {
        int tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
    }


    /**
     * 76. Minimum Window Substring
     *
     * Given a string S and a string T, find the minimum window in S
     * which will contain all the characters in T in complexity O(n).
     For example,
     S = "ADOBECODEBANC"
     T = "ABC"
     Minimum window is "BANC".

     Note:
     If there is no such window in S that covers all characters in T, return the empty string "".
     If there are multiple such windows, you are guaranteed that
     there will always be only one unique minimum window in S.
     */

    public String minWindow(String s, String t) {
        int[] counts = new int[256];
        for (char c : t.toCharArray()) {
            counts[c]++;
        }

        int start = 0;
        int end = 0;
        int remain = t.length();

        int minStart = 0;
        int minLen = Integer.MAX_VALUE;


        while (end < s.length()) {
            if (counts[s.charAt(end)] > 0) {  //only the char in t has positive value
                remain--;
            }

            counts[s.charAt(end)]--;
            end++;

            while (remain == 0) {
                if (end - start < minLen) {
                    minStart = start;
                    minLen = end - start;
                }
                counts[s.charAt(start)]++;
                if (counts[s.charAt(start)] > 0) {
                    remain++;
                }
                start++;
            }
        }

        if (minLen == Integer.MAX_VALUE) {
            return "";
        }
        return s.substring(minStart, minStart + minLen);
    }


//    77. Combinations
//    Given two integers n and k, return all possible combinations of k numbers out of 1 ... n.
//
//            For example,
//    If n = 4 and k = 2, a solution is:
//
//            [
//            [2,4],
//            [3,4],
//            [2,3],
//            [1,2],
//            [1,3],
//            [1,4],
//            ]

    public ArrayList<ArrayList<Integer>> combine(int n, int k) {
        ArrayList<ArrayList<Integer>> rst = new ArrayList<ArrayList<Integer>>();
        ArrayList<Integer> solution = new ArrayList<Integer>();

        helper(rst, solution, n, k, 1);
        return rst;
    }

    private void helper(
            ArrayList<ArrayList<Integer>> rst,
            ArrayList<Integer> solution,
            int n,
            int k,
            int start) {

        if (solution.size() == k){
            rst.add(new ArrayList(solution));
            return;
        }

        for(int i = start; i<= n; i++){
            solution.add(i);

            // the new start should be after the next number after i
            helper(rst, solution, n, k, i+1);
            solution.remove(solution.size() - 1);
        }
    }

//    90. Subsets II
//    Given a collection of integers that might contain duplicates, nums, return all possible subsets.
//
//            Note: The solution set must not contain duplicate subsets.
//
//    For example,
//    If nums = [1,2,2], a solution is:
//
//            [
//            [2],
//            [1],
//            [1,2,2],
//            [2,2],
//            [1,2],
//            []
//            ]

//    78. Subsets
//    Given a set of distinct integers, nums, return all possible subsets.
//
//            Note: The solution set must not contain duplicate subsets.
//
//    For example,
//    If nums = [1,2,3], a solution is:
//
//            [
//            [3],
//            [1],
//            [2],
//            [1,2,3],
//            [1,3],
//            [2,3],
//            [1,2],
//            []
//            ]

    public ArrayList<ArrayList<Integer>> subsets(int[] num) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
        if(num == null || num.length == 0) {
            return result;
        }
        ArrayList<Integer> list = new ArrayList<Integer>();
        Arrays.sort(num);
        subsetsHelper(result, list, num, 0);

        return result;
    }


    private void subsetsHelper(ArrayList<ArrayList<Integer>> result,
                               ArrayList<Integer> list, int[] num, int pos) {

        result.add(new ArrayList<>(list));

        for (int i = pos; i < num.length; i++) {

            list.add(num[i]);
            subsetsHelper(result, list, num, i + 1);
            list.remove(list.size() - 1);
        }
    }


//    79. Word Search
//    Given a 2D board and a word, find if the word exists in the grid.
//
//    The word can be constructed from letters of sequentially adjacent cell,
// where "adjacent" cells are those horizontally or vertically neighboring.
// The same letter cell may not be used more than once.
//
//    For example,
//    Given board =
//
//    [
//            ['A','B','C','E'],
//            ['S','F','C','S'],
//            ['A','D','E','E']
//            ]
//    word = "ABCCED", -> returns true,
//    word = "SEE", -> returns true,
//    word = "ABCB", -> returns false.


    public boolean exist(char[][] board, String word) {
        if(board == null || board.length == 0)
            return false;
        if(word.length() == 0)
            return true;

        for(int i = 0; i< board.length; i++){
            for(int j=0; j< board[0].length; j++){
                if(board[i][j] == word.charAt(0)){

                    boolean rst = find(board, i, j, word, 0);
                    if(rst)
                        return true;
                }
            }
        }
        return false;
    }

    private boolean find(char[][] board, int i, int j, String word, int start){
        if(start == word.length())
            return true;

        if (i < 0 || i>= board.length ||
                j < 0 || j >= board[0].length ||
                board[i][j] != word.charAt(start)){
            return false;
        }
        // we can also use visited
        board[i][j] = '#'; // should remember to mark it
        boolean rst = find(board, i-1, j, word, start+1)
                || find(board, i, j-1, word, start+1)
                || find(board, i+1, j, word, start+1)
                || find(board, i, j+1, word, start+1);
        board[i][j] = word.charAt(start);
        return rst;
    }

    //    26 Remove Duplicates from Sorted Array

//    80. Remove Duplicates from Sorted Array II
//    Follow up for "Remove Duplicates":
//    What if duplicates are allowed at most twice?
//
//    For example,
//    Given sorted array nums = [1,1,1,2,2,3],
//
//    Your function should return length = 5, with the first five elements of
// nums being 1, 1, 2, 2 and 3. It doesn't matter what you leave beyond the new length.

    public int removeDuplicates(int[] nums) {
        int i = 0;
        for (int n : nums)
            if (i < 2 || n > nums[i-2])
                nums[i++] = n;
        return i;
    }

    //   33  Search in Rotated Sorted Array

//    81. Search in Rotated Sorted Array II
//    Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.
//
//            (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
//
//    Write a function to determine if a given target is in the array.
//
//    The array may contain duplicates.


    public boolean search(int[] nums, int target) {
        // note here end is initialized to len instead of (len-1)
        int start = 0, end = nums.length;
        while (start < end) {
            int mid = (start + end) / 2;
            if (nums[mid] == target) return true;
            if (nums[start] < nums[mid] ) { // nums[start..mid] is sorted
                // check if target in left half
                if ( nums[start] <= target && target < nums[mid] ) end = mid;
                else start = mid + 1;
            } else if (nums[start] > nums[mid]) { // nums[mid..end] is sorted
                // check if target in right half
                if (target > nums[mid] && target < nums[start]) start = mid + 1;
                else end = mid;
            } else { // have no idea about the array, but we can exclude nums[start] because nums[start] == nums[mid]
                start++;
            }
        }

        return false;
    }


//    82. Remove Duplicates from Sorted List II
//    Given a sorted linked list, delete all nodes that have duplicate numbers,
// leaving only distinct numbers from the original list.
//
//    For example,
//    Given 1->2->3->3->4->4->5, return 1->2->5.
//    Given 1->1->1->2->3, return 2->3.


    public ListNode deleteDuplicates(ListNode head) {
        if(head == null || head.next == null)
            return head;

        ListNode dummy = new ListNode(0);
        dummy.next = head;
        head = dummy;

        while (head.next != null && head.next.next != null) {
            if (head.next.val == head.next.next.val) {
                int val = head.next.val;
                while (head.next != null && head.next.val == val) {
                    head.next = head.next.next;
                }
            } else {
                head = head.next;
            }
        }

        return dummy.next;
    }


//    83. Remove Duplicates from Sorted List
//    Given a sorted linked list, delete all duplicates such that each element appear only once.
//
//    For example,
//    Given 1->1->2, return 1->2.
//    Given 1->1->2->3->3, return 1->2->3.

    public ListNode deleteDuplicates2(ListNode head) {
        if (head == null) {
            return null;
        }

        ListNode node = head;
        while (node.next != null) {
            if (node.val == node.next.val) {
                node.next = node.next.next;
            } else {
                node = node.next;
            }
        }
        return head;
    }


//    84. Largest Rectangle in Histogram
//    Given n non-negative integers representing the histogram's
// bar height where the width of each bar is 1, find the area of
// largest rectangle in the histogram.

    public int largestRectangleArea(int[] height) {
        if (height == null || height.length == 0) {
            return 0;
        }

        Stack<Integer> stack = new Stack<Integer>();
        int max = 0;
        for (int i = 0; i <= height.length; i++) {
            int curt = (i == height.length) ? -1 : height[i];
            while (!stack.isEmpty() && curt <= height[stack.peek()]) {
                int h = height[stack.pop()];
                int w = stack.isEmpty() ? i : i - 1  - (stack.peek() + 1) + 1;
                max = Math.max(max, h * w);
            }
            stack.push(i);
        }

        return max;
    }

    /**
     * 85. Maximal Rectangle
     *
     * Given a 2D binary matrix filled with 0's and 1's,
     * find the largest rectangle containing only 1's and return its area.
     For example, given the following matrix:
     1 0 1 0 0
     1 0 1 1 1
     1 1 1 1 1
     1 0 0 1 0
     Return 6.
     */

    public int maximalRectangle(char[][] matrix) {
        if (matrix==null||matrix.length==0||matrix[0].length==0)
            return 0;
        int rLen = matrix.length;       // row length
        int cLen = matrix[0].length;    // column length

        // height array
        int[] h = new int[cLen+1];
        h[cLen]=0;
        int max = 0;


        for (int row=0;row<rLen;row++) {
            Stack<Integer> s = new Stack<Integer>();
            for (int i=0;i<cLen+1;i++) {
                if (i<cLen) {
                    if (matrix[row][i] == '1')
                        h[i] += 1;
                    else h[i] = 0;
                }

                while(!s.isEmpty()&&h[i]<h[s.peek()]){
                    int top = s.pop();
                    int area = h[top]*(s.isEmpty()?i:(i-s.peek()-1));
                    if (area>max)
                        max = area;
                }
                s.push(i);

            }
        }
        return max;
    }

//    public int maximalRectangle(char[][] matrix) {
//        if (matrix.length == 0) {
//            return 0;
//        }
//        int m = matrix.length;
//        int n = matrix[0].length;
//        int[] left = new int[n];
//        int[] right = new int[n];
//        int[] height = new int[n];
//        Arrays.fill(left, 0);
//        Arrays.fill(right, n);
//        Arrays.fill(height, 0);
//        int maxA = 0;
//        for (int i = 0; i < m; i++) {
//            int currLeft = 0;
//            int currRight = n;
//
//            //compute height, this can be achieved from either side
//            for (int j = 0; j < n; j++) {
//                if (matrix[i][j] == '1') {
//                    height[j]++;
//                } else {
//                    height[j] = 0;
//                }
//            }
//
//            //compute left, from left to right
//            for (int j = 0; j < n; j++) {
//                if (matrix[i][j] == '1') {
//                    left[j] = Math.max(left[j], currLeft); //keep the start position
//                } else {
//                    left[j] = 0;
//                    currLeft = j + 1;  //the start position of 1 from left
//                }
//            }
//
//            //compute right, from right to left
//            for (int j = n - 1; j >= 0; j--) {
//                if (matrix[i][j] == '1') {
//                    right[j] = Math.min(right[j], currRight);
//                } else {
//                    right[j] = n;
//                    currRight = j;
//                }
//            }
//
//            //compute rectangle area, this can be achieved from either side
//            for (int j = 0; j < n; j++) {
//                maxA = Math.max(maxA, (right[j] - left[j]) * height[j]);
//            }
//        }
//        return maxA;
//    }

//    86. Partition List
//    Given a linked list and a value x, partition it such that all nodes
// less than x come before nodes greater than or equal to x.
//
//    You should preserve the original relative order of the nodes in each of the two partitions.
//
//    For example,
//    Given 1->4->3->2->5->2 and x = 3,
//    return 1->2->2->4->3->5.

    public ListNode partition(ListNode head, int x) {
        if (head == null) {
            return null;
        }

        ListNode leftDummy = new ListNode(0);
        ListNode rightDummy = new ListNode(0);
        ListNode left = leftDummy, right = rightDummy;

        while (head != null) {
            if (head.val < x) {
                left.next = head;
                left = head;
            } else {
                right.next = head;
                right = head;
            }
            head = head.next;
        }

        right.next = null;
        left.next = rightDummy.next;
        return leftDummy.next;
    }


//    87. Scramble String
//    Given a string s1, we may represent it as a binary tree by
// partitioning it to two non-empty substrings recursively.
//
//    Below is one possible representation of s1 = "great":
//
//    great
//    /    \
//    gr    eat
//    / \    /  \
//    g   r  e   at
//               / \
//               a   t
//    To scramble the string, we may choose any non-leaf node and swap its two children.
//
//    For example, if we choose the node "gr" and swap its two children,
// it produces a scrambled string "rgeat".
//
//    rgeat
//    /    \
//    rg    eat
//    / \    /  \
//    r   g  e   at
//               / \
//              a   t
//    We say that "rgeat" is a scrambled string of "great".
//
//    Similarly, if we continue to swap the children of nodes "eat" and "at",
// it produces a scrambled string "rgtae".
//
//    rgtae
//    /    \
//    rg    tae
//    / \    /  \
//    r   g  ta  e
//           / \
//          t   a
//    We say that "rgtae" is a scrambled string of "great".
//
//    Given two strings s1 and s2 of the same length, determine if s2 is a scrambled string of s1.


    HashMap<String, Boolean> hash = new HashMap<String, Boolean>();

    public boolean isScramble(String s1, String s2) {

        if (s1.length() != s2.length())
            return false;

        if (hash.containsKey(s1 + "#" + s2))
            return hash.get(s1 + "#" + s2);

        int n = s1.length();
        if (n == 1) {
            return s1.charAt(0) == s2.charAt(0);
        }
        for (int k = 1; k < n; ++k) {
            if (isScramble(s1.substring(0, k), s2.substring(0, k)) &&
                    isScramble(s1.substring(k, n), s2.substring(k, n))
                    ||
                    isScramble(s1.substring(0, k), s2.substring(n - k, n)) &&
                            isScramble(s1.substring(k, n), s2.substring(0, n - k))
                    ) {
                hash.put(s1 + "#" + s2, true);
                return true;
            }
        }
        hash.put(s1 + "#" + s2, false);
        return false;
    }


//    88. Merge Sorted Array
//    Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.
//
//            Note:
//    You may assume that nums1 has enough space (size that is greater or equal to m + n)
// to hold additional elements from nums2.
// The number of elements initialized in nums1 and nums2 are m and n respectively.

    public void mergeSortedArray(int[] A, int m, int[] B, int n) {
        int i = m-1, j = n-1, index = m + n - 1;
        while (i >= 0 && j >= 0) {
            if (A[i] > B[j]) {
                A[index--] = A[i--];
            } else {
                A[index--] = B[j--];
            }
        }
        while (i >= 0) {
            A[index--] = A[i--];
        }
        while (j >= 0) {
            A[index--] = B[j--];
        }
    }


//    89. Gray Code
//
//    The gray code is a binary numeral system where two successive values differ in only one bit.
//
//    Given a non-negative integer n representing the total number of bits in the code,
// print the sequence of gray code. A gray code sequence must begin with 0.
//
//    For example, given n = 2, return [0,1,3,2]. Its gray code sequence is:
//
//            00 - 0
//            01 - 1
//            11 - 3
//            10 - 2
//    Note:
//    For a given n, a gray code sequence is not uniquely defined.
//
//    For example, [0,2,3,1] is also a valid gray code sequence according to the above definition.
//
//    For now, the judge is able to judge based on one instance of gray code sequence.
// Sorry about that.


    public ArrayList<Integer> grayCode(int n) {
        ArrayList<Integer> result = new ArrayList<Integer>();
        if (n <= 1) {
            for (int i = 0; i <= n; i++){
                result.add(i);
            }
            return result;
        }
        result = grayCode(n - 1);

        ArrayList<Integer> r1 = new ArrayList<>(result);
        Collections.reverse(r1);

        int x = 1 << (n-1);
        for (int i = 0; i < r1.size(); i++) {
            r1.set(i, r1.get(i) + x);
        }
        result.addAll(r1);
        return result;
    }


//    78. Subsets
//    Given a set of distinct integers, nums, return all possible subsets.
//
//            Note: The solution set must not contain duplicate subsets.


//    90. Subsets II
//    Given a collection of integers that might contain duplicates, nums, return all possible subsets.
//
//            Note: The solution set must not contain duplicate subsets.
//
//    For example,
//    If nums = [1,2,2], a solution is:
//
//            [
//            [2],
//            [1],
//            [1,2,2],
//            [2,2],
//            [1,2],
//            []
//            ]

    public ArrayList<ArrayList<Integer>> subsetsWithDup(int[] nums) {
        // write your code here
        ArrayList<ArrayList<Integer>> results = new ArrayList<>();
        if( nums == null || nums.length == 0 ) return results;

        Arrays.sort(nums);

        ArrayList<Integer> subset = new ArrayList<>();
        helper(nums, 0, subset, results);

        return results;


    }
    public void helper(int[] nums, int startIndex, ArrayList<Integer> subset, ArrayList<ArrayList<Integer>> results){

        results.add(new ArrayList<>(subset));

        for(int i=startIndex; i<nums.length; i++){
            if(i>startIndex && nums[i]==nums[i-1] ){
                continue;
            }
            subset.add(nums[i]);
            helper(nums, i+1, subset, results);
            subset.remove(subset.size()-1);
        }
    }


//    91. Decode Ways
    //            639. Decode Ways II
//    A message containing letters from A-Z is being encoded to numbers using the following mapping:
//
//            'A' -> 1
//            'B' -> 2
//            ...
//            'Z' -> 26
//    Given an encoded message containing digits, determine the total number of ways to decode it.
//
//    For example,
//    Given encoded message "12", it could be decoded as "AB" (1 2) or "L" (12).
//
//    The number of ways decoding "12" is 2.

    public int numDecodings(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        int[] nums = new int[s.length() + 1];
        nums[0] = 1;
        nums[1] = s.charAt(0) != '0' ? 1 : 0;
        for (int i = 2; i <= s.length(); i++) {
            if (s.charAt(i - 1) != '0') {
                nums[i] = nums[i - 1];
            }

            int twoDigits = (s.charAt(i - 2) - '0') * 10 + s.charAt(i - 1) - '0';
            if (twoDigits >= 10 && twoDigits <= 26) {
                nums[i] += nums[i - 2];
            }
        }
        return nums[s.length()];
    }


//    92. Reverse Linked List II
//    Reverse a linked list from position m to n. Do it in-place and in one-pass.
//
//    For example:
//    Given 1->2->3->4->5->NULL, m = 2 and n = 4,
//
//    return 1->4->3->2->5->NULL.
//
//            Note:
//    Given m, n satisfy the following condition:
//            1 ≤ m ≤ n ≤ length of list.

    public ListNode reverseBetween(ListNode head, int m, int n) {
        if (m >= n || head == null) {
            return head;
        }

        ListNode dummy = new ListNode(0);
        dummy.next = head;
        head = dummy;

        for (int i = 1; i < m; i++) {
            if (head == null) {
                return null;
            }
            head = head.next;
        }


        ListNode premNode = head;
        ListNode mNode = head.next;
        ListNode nNode = mNode, postnNode = mNode.next;
        for (int i = m; i < n; i++) {
//            if (postnNode == null) {
//                return null;
//            }
            ListNode temp = postnNode.next;
            postnNode.next = nNode;
            nNode = postnNode;
            postnNode = temp;
        }
        mNode.next = postnNode;
        premNode.next = nNode;

        return dummy.next;
    }


//    206. Reverse Linked List

    public ListNode reverse(ListNode head) {
        ListNode prev = null;
        while (head != null) {
            ListNode temp = head.next;
            head.next = prev;
            prev = head;
            head = temp;
        }
        return prev;
    }


//    93. Restore IP Addresses
//    Given a string containing only digits, restore it by
// returning all possible valid IP address combinations.
//
//    For example:
//    Given "25525511135",
//
//            return ["255.255.11.135", "255.255.111.35"]. (Order does not matter)


    public ArrayList<String> restoreIpAddresses(String s) {
        ArrayList<String> result = new ArrayList<String>();
        ArrayList<String> list = new ArrayList<String>();

        if(s.length() <4 || s.length() > 12)
            return result;

        helper(result, list, s , 0);
        return result;
    }

    public void helper(ArrayList<String> result, ArrayList<String> list, String s, int start){
        if(list.size() == 4){
            if(start != s.length())
                return;

            StringBuffer sb = new StringBuffer();
            for(String tmp: list){
                sb.append(tmp);
                sb.append(".");
            }
            sb.deleteCharAt(sb.length()-1);
            result.add(sb.toString());
            return;
        }

        for(int i=start; i<s.length() && i < start+3; i++){
            String tmp = s.substring(start, i+1);
            if(isvalid(tmp)){
                list.add(tmp);
                helper(result, list, s, i+1);
                list.remove(list.size()-1);
            }
        }
    }

    private boolean isvalid(String s){
        if(s.charAt(0) == '0')
            return s.equals("0"); // to eliminate cases like "00", "10"
        int digit = Integer.valueOf(s);
        return digit >= 0 && digit <= 255;
    }



//    94. Binary Tree Inorder Traversal
//    Given a binary tree, return the inorder traversal of its nodes' values.
//
//    For example:
//    Given binary tree [1,null,2,3],
//            1
//            \
//            2
//            /
//            3
//            return [1,3,2].

//    public ArrayList<Integer> inorderTraversal(TreeNode root) {
//        Stack<TreeNode> stack = new Stack<TreeNode>();
//        ArrayList<Integer> result = new ArrayList<Integer>();
//        TreeNode curt = root;
//        while (curt != null || !stack.empty()) {
//            while (curt != null) {
//                stack.add(curt);
//                curt = curt.left;
//            }
//            curt = stack.pop();
//            result.add(curt.val);
//            curt = curt.right;
//        }
//        return result;
//    }


    public ArrayList<Integer> inorderTraversal(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<>();
        helper95(root,result);
        return result;
    }

    public void helper95(TreeNode root, ArrayList<Integer> result){
        if(root == null) return;
        helper95(root.left, result);
        result.add(root.val);
        helper95(root.right, result);
    }



    //    96. Unique Binary Search Trees
    //Given n, how many structurally unique BST's (binary search trees) that store values 1 ... n?

    /*
The case for 3 elements example
Count[3] = Count[0]*Count[2]  (1 as root)
      + Count[1]*Count[1]  (2 as root)
      + Count[2]*Count[0]  (3 as root)

Therefore, we can get the equation:
Count[i] = ∑ Count[0...k] * [ k+1....i]     0<=k<i-1

*/
    public int numTrees(int n) {
        int[] count = new int[n+1];
        count[0] = 1;
        count[1] = 1;

        for(int i=2;  i<= n; i++){
            for(int j=0; j<i; j++){
                count[i] += count[j] * count[i - j - 1];
            }
        }
        return count[n];
    }


    //    95. Unique Binary Search Trees II

    /**
     * 95. Unique Binary Search Trees II
     *
     * Given an integer n, generate all structurally unique BST's
     * (binary search trees) that store values 1...n.
     For example,
     Given n = 3, your program should return all 5 unique BST's shown below.
     1         3     3      2      1
     \       /     /      / \      \
     3     2     1      1   3      2
     /     /       \                 \
     2     1         2                 3*/

    public List<TreeNode> generateTrees(int n) {

        return generateTrees(1, n);
    }

    private List<TreeNode> generateTrees(int start, int end) {
        List<TreeNode> result = new ArrayList();
        if (start > end) {
            result.add(null);
            return result;
        }
        if (start == end) {
            result.add(new TreeNode(start));
            return result;
        }

        for (int i = start; i <= end; i++) {
            List<TreeNode> leftList = generateTrees(start, i - 1);
            List<TreeNode> rightList = generateTrees(i + 1, end);
            for (TreeNode left : leftList) {
                for (TreeNode right : rightList) {
                    TreeNode root = new TreeNode(i);
                    root.left = left;
                    root.right = right;
                    result.add(root);
                }
            }
        }
        return result;
    }



    /**
     * 97. Interleaving String
     *
     * Given s1, s2, s3, find whether s3 is formed by the interleaving of s1 and s2.
     * For example,
     * Given:
     * s1 = "aabcc",
     * s2 = "dbbca",
     * When s3 = "aadbbcbcac", return true.
     * When s3 = "aadbbbaccc", return false.
     */
    Set<String> mem = new HashSet<>();
    public boolean isInterleave(String s1, String s2, String s3) {
        if(s1.length()+s2.length()!=s3.length())
            return false;
        if(s1.length() == 0 && s2.length() == 0 && s3.length() == 0)
            return true;
        if(mem.contains(s1+"#"+s2))
            return false;
        if(s3.length() > 0){
            if(s1.length() > 0 &&  s1.charAt(0) == s3.charAt(0))
                if(isInterleave(s1.substring(1),s2,s3.substring(1))) return true;
            if(s2.length() > 0 && s2.charAt(0) == s3.charAt(0))
                if(isInterleave(s1,s2.substring(1),s3.substring(1))) return true;
        }
        mem.add(s1+"#"+s2);
        return false;
    }
//    public boolean isInterleave(String s1, String s2, String s3) {
//        int m = s1.length();
//        int n = s2.length();
//        if (m + n != s3.length()) {
//            return false;
//        }
//
//        boolean[][] dp = new boolean[m + 1][n + 1];
//
//        dp[0][0] = true;
//
//        for (int i = 0; i < m; i++) {
//            if (s1.charAt(i) == s3.charAt(i)) {
//                dp[i + 1][0] = true;
//            } else {
//                //if one char fails, that means it breaks, the rest of the chars won't matter any more.
//                //Mian and I found one missing test case on Lintcode: ["b", "aabccc", "aabbbcb"]
//                //if we don't break, here, Lintcode could still accept this code, but Leetcode fails it.
//                break;
//            }
//        }
//
//        for (int j = 0; j < n; j++) {
//            if (s2.charAt(j) == s3.charAt(j)) {
//                dp[0][j + 1] = true;
//            } else {
//                break;
//            }
//        }
//
//        for (int i = 1; i <= m; i++) {
//            for (int j = 1; j <= n; j++) {
//                int k = i + j - 1;
//                dp[i][j] = (s1.charAt(i - 1) == s3.charAt(k) && dp[i - 1][j])
//                        || (s2.charAt(j - 1) == s3.charAt(k) && dp[i][j - 1]);
//            }
//        }
//
//        return dp[m][n];
//    }


//    98. Validate Binary Search Tree
//    Given a binary tree, determine if it is a valid binary search tree (BST).
//
//    Assume a BST is defined as follows:
//
//    The left subtree of a node contains only nodes with keys less than the node's key.
//    The right subtree of a node contains only nodes with keys greater than the node's key.
//    Both the left and right subtrees must also be binary search trees.
//    Example 1:
//            2
//            / \
//            1   3
//    Binary tree [2,1,3], return true.
//    Example 2:
//            1
//            / \
//            2   3
//    Binary tree [1,2,3], return false.


    public boolean isValidBST(TreeNode root) {

        return isValidBST(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }

    public boolean isValidBST(TreeNode root, long minVal, long maxVal) {
        if (root == null) return true;
        if (root.val >= maxVal || root.val <= minVal) return false;
        return isValidBST(root.left, minVal, root.val)
                && isValidBST(root.right, root.val, maxVal);
    }


//    99. Recover Binary Search Tree
//    Two elements of a binary search tree (BST) are swapped by mistake.
//
//    Recover the tree without changing its structure.

    private TreeNode firstElement = null;
    private TreeNode secondElement = null;
    private TreeNode prevElement = new TreeNode(Integer.MIN_VALUE);

    public void recoverTree(TreeNode root) {
        // traverse and get two elements
        traverse(root);
        // swap
        int temp = firstElement.val;
        firstElement.val = secondElement.val;
        secondElement.val = temp;
    }
    //inorder traverse.
    private void traverse(TreeNode root) {
        if (root == null) {
            return;
        }
        traverse(root.left);
        if (firstElement == null && root.val < prevElement.val) {
            firstElement = prevElement;
        }
        if (firstElement != null && root.val < prevElement.val) {
            secondElement = root;
        }
        prevElement = root;
        traverse(root.right);
    }





//    100. Same Tree
//    Given two binary trees, write a function to check if they are equal or not.
//
//    Two binary trees are considered equal if they are structurally identical
// and the nodes have the same value.

    public boolean isSameTree(TreeNode p, TreeNode q) {
        if(p==null && q==null){
            return true;
        }else if(p==null || q==null){
            return false;
        }

        if(p.val==q.val){
            return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
        }else{
            return false;
        }
    }
}
