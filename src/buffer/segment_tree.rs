//! Segment tree data structures for efficient prioritized experience replay.
//!
//! This module provides O(log n) operations for priority-based sampling:
//! - `SumTree`: Enables proportional sampling based on priorities.
//! - `MinTree`: Tracks minimum priority for importance sampling weights.
//!
//! These structures are used by the replay buffer to implement efficient PER.

/// Sum tree for efficient priority-based sampling.
///
/// A sum tree is a complete binary tree where each parent node holds the
/// sum of its children. This enables O(log n) sampling proportional to
/// priorities and O(log n) priority updates.
///
/// # Structure
///
/// ```text
///        [13]           <- root: sum of all priorities
///       /    \
///     [6]    [7]        <- internal nodes: partial sums
///    /  \   /  \
///   [1] [5][3] [4]      <- leaves: actual priorities
/// ```
///
/// # Example
///
/// ```ignore
/// let mut tree = SumTree::new(4);
/// tree.update(0, 1.0);
/// tree.update(1, 3.0);
/// tree.update(2, 2.0);
/// tree.update(3, 4.0);
///
/// // Sample proportionally to priorities
/// let (idx, priority) = tree.sample(5.0); // Returns index 2 or 3
/// ```
#[derive(Debug, Clone)]
pub struct SumTree {
    /// Number of leaf nodes (buffer capacity).
    capacity: usize,
    /// Tree array: internal nodes + leaf nodes.
    /// Size = 2 * capacity - 1 (for a complete binary tree).
    tree: Vec<f32>,
}

impl SumTree {
    /// Create a new sum tree with given capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Number of leaf nodes (must be > 0)
    ///
    /// # Panics
    ///
    /// Panics if capacity is 0.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Capacity must be positive");
        Self {
            capacity,
            tree: vec![0.0; 2 * capacity - 1],
        }
    }

    /// Update the priority at a leaf index.
    ///
    /// # Arguments
    ///
    /// * `idx` - Leaf index (0..capacity)
    /// * `priority` - New priority value (must be >= 0)
    ///
    /// # Complexity
    ///
    /// O(log n) where n is the capacity.
    #[inline]
    pub fn update(&mut self, idx: usize, priority: f32) {
        debug_assert!(idx < self.capacity, "Index out of bounds");
        debug_assert!(priority >= 0.0, "Priority must be non-negative");

        // Convert leaf index to tree array index
        let tree_idx = idx + self.capacity - 1;
        let change = priority - self.tree[tree_idx];
        self.tree[tree_idx] = priority;

        // Propagate change up to root
        let mut parent = tree_idx;
        while parent > 0 {
            parent = (parent - 1) / 2;
            self.tree[parent] += change;
        }
    }

    /// Sample a leaf index based on a value in [0, total).
    ///
    /// Returns the leaf index and its priority. The sampling is proportional
    /// to priorities: higher priority leaves are more likely to be sampled.
    ///
    /// # Arguments
    ///
    /// * `value` - Random value in [0, total())
    ///
    /// # Returns
    ///
    /// Tuple of (leaf_index, priority).
    ///
    /// # Complexity
    ///
    /// O(log n) where n is the capacity.
    #[inline]
    pub fn sample(&self, value: f32) -> (usize, f32) {
        debug_assert!(value >= 0.0 && value <= self.tree[0] + 1e-6);

        let mut idx = 0;
        let mut value = value;

        // Traverse down the tree
        while idx < self.capacity - 1 {
            let left = 2 * idx + 1;
            let right = left + 1;

            if value <= self.tree[left] {
                idx = left;
            } else {
                value -= self.tree[left];
                idx = right;
            }
        }

        // Convert tree index to leaf index
        let leaf_idx = idx - (self.capacity - 1);
        (leaf_idx, self.tree[idx])
    }

    /// Get the total sum of all priorities.
    ///
    /// # Returns
    ///
    /// Sum of all leaf priorities.
    #[inline]
    pub fn total(&self) -> f32 {
        self.tree[0]
    }

    /// Get the priority at a specific leaf index.
    ///
    /// # Arguments
    ///
    /// * `idx` - Leaf index (0..capacity)
    ///
    /// # Returns
    ///
    /// Priority value at the given index.
    #[inline]
    pub fn get(&self, idx: usize) -> f32 {
        debug_assert!(idx < self.capacity);
        self.tree[idx + self.capacity - 1]
    }

    /// Clear all priorities (set to zero).
    pub fn clear(&mut self) {
        self.tree.fill(0.0);
    }

    /// Get the capacity (number of leaf nodes).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Min tree for tracking minimum priority efficiently.
///
/// A min tree is a complete binary tree where each parent node holds the
/// minimum of its children. This enables O(log n) queries for the minimum
/// priority and O(log n) priority updates.
///
/// Used in PER to compute the maximum importance sampling weight.
#[derive(Debug, Clone)]
pub struct MinTree {
    /// Number of leaf nodes (buffer capacity).
    capacity: usize,
    /// Tree array: internal nodes + leaf nodes.
    tree: Vec<f32>,
}

impl MinTree {
    /// Create a new min tree with given capacity.
    ///
    /// All leaves are initialized to f32::MAX (no valid priorities yet).
    ///
    /// # Arguments
    ///
    /// * `capacity` - Number of leaf nodes (must be > 0)
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Capacity must be positive");
        Self {
            capacity,
            tree: vec![f32::MAX; 2 * capacity - 1],
        }
    }

    /// Update the priority at a leaf index.
    ///
    /// # Arguments
    ///
    /// * `idx` - Leaf index (0..capacity)
    /// * `priority` - New priority value
    ///
    /// # Complexity
    ///
    /// O(log n) where n is the capacity.
    #[inline]
    pub fn update(&mut self, idx: usize, priority: f32) {
        debug_assert!(idx < self.capacity, "Index out of bounds");

        let tree_idx = idx + self.capacity - 1;
        self.tree[tree_idx] = priority;

        // Propagate minimum up to root
        let mut parent = tree_idx;
        while parent > 0 {
            parent = (parent - 1) / 2;
            let left = 2 * parent + 1;
            let right = left + 1;
            self.tree[parent] = self.tree[left].min(self.tree[right]);
        }
    }

    /// Get the minimum priority across all leaves.
    ///
    /// # Returns
    ///
    /// Minimum priority value (f32::MAX if tree is empty).
    #[inline]
    pub fn min(&self) -> f32 {
        self.tree[0]
    }

    /// Get the priority at a specific leaf index.
    #[inline]
    pub fn get(&self, idx: usize) -> f32 {
        debug_assert!(idx < self.capacity);
        self.tree[idx + self.capacity - 1]
    }

    /// Clear all priorities (set to f32::MAX).
    pub fn clear(&mut self) {
        self.tree.fill(f32::MAX);
    }

    /// Get the capacity (number of leaf nodes).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_tree_basic() {
        let mut tree = SumTree::new(4);

        tree.update(0, 1.0);
        tree.update(1, 2.0);
        tree.update(2, 3.0);
        tree.update(3, 4.0);

        assert!((tree.total() - 10.0).abs() < 1e-6);
        assert!((tree.get(0) - 1.0).abs() < 1e-6);
        assert!((tree.get(3) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_sum_tree_sampling() {
        let mut tree = SumTree::new(4);

        tree.update(0, 1.0);
        tree.update(1, 2.0);
        tree.update(2, 3.0);
        tree.update(3, 4.0);

        // Value 0.5 should return index 0 (priority 1.0)
        let (idx, priority) = tree.sample(0.5);
        assert_eq!(idx, 0);
        assert!((priority - 1.0).abs() < 1e-6);

        // Value 2.5 should return index 1 (priority 2.0, cumsum = 3.0)
        let (idx, priority) = tree.sample(2.5);
        assert_eq!(idx, 1);
        assert!((priority - 2.0).abs() < 1e-6);

        // Value 5.0 should return index 2 (cumsum = 6.0)
        let (idx, priority) = tree.sample(5.0);
        assert_eq!(idx, 2);
        assert!((priority - 3.0).abs() < 1e-6);

        // Value 9.0 should return index 3
        let (idx, priority) = tree.sample(9.0);
        assert_eq!(idx, 3);
        assert!((priority - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_sum_tree_update() {
        let mut tree = SumTree::new(4);

        tree.update(0, 1.0);
        assert!((tree.total() - 1.0).abs() < 1e-6);

        tree.update(0, 5.0);
        assert!((tree.total() - 5.0).abs() < 1e-6);

        tree.update(1, 3.0);
        assert!((tree.total() - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_sum_tree_clear() {
        let mut tree = SumTree::new(4);
        tree.update(0, 1.0);
        tree.update(1, 2.0);

        tree.clear();
        assert!((tree.total()).abs() < 1e-6);
    }

    #[test]
    fn test_min_tree_basic() {
        let mut tree = MinTree::new(4);

        tree.update(0, 4.0);
        tree.update(1, 2.0);
        tree.update(2, 1.0);
        tree.update(3, 3.0);

        assert!((tree.min() - 1.0).abs() < 1e-6);
        assert!((tree.get(2) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_min_tree_update() {
        let mut tree = MinTree::new(4);

        tree.update(0, 5.0);
        assert!((tree.min() - 5.0).abs() < 1e-6);

        tree.update(1, 2.0);
        assert!((tree.min() - 2.0).abs() < 1e-6);

        tree.update(0, 1.0);
        assert!((tree.min() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_min_tree_clear() {
        let mut tree = MinTree::new(4);
        tree.update(0, 1.0);

        tree.clear();
        assert!(tree.min() == f32::MAX);
    }

    #[test]
    fn test_large_tree() {
        let capacity = 1024;
        let mut sum_tree = SumTree::new(capacity);
        let mut min_tree = MinTree::new(capacity);

        for i in 0..capacity {
            let priority = (i + 1) as f32;
            sum_tree.update(i, priority);
            min_tree.update(i, priority);
        }

        // Sum of 1..=1024 = 1024 * 1025 / 2 = 524800
        let expected_sum = (capacity * (capacity + 1) / 2) as f32;
        assert!((sum_tree.total() - expected_sum).abs() < 1.0);
        assert!((min_tree.min() - 1.0).abs() < 1e-6);
    }
}
