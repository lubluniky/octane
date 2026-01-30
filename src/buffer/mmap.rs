//! Memory-mapped replay buffer for large-scale experience storage.
//!
//! This module provides a replay buffer implementation that uses memory-mapped
//! files to store transitions, enabling buffers with 100M+ transitions without
//! RAM pressure. Data is stored on disk and loaded lazily when needed.
//!
//! # Features
//!
//! - **Massive capacity**: Store billions of transitions without RAM constraints.
//! - **Lazy loading**: Only loads batches when sampled.
//! - **Configurable cache**: LRU cache for frequently accessed transitions.
//! - **Persistence**: Buffer state survives process restarts.
//!
//! # Performance Considerations
//!
//! - Sequential writes are fast (append-only).
//! - Random reads may be slower than in-memory buffers.
//! - SSD storage is recommended for best performance.
//! - Cache size should be tuned based on batch size and sampling patterns.

use crate::buffer::ReplayBatch;
use crate::core::{Device, OctaneError, Result};
use candle_core::Tensor;
use memmap2::{MmapMut, MmapOptions};
use rand::prelude::*;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
// The flush() method on MmapMut comes from memmap2 directly
use std::path::{Path, PathBuf};

/// Configuration for the memory-mapped replay buffer.
#[derive(Debug, Clone)]
pub struct MmapBufferConfig {
    /// Maximum number of transitions.
    pub capacity: usize,
    /// Observation dimension.
    pub obs_dim: usize,
    /// Action dimension.
    pub action_dim: usize,
    /// Directory for memory-mapped files.
    pub storage_dir: PathBuf,
    /// Number of transitions to cache in memory (LRU).
    pub cache_size: usize,
    /// Prefix for file names.
    pub file_prefix: String,
}

impl Default for MmapBufferConfig {
    fn default() -> Self {
        Self {
            capacity: 10_000_000, // 10M transitions by default
            obs_dim: 4,
            action_dim: 1,
            storage_dir: PathBuf::from("/tmp/octane_buffer"),
            cache_size: 100_000, // Cache 100k transitions
            file_prefix: "replay".to_string(),
        }
    }
}

impl MmapBufferConfig {
    /// Create a new config with specified dimensions.
    pub fn new(obs_dim: usize, action_dim: usize) -> Self {
        Self {
            obs_dim,
            action_dim,
            ..Default::default()
        }
    }

    /// Set the buffer capacity.
    pub fn capacity(mut self, cap: usize) -> Self {
        self.capacity = cap;
        self
    }

    /// Set the storage directory.
    pub fn storage_dir<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.storage_dir = path.as_ref().to_path_buf();
        self
    }

    /// Set the cache size.
    pub fn cache_size(mut self, size: usize) -> Self {
        self.cache_size = size;
        self
    }

    /// Set the file prefix.
    pub fn file_prefix(mut self, prefix: &str) -> Self {
        self.file_prefix = prefix.to_string();
        self
    }
}

/// Size of a single transition in bytes.
fn transition_size(obs_dim: usize, action_dim: usize) -> usize {
    // obs + action + reward + next_obs + done
    // All stored as f32 (4 bytes each)
    (obs_dim + action_dim + 1 + obs_dim + 1) * std::mem::size_of::<f32>()
}

/// Memory-mapped replay buffer for large-scale storage.
///
/// Uses memory-mapped files to store transitions on disk, enabling buffers
/// with hundreds of millions of transitions without exhausting RAM.
///
/// # Example
///
/// ```ignore
/// use octane::buffer::{MmapReplayBuffer, MmapBufferConfig};
///
/// let config = MmapBufferConfig::new(84 * 84 * 4, 4)  // Atari-sized
///     .capacity(100_000_000)  // 100M transitions
///     .storage_dir("/data/replay")
///     .cache_size(1_000_000);  // 1M in-memory cache
///
/// let mut buffer = MmapReplayBuffer::new(config, Device::Cpu)?;
///
/// // Use like a normal replay buffer
/// buffer.add(&obs, &action, reward, &next_obs, done)?;
/// let batch = buffer.sample(256)?;
/// ```
pub struct MmapReplayBuffer {
    /// Configuration.
    config: MmapBufferConfig,
    /// Device for tensor creation.
    device: Device,
    /// Memory-mapped file for transitions.
    mmap: Option<MmapMut>,
    /// File handle.
    file: File,
    /// File path.
    file_path: PathBuf,
    /// Current write position (ring buffer index).
    position: usize,
    /// Number of valid transitions.
    size: usize,
    /// Transition size in bytes.
    transition_bytes: usize,
    /// LRU cache for recently accessed transitions.
    cache: HashMap<usize, CachedTransition>,
    /// LRU order (most recent at front).
    lru_order: Vec<usize>,
    /// Random number generator.
    rng: StdRng,
}

/// A cached transition.
#[derive(Debug, Clone)]
struct CachedTransition {
    obs: Vec<f32>,
    action: Vec<f32>,
    reward: f32,
    next_obs: Vec<f32>,
    done: f32,
}

impl MmapReplayBuffer {
    /// Create a new memory-mapped replay buffer.
    ///
    /// # Arguments
    ///
    /// * `config` - Buffer configuration
    /// * `device` - Device for tensor creation
    ///
    /// # Returns
    ///
    /// A new `MmapReplayBuffer` ready for use.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage directory cannot be created or
    /// the memory-mapped file cannot be opened.
    pub fn new(config: MmapBufferConfig, device: Device) -> Result<Self> {
        // Create storage directory if it doesn't exist
        std::fs::create_dir_all(&config.storage_dir)?;

        let transition_bytes = transition_size(config.obs_dim, config.action_dim);
        let file_size = config.capacity * transition_bytes;

        let file_path = config
            .storage_dir
            .join(format!("{}.bin", config.file_prefix));

        // Open or create the file
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&file_path)?;

        // Set file size
        file.set_len(file_size as u64)?;

        // Create memory map
        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };

        let cache_size = config.cache_size;

        Ok(Self {
            config,
            device,
            mmap: Some(mmap),
            file,
            file_path,
            position: 0,
            size: 0,
            transition_bytes,
            cache: HashMap::with_capacity(cache_size),
            lru_order: Vec::with_capacity(cache_size),
            rng: StdRng::from_entropy(),
        })
    }

    /// Open an existing memory-mapped buffer.
    ///
    /// # Arguments
    ///
    /// * `config` - Buffer configuration
    /// * `device` - Device for tensor creation
    /// * `size` - Number of valid transitions already stored
    /// * `position` - Current write position
    ///
    /// # Returns
    ///
    /// The opened buffer with existing data.
    pub fn open(
        config: MmapBufferConfig,
        device: Device,
        size: usize,
        position: usize,
    ) -> Result<Self> {
        let mut buffer = Self::new(config, device)?;
        buffer.size = size;
        buffer.position = position;
        Ok(buffer)
    }

    /// Add a transition to the buffer.
    ///
    /// # Arguments
    ///
    /// * `obs` - Current observation
    /// * `action` - Action taken
    /// * `reward` - Reward received
    /// * `next_obs` - Next observation
    /// * `done` - Whether episode terminated
    pub fn add(
        &mut self,
        obs: &[f32],
        action: &[f32],
        reward: f32,
        next_obs: &[f32],
        done: bool,
    ) -> Result<()> {
        let mmap = self
            .mmap
            .as_mut()
            .ok_or_else(|| OctaneError::Buffer("Memory map not available".to_string()))?;

        let offset = self.position * self.transition_bytes;
        let obs_dim = self.config.obs_dim;
        let action_dim = self.config.action_dim;

        // Write transition to memory-mapped file
        let mut cursor = offset;

        // Write observation
        for &val in obs.iter().take(obs_dim) {
            mmap[cursor..cursor + 4].copy_from_slice(&val.to_le_bytes());
            cursor += 4;
        }

        // Write action
        for &val in action.iter().take(action_dim) {
            mmap[cursor..cursor + 4].copy_from_slice(&val.to_le_bytes());
            cursor += 4;
        }

        // Write reward
        mmap[cursor..cursor + 4].copy_from_slice(&reward.to_le_bytes());
        cursor += 4;

        // Write next observation
        for &val in next_obs.iter().take(obs_dim) {
            mmap[cursor..cursor + 4].copy_from_slice(&val.to_le_bytes());
            cursor += 4;
        }

        // Write done flag
        let done_val = if done { 1.0f32 } else { 0.0f32 };
        mmap[cursor..cursor + 4].copy_from_slice(&done_val.to_le_bytes());

        // Invalidate cache entry if it exists
        if self.cache.contains_key(&self.position) {
            self.cache.remove(&self.position);
            self.lru_order.retain(|&x| x != self.position);
        }

        // Update position and size
        self.position = (self.position + 1) % self.config.capacity;
        self.size = (self.size + 1).min(self.config.capacity);

        Ok(())
    }

    /// Add a transition using Tensor inputs.
    pub fn add_tensor(
        &mut self,
        obs: &Tensor,
        action: &Tensor,
        reward: f32,
        next_obs: &Tensor,
        done: bool,
    ) -> Result<()> {
        let obs_vec: Vec<f32> = obs.flatten_all()?.to_vec1()?;
        let action_vec: Vec<f32> = action.flatten_all()?.to_vec1()?;
        let next_obs_vec: Vec<f32> = next_obs.flatten_all()?.to_vec1()?;

        self.add(&obs_vec, &action_vec, reward, &next_obs_vec, done)
    }

    /// Read a transition from the memory-mapped file.
    fn read_transition(&self, idx: usize) -> Result<CachedTransition> {
        let mmap = self
            .mmap
            .as_ref()
            .ok_or_else(|| OctaneError::Buffer("Memory map not available".to_string()))?;

        let offset = idx * self.transition_bytes;
        let obs_dim = self.config.obs_dim;
        let action_dim = self.config.action_dim;

        let mut cursor = offset;
        let mut obs = Vec::with_capacity(obs_dim);
        let mut action = Vec::with_capacity(action_dim);
        let mut next_obs = Vec::with_capacity(obs_dim);

        // Read observation
        for _ in 0..obs_dim {
            let bytes: [u8; 4] = mmap[cursor..cursor + 4].try_into().unwrap();
            obs.push(f32::from_le_bytes(bytes));
            cursor += 4;
        }

        // Read action
        for _ in 0..action_dim {
            let bytes: [u8; 4] = mmap[cursor..cursor + 4].try_into().unwrap();
            action.push(f32::from_le_bytes(bytes));
            cursor += 4;
        }

        // Read reward
        let reward_bytes: [u8; 4] = mmap[cursor..cursor + 4].try_into().unwrap();
        let reward = f32::from_le_bytes(reward_bytes);
        cursor += 4;

        // Read next observation
        for _ in 0..obs_dim {
            let bytes: [u8; 4] = mmap[cursor..cursor + 4].try_into().unwrap();
            next_obs.push(f32::from_le_bytes(bytes));
            cursor += 4;
        }

        // Read done
        let done_bytes: [u8; 4] = mmap[cursor..cursor + 4].try_into().unwrap();
        let done = f32::from_le_bytes(done_bytes);

        Ok(CachedTransition {
            obs,
            action,
            reward,
            next_obs,
            done,
        })
    }

    /// Get a transition, using cache if available.
    fn get_transition(&mut self, idx: usize) -> Result<CachedTransition> {
        // Check cache first
        if let Some(cached) = self.cache.get(&idx) {
            // Update LRU order
            self.lru_order.retain(|&x| x != idx);
            self.lru_order.insert(0, idx);
            return Ok(cached.clone());
        }

        // Read from disk
        let transition = self.read_transition(idx)?;

        // Add to cache
        if self.cache.len() >= self.config.cache_size {
            // Evict least recently used
            if let Some(lru_idx) = self.lru_order.pop() {
                self.cache.remove(&lru_idx);
            }
        }

        self.cache.insert(idx, transition.clone());
        self.lru_order.insert(0, idx);

        Ok(transition)
    }

    /// Sample a batch of transitions uniformly at random.
    pub fn sample(&mut self, batch_size: usize) -> Result<ReplayBatch> {
        if self.size < batch_size {
            return Err(OctaneError::Buffer(format!(
                "Not enough samples: {} < {}",
                self.size, batch_size
            )));
        }

        let indices: Vec<usize> = (0..batch_size)
            .map(|_| self.rng.gen_range(0..self.size))
            .collect();

        self.get_batch(&indices)
    }

    /// Get a batch from specific indices.
    fn get_batch(&mut self, indices: &[usize]) -> Result<ReplayBatch> {
        let batch_size = indices.len();
        let obs_dim = self.config.obs_dim;
        let action_dim = self.config.action_dim;

        let mut obs_batch = Vec::with_capacity(batch_size * obs_dim);
        let mut action_batch = Vec::with_capacity(batch_size * action_dim);
        let mut reward_batch = Vec::with_capacity(batch_size);
        let mut next_obs_batch = Vec::with_capacity(batch_size * obs_dim);
        let mut done_batch = Vec::with_capacity(batch_size);

        for &idx in indices {
            let t = self.get_transition(idx)?;
            obs_batch.extend_from_slice(&t.obs);
            action_batch.extend_from_slice(&t.action);
            reward_batch.push(t.reward);
            next_obs_batch.extend_from_slice(&t.next_obs);
            done_batch.push(t.done);
        }

        let candle_device = self.device.to_candle()?;

        Ok(ReplayBatch {
            observations: Tensor::from_slice(&obs_batch, (batch_size, obs_dim), &candle_device)?,
            actions: Tensor::from_slice(&action_batch, (batch_size, action_dim), &candle_device)?,
            rewards: Tensor::from_slice(&reward_batch, (batch_size,), &candle_device)?,
            next_observations: Tensor::from_slice(
                &next_obs_batch,
                (batch_size, obs_dim),
                &candle_device,
            )?,
            dones: Tensor::from_slice(&done_batch, (batch_size,), &candle_device)?,
            indices: indices.to_vec(),
            weights: None,
        })
    }

    /// Flush changes to disk.
    pub fn flush(&mut self) -> Result<()> {
        if let Some(ref mmap) = self.mmap {
            mmap.flush()?;
        }
        Ok(())
    }

    /// Get number of stored transitions.
    #[inline]
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Get buffer capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.config.capacity
    }

    /// Check if buffer can provide a batch of given size.
    #[inline]
    pub fn can_sample(&self, batch_size: usize) -> bool {
        self.size >= batch_size
    }

    /// Get current write position.
    #[inline]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Get cache size.
    #[inline]
    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }

    /// Get the file path.
    pub fn file_path(&self) -> &Path {
        &self.file_path
    }

    /// Clear the buffer.
    ///
    /// This resets the position and size but does not zero the file.
    pub fn clear(&mut self) {
        self.position = 0;
        self.size = 0;
        self.cache.clear();
        self.lru_order.clear();
    }

    /// Set random seed for reproducibility.
    pub fn seed(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
    }

    /// Get statistics about the buffer.
    pub fn stats(&self) -> MmapBufferStats {
        MmapBufferStats {
            size: self.size,
            capacity: self.config.capacity,
            position: self.position,
            cache_size: self.cache.len(),
            max_cache_size: self.config.cache_size,
            file_size_bytes: self.config.capacity * self.transition_bytes,
            transition_size_bytes: self.transition_bytes,
        }
    }
}

impl Drop for MmapReplayBuffer {
    fn drop(&mut self) {
        // Flush before dropping
        let _ = self.flush();
    }
}

/// Statistics about the memory-mapped buffer.
#[derive(Debug, Clone)]
pub struct MmapBufferStats {
    /// Number of stored transitions.
    pub size: usize,
    /// Maximum capacity.
    pub capacity: usize,
    /// Current write position.
    pub position: usize,
    /// Current cache size.
    pub cache_size: usize,
    /// Maximum cache size.
    pub max_cache_size: usize,
    /// Total file size in bytes.
    pub file_size_bytes: usize,
    /// Size of each transition in bytes.
    pub transition_size_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_config(dir: &Path) -> MmapBufferConfig {
        MmapBufferConfig::new(4, 2)
            .capacity(100)
            .storage_dir(dir)
            .cache_size(10)
    }

    #[test]
    fn test_mmap_buffer_basic() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config = make_config(temp_dir.path());
        let buffer = MmapReplayBuffer::new(config, Device::Cpu)?;

        assert!(buffer.is_empty());
        assert_eq!(buffer.capacity(), 100);

        Ok(())
    }

    #[test]
    fn test_mmap_add_and_sample() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config = make_config(temp_dir.path());
        let mut buffer = MmapReplayBuffer::new(config, Device::Cpu)?;

        // Add transitions
        for i in 0..50 {
            let obs = vec![i as f32; 4];
            let action = vec![0.0, 1.0];
            let next_obs = vec![(i + 1) as f32; 4];
            buffer.add(&obs, &action, i as f32 * 0.1, &next_obs, i % 10 == 9)?;
        }

        assert_eq!(buffer.len(), 50);
        assert!(buffer.can_sample(32));

        // Sample
        let batch = buffer.sample(32)?;
        assert_eq!(batch.observations.dims(), &[32, 4]);
        assert_eq!(batch.actions.dims(), &[32, 2]);

        Ok(())
    }

    #[test]
    fn test_mmap_overflow() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config = MmapBufferConfig::new(2, 1)
            .capacity(10)
            .storage_dir(temp_dir.path());
        let mut buffer = MmapReplayBuffer::new(config, Device::Cpu)?;

        // Add more than capacity
        for i in 0..25 {
            buffer.add(&[i as f32, i as f32], &[0.0], 1.0, &[0.0, 0.0], false)?;
        }

        assert_eq!(buffer.len(), 10); // Capped at capacity
        assert_eq!(buffer.position(), 5); // 25 % 10 = 5

        Ok(())
    }

    #[test]
    fn test_mmap_cache() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config = MmapBufferConfig::new(4, 2)
            .capacity(100)
            .storage_dir(temp_dir.path())
            .cache_size(5);
        let mut buffer = MmapReplayBuffer::new(config, Device::Cpu)?;

        // Add transitions
        for i in 0..20 {
            let obs = vec![i as f32; 4];
            buffer.add(&obs, &[0.0, 1.0], 1.0, &obs, false)?;
        }

        // Sample to populate cache
        for _ in 0..10 {
            let _ = buffer.sample(5)?;
        }

        // Cache should not exceed max size
        assert!(buffer.cache_len() <= 5);

        Ok(())
    }

    #[test]
    fn test_mmap_flush() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config = make_config(temp_dir.path());
        let mut buffer = MmapReplayBuffer::new(config, Device::Cpu)?;

        buffer.add(&[1.0; 4], &[0.0, 1.0], 1.0, &[2.0; 4], false)?;
        buffer.flush()?;

        Ok(())
    }

    #[test]
    fn test_mmap_stats() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config = make_config(temp_dir.path());
        let mut buffer = MmapReplayBuffer::new(config, Device::Cpu)?;

        for i in 0..30 {
            buffer.add(&[i as f32; 4], &[0.0, 1.0], 1.0, &[0.0; 4], false)?;
        }

        let stats = buffer.stats();
        assert_eq!(stats.size, 30);
        assert_eq!(stats.capacity, 100);
        assert_eq!(stats.position, 30);
        assert_eq!(stats.max_cache_size, 10);

        // Transition size: (4 + 2 + 1 + 4 + 1) * 4 = 48 bytes
        assert_eq!(stats.transition_size_bytes, 48);

        Ok(())
    }

    #[test]
    fn test_mmap_clear() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config = make_config(temp_dir.path());
        let mut buffer = MmapReplayBuffer::new(config, Device::Cpu)?;

        for i in 0..20 {
            buffer.add(&[i as f32; 4], &[0.0, 1.0], 1.0, &[0.0; 4], false)?;
        }

        buffer.clear();
        assert!(buffer.is_empty());
        assert_eq!(buffer.position(), 0);
        assert_eq!(buffer.cache_len(), 0);

        Ok(())
    }

    #[test]
    fn test_mmap_persistence() -> Result<()> {
        let temp_dir = TempDir::new()?;

        // Write data
        {
            let config = make_config(temp_dir.path());
            let mut buffer = MmapReplayBuffer::new(config, Device::Cpu)?;

            for i in 0..10 {
                buffer.add(&[i as f32; 4], &[0.0, 1.0], i as f32, &[0.0; 4], false)?;
            }
            buffer.flush()?;
        }

        // Reopen and verify
        {
            let config = make_config(temp_dir.path());
            let mut buffer = MmapReplayBuffer::open(config, Device::Cpu, 10, 10)?;

            assert_eq!(buffer.len(), 10);

            // Read back the data
            let t = buffer.get_transition(5)?;
            assert!((t.obs[0] - 5.0).abs() < 1e-6);
            assert!((t.reward - 5.0).abs() < 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_mmap_tensor_add() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config = make_config(temp_dir.path());
        let mut buffer = MmapReplayBuffer::new(config, Device::Cpu)?;

        let candle_device = Device::Cpu.to_candle()?;
        let obs = Tensor::ones((4,), candle_core::DType::F32, &candle_device)?;
        let action = Tensor::zeros((2,), candle_core::DType::F32, &candle_device)?;
        let next_obs = Tensor::ones((4,), candle_core::DType::F32, &candle_device)?;

        buffer.add_tensor(&obs, &action, 1.0, &next_obs, false)?;
        assert_eq!(buffer.len(), 1);

        Ok(())
    }
}
