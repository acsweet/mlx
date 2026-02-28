// Copyright © 2025 Apple Inc.
#pragma once

#include <cstdlib>
#include <new>

namespace mlx::core {

// 32-byte aligned buffer with grow-only reallocation (for thread_local reuse).
template <typename T>
class aligned_unique_ptr {
 private:
  T* ptr_;
  size_t size_;

 public:
  aligned_unique_ptr() : ptr_(nullptr), size_(0) {}

  explicit aligned_unique_ptr(size_t size) : size_(size) {
    ptr_ = static_cast<T*>(aligned_alloc(32, size * sizeof(T)));
    if (!ptr_)
      throw std::bad_alloc();
  }

  ~aligned_unique_ptr() {
    if (ptr_)
      free(ptr_);
  }

  aligned_unique_ptr(aligned_unique_ptr&& other) noexcept
      : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
  }

  aligned_unique_ptr& operator=(aligned_unique_ptr&& other) noexcept {
    if (this != &other) {
      if (ptr_)
        free(ptr_);
      ptr_ = other.ptr_;
      size_ = other.size_;
      other.ptr_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  aligned_unique_ptr(const aligned_unique_ptr&) = delete;
  aligned_unique_ptr& operator=(const aligned_unique_ptr&) = delete;

  T* get() const { return ptr_; }

  void reset(size_t new_size) {
    if (new_size > size_) {
      if (ptr_)
        free(ptr_);
      ptr_ = static_cast<T*>(aligned_alloc(32, new_size * sizeof(T)));
      if (!ptr_)
        throw std::bad_alloc();
      size_ = new_size;
    }
  }
};

} // namespace mlx::core
