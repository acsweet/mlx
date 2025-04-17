// Copyright © 2025 Apple Inc.
#pragma once

#include <cstdlib>
#include <memory>
#include <new>
#include <type_traits>

namespace mlx::core {

// Alignment constant for AVX (256-bit / 32-byte alignment)
constexpr size_t AVX_ALIGNMENT = 32;

// Aligned allocator for use with STL containers like std::vector
template <typename T, std::size_t Alignment = AVX_ALIGNMENT>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    // Required for compatibility with std::allocator
    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept = default;
    
    // Enable conversion between compatible allocator types
    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    // Allocate aligned memory
    pointer allocate(size_type n) {
        if (n == 0) {
            return nullptr;
        }
        
        if (n > std::numeric_limits<size_type>::max() / sizeof(T)) {
            throw std::bad_alloc();
        }
        
        void* ptr = std::aligned_alloc(Alignment, n * sizeof(T));
        if (ptr == nullptr) {
            throw std::bad_alloc();
        }
        
        return static_cast<pointer>(ptr);
    }

    // Deallocate aligned memory
    void deallocate(pointer p, size_type) noexcept {
        std::free(p);
    }
    
    // Required for C++20 allocator completeness
    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }
    
    template <typename U>
    void destroy(U* p) {
        p->~U();
    }
    
    // Check if two allocators are equivalent (they always are for this implementation)
    bool operator==(const AlignedAllocator&) const noexcept {
        return true;
    }
    
    bool operator!=(const AlignedAllocator&) const noexcept {
        return false;
    }
};

} // namespace mlx::core