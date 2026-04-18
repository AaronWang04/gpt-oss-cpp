#include "kv_cache.h"

KVCache::KVCache(std::size_t num_layers) : k_cache(num_layers), v_cache(num_layers) {}

void KVCache::append(std::size_t layer,
                     std::span<const float> k_new,
                     std::span<const float> v_new) {
    k_cache[layer].insert(k_cache[layer].end(), k_new.begin(), k_new.end());
    v_cache[layer].insert(v_cache[layer].end(), v_new.begin(), v_new.end());
}
