// Standalone verification of the two ankerl::unordered_dense behaviors that
// the YutrelUnityPlugin material-update path relies on / must defend against:
//
//  1) Move construction leaves the source map EMPTY (fix 1: removing the
//     destructive clear() before std::move is safe — the previous
//     material_revision baselines are preserved in the moved-to map while the
//     source can be re-populated by emplace).
//  2) Rehash (inserting more keys) invalidates pointers captured into the
//     map's flat value storage (fix 2: the registry must own MaterialTextureSet
//     through unique_ptr so `&set->images[i]` stays stable).
//
// Uses the exact header shipped in the repo.
#include <cstdio>
#include <array>
#include <cstdint>
#include <vector>
#include <memory>

#include <luisa/core/stl/unordered_dense.h>

struct MaterialTextureSetStub {
    std::array<int, 9u> images{};
};

// The repo's header omits default template args on the `map` alias; spell out
// the same instantiation luisa uses (flat dense map over a std::vector).
template <typename K, typename V>
using flat_map = ankerl::unordered_dense::detail::table<
    K, V,
    std::hash<K>, std::equal_to<>,
    std::allocator<std::pair<K, V>>,
    ankerl::unordered_dense::bucket_type::standard,
    std::vector<std::pair<K, V>>>;

int main() {
    // --- 1) move semantics ---
    {
        flat_map<uint64_t, int> src;
        src[1] = 100;
        src[2] = 200;
        auto previous_applied = std::move(src);
        bool source_empty = src.empty();
        bool baseline_preserved = previous_applied.at(1) == 100 && previous_applied.at(2) == 200;
        // Re-populate the source the way apply_scene_updates does.
        src.emplace(1, 1);
        src.emplace(2, 2);
        std::printf("[move] source empty after move: %d (expect 1)\n", source_empty ? 1 : 0);
        std::printf("[move] baseline preserved in moved-to map: %d (expect 1)\n", baseline_preserved ? 1 : 0);
        std::printf("[move] source re-populatable via emplace: %d (expect 1)\n",
                    (src.size() == 2 && src.at(1) == 1) ? 1 : 0);
    }

    // --- 2) reference invalidation on rehash ---
    {
        flat_map<uint64_t, MaterialTextureSetStub> registry;
        // Material A: capture slot pointers (as UnityMaterialParamTextureSpec does).
        auto &set_a = registry[1000u];
        auto *slot_a0 = &set_a.images[0u];
        auto *slot_a1 = &set_a.images[1u];
        *slot_a0 = 1;
        *slot_a1 = 2;
        // More materials inserted later (as create_scene does for the other
        // submeshes). With flat dense storage this rehashes and MOVES the
        // value for material A.
        for (auto i = 1u; i < 64u; i++) {
            registry[1000u + i] = MaterialTextureSetStub{};
        }
        // Simulate UnityMaterialParamTexture::build writing through the stale
        // captured pointer (as the pre-fix code did).
        *slot_a0 = 999;
        *slot_a1 = 888;
        auto &set_a_now = registry[1000u];
        bool pointer_survived =
            set_a_now.images[0u] == 999 && set_a_now.images[1u] == 888;
        std::printf("[rehash] captured pointer still valid after 63 more inserts: %d (expect 0 = dangling; this is why unique_ptr ownership is required)\n",
                    pointer_survived ? 1 : 0);

        // With unique_ptr ownership (the fix): the set object never moves.
        flat_map<uint64_t, std::unique_ptr<MaterialTextureSetStub>> fixed;
        auto &entry_a = fixed[1000u];
        if (!entry_a) { entry_a = std::make_unique<MaterialTextureSetStub>(); }
        auto *fixed_slot_a0 = &entry_a->images[0u];
        *fixed_slot_a0 = 1;
        for (auto i = 1u; i < 64u; i++) {
            auto &e = fixed[1000u + i];
            if (!e) { e = std::make_unique<MaterialTextureSetStub>(); }
        }
        *fixed_slot_a0 = 777;
        bool unique_ptr_survived = fixed[1000u]->images[0u] == 777;
        std::printf("[unique_ptr] slot pointer stable under rehash: %d (expect 1)\n",
                    unique_ptr_survived ? 1 : 0);
    }
    return 0;
}
