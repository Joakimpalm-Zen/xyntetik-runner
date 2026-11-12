#include "metal_admission.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

int main(void) {
    const uint64_t GB = 1000000000ull;
    metal_weight_limits reported_16gb_mac = {
        .working_set = 127ull * GB / 10,
        .max_buffer = 95ull * GB / 10,
    };
    uint64_t budget = metal_full_weight_budget(reported_16gb_mac, GB / 2);

    // Several zero-copy buffers can hold this 10.3 GB file. It is below the
    // working-set budget but above one MTLBuffer, so selecting a layer split
    // here would preserve the exact ceiling the multi-buffer path removed.
    uint64_t model = 103ull * GB / 10;
    assert(model > reported_16gb_mac.max_buffer);
    assert(model <= budget);

    // Multi-buffer wrapping changes no residency arithmetic: a file beyond
    // the working-set budget must still take the partial/refusal path.
    assert(12ull * GB > budget);

    // R6.68.88: the score scratch is bounded. A default context keeps the
    // whole batch in one attention dispatch, as before; a longer one holds
    // 512 MiB of scores, and never fewer than 64 columns.
    const uint64_t MiB = 1024ull * 1024;
    assert(metal_att_cols(512, 32, 4096, METAL_ATT_BYTES) == 512);
    assert(metal_att_cols(512, 32, 8192, METAL_ATT_BYTES) == 512);
    assert(metal_att_cols(512, 32, 32768, METAL_ATT_BYTES) == 128);
    assert(metal_att_cols(512, 32, 131072, METAL_ATT_BYTES) == 64);
    assert(metal_att_cols(512, 64, 131072, METAL_ATT_BYTES) == 64);
    assert(metal_att_cols(8, 32, 4096, METAL_ATT_BYTES) == 8);     // a narrow batch: its own columns
    assert(metal_att_cols(150, 4, 1024, 1 * MiB) == 64);           // whole 8-column tiles
    assert(metal_att_cols(150, 4, 1024, 1 * MiB + 200000) == 72);
    // and the load's budget counts what a long context's scores take past
    // the 256 MiB the flat allowance always carried: nothing at a default
    // context, 256 MiB at 32K, 768 MiB at 128K (64 columns are 1 GiB there)
    assert(metal_other_bytes(7 * MiB, 512, 32, 4096) == (7 + 512) * MiB);
    assert(metal_other_bytes(0, 512, 32, 32768) == (256 + 512) * MiB);
    assert(metal_other_bytes(0, 512, 32, 131072) == (768 + 512) * MiB);

    puts("metal admission policy ok");
    return 0;
}
