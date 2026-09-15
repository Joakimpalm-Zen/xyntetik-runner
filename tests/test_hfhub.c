// The decision half of `-hf owner/repo[:TAG]`: which file a spec names among
// a repository's files, and what the user is told when it cannot say. Pure,
// no network, so every rule is pinned here; the transfer half is gated by
// tests/test_hf_fetch.py against a fake hub.
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "hfhub.h"

static int pick(const char *const *names, int n, const char *tag, char *err, size_t cap) {
    err[0] = 0;
    return hf_select_file(names, n, tag, err, cap);
}

int main(void) {
    char err[512];

    // one GGUF, no tag: that one, and the card's README is not a candidate
    const char *one[] = { "README.md", "granite-4.1-3b-Q8_0.gguf", ".gitattributes" };
    assert(pick(one, 3, NULL, err, sizeof err) == 1);
    assert(pick(one, 3, "", err, sizeof err) == 1);

    // the tag is a whole token, case-insensitive; Q4_K is not Q4_K_M
    const char *ladder[] = { "m-Q4_K_M.gguf", "m-Q4_K_S.gguf", "m-Q8_0.gguf", "m-Q4_K_M.gguf.envelope.json" };
    assert(pick(ladder, 4, "Q4_K_M", err, sizeof err) == 0);
    assert(pick(ladder, 4, "q8_0", err, sizeof err) == 2);
    assert(pick(ladder, 4, "Q4_K", err, sizeof err) == -1);
    assert(strstr(err, "Q4_K") && strstr(err, "m-Q4_K_M.gguf"));   // the reason names what exists

    // several files and no tag: refuse, and list them so the user can choose
    assert(pick(ladder, 4, NULL, err, sizeof err) == -1);
    assert(strstr(err, "m-Q8_0.gguf") && strstr(err, ":TAG"));

    // a vision projector beside one model file is not a second candidate
    const char *vis[] = { "mmproj-model-f16.gguf", "gemma-4-E4B-it-Q4_K_M.gguf" };
    assert(pick(vis, 2, NULL, err, sizeof err) == 1);

    // a multipart file is one candidate, answered by its first part
    const char *parts[] = { "big-Q8_0-00002-of-00003.gguf", "big-Q8_0-00001-of-00003.gguf",
                            "big-Q8_0-00003-of-00003.gguf", "big-Q4_K_M.gguf" };
    assert(pick(parts, 4, "Q8_0", err, sizeof err) == 1);
    assert(pick(parts, 4, "Q4_K_M", err, sizeof err) == 3);
    unsigned count = 0; size_t plen = 0;
    assert(hf_is_first_part("big-Q8_0-00001-of-00003.gguf", &count, &plen));
    assert(count == 3 && plen == strlen("big-Q8_0"));
    assert(!hf_is_first_part("big-Q8_0-00002-of-00003.gguf", &count, &plen));
    assert(!hf_is_first_part("big-Q4_K_M.gguf", &count, &plen));

    // nothing loadable at all
    const char *none[] = { "README.md", "config.json" };
    assert(pick(none, 2, NULL, err, sizeof err) == -1);
    assert(strstr(err, "no GGUF"));

    // a file under a directory keeps its path but matches on the base name
    const char *nested[] = { "Q4_K_M/model-Q4_K_M.gguf", "Q8_0/model-Q8_0.gguf" };
    assert(pick(nested, 2, "q4_k_m", err, sizeof err) == 0);

    puts("hfhub tests ok");
    return 0;
}
