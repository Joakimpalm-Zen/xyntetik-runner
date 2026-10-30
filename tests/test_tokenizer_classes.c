// The code-point classes the BPE pre-tokenizers split on, and the pre-token
// boundaries that follow from them.
//
// Absolute anchors. The class expectations are transcribed from the Unicode
// 16.0 charts for Latin-1 Supplement (U+0080..U+00FF) and for Superscripts and
// Subscripts, Currency Symbols and Combining Diacritical Marks for Symbols
// (U+2070..U+20FF), written as ranges here independently of the generated
// table (src/unicode_classes.h). The boundary cases are the publishers' reference
// tokenizers' own pre-tokens (Hugging Face tokenizers over each family's
// tokenizer.json): "°)" is one pre-token, and one o200k token (Kvist-14B,
// gpt-oss), where the runner used to split it because it read every code
// point above U+007F as a letter.
#include "../src/tokenizer.c"

#include <assert.h>
#include <stdio.h>
#include <string.h>

enum { K_LETTER = 0, K_NUMBER = 1, K_OTHER = 2, K_SPACE = 3 };

static int expected_class(uint32_t c) {
    if (c == 0x85 || c == 0xA0) return K_SPACE;              // NEL, NO-BREAK SPACE
    if (c >= 0x80 && c <= 0x9F) return K_OTHER;               // C1 controls
    if (c >= 0xA1 && c <= 0xBF) {
        if (c == 0xAA || c == 0xB5 || c == 0xBA) return K_LETTER;   // ª µ º
        if (c == 0xB2 || c == 0xB3 || c == 0xB9 ||
            (c >= 0xBC && c <= 0xBE)) return K_NUMBER;               // ² ³ ¹ ¼ ½ ¾
        return K_OTHER;                                     // ¡ ¢ £ ... § © « ® ° ± ¶ · » ¿
    }
    if (c >= 0xC0 && c <= 0xFF) return (c == 0xD7 || c == 0xF7) ? K_OTHER : K_LETTER;
    if (c == 0x2070 || (c >= 0x2074 && c <= 0x2079) ||
        (c >= 0x2080 && c <= 0x2089)) return K_NUMBER;      // superscript and subscript digits
    if (c == 0x2071 || c == 0x207F || (c >= 0x2090 && c <= 0x209C))
        return K_LETTER;                                    // ⁱ ⁿ ₐ..ₜ (modifier letters)
    if (c >= 0x2070 && c <= 0x20FF) return K_OTHER;          // signs, currency, marks, unassigned
    return -1;
}

static void test_classes_match_unicode(void) {
    int checked = 0;
    for (uint32_t c = 0x80; c <= 0x20FF; c++) {
        if (c == 0x100) c = 0x2070;
        int want = expected_class(c);
        assert(want >= 0);
        int got = cp_class(c);
        if (got != want) {
            fprintf(stderr, "U+%04X: class %d, Unicode says %d\n", (unsigned)c, got, want);
            assert(0);
        }
        assert(cp_letter(c) == (want == K_LETTER));
        assert(cp_digit(c) == (want == K_NUMBER));
        assert(cp_space(c) == (want == K_SPACE));
        assert(cp_mark(c) == (c >= 0x20D0 && c <= 0x20F0));
        checked++;
    }
    assert(checked == 128 + 144);
    // the one cased letter in the A0 row, for the case-splitting families
    assert(cp_letter_lowerish(0xB5) && !cp_letter_upperish(0xB5));
    // beyond the two blocks, where the hand-written ranges read every code
    // point above U+007F as a letter unless a listed block said otherwise
    assert(cp_class(0xFF1A) == K_OTHER && cp_class(0xFF09) == K_OTHER);   // full-width : and )
    assert(cp_class(0xFEFF) == K_OTHER);                                  // ZERO WIDTH NO-BREAK SPACE
    assert(cp_class(0x0663) == K_NUMBER);                                 // ARABIC-INDIC DIGIT THREE
    assert(cp_mark(0x0483) && cp_class(0x0483) == K_OTHER);              // COMBINING CYRILLIC TITLO
    assert(cp_class(0x1680) == K_SPACE);                                  // OGHAM SPACE MARK
    assert(cp_class(0xE000) == K_OTHER);                                  // private use
    // case beyond Latin, Greek and Cyrillic
    assert(cp_letter_upperish(0x0531) && !cp_letter_lowerish(0x0531));    // ARMENIAN CAPITAL AYB
    assert(cp_letter_lowerish(0x0562) && !cp_letter_upperish(0x0562));    // ARMENIAN SMALL BEN
    assert(cp_letter_upperish(0x01C5) && !cp_letter_lowerish(0x01C5));    // titlecase DŽ
    assert(cp_letter_upperish(0x4E00) && cp_letter_lowerish(0x4E00));     // a caseless letter
    // the Specials block: nothing in it is a letter (U+FFFD REPLACEMENT CHARACTER)
    for (uint32_t c = 0xFFF0; c <= 0xFFFF; c++) assert(!cp_letter(c) && !cp_digit(c) && !cp_space(c));
}

typedef int (*split_fn)(const uint32_t *cp, int i, int ncp);

static int split_llama3(const uint32_t *cp, int i, int ncp) { return pre_split_next(cp, i, ncp, 3, false, false); }
static int split_qwen2(const uint32_t *cp, int i, int ncp)  { return pre_split_next(cp, i, ncp, 1, false, false); }
// MiniCPM5: the digit groups are split off before the Llama 3 regex runs
static int split_minicpm5(const uint32_t *cp, int i, int ncp) { return pre_split_next(cp, i, ncp, 3, false, true); }
static int split_spark25(const uint32_t *cp, int i, int ncp) { return spark25_split_next(cp, i, ncp); }

// "a|b|c" is the expected pre-token sequence for text "abc"
static void expect_split(const char *family, split_fn fn, const char *spec) {
    char text[64] = "";
    size_t tn = 0;
    for (const char *p = spec; *p; p++) if (*p != '|') text[tn++] = *p;
    text[tn] = 0;
    uint32_t cp[64];
    size_t off[65];
    int ncp = 0;
    for (size_t i = 0; i < tn; ) {
        int l = u8_len((uint8_t)text[i]);
        off[ncp] = i;
        cp[ncp++] = u8_decode(text + i, l);
        i += (size_t)l;
    }
    off[ncp] = tn;
    char got[160] = "";
    size_t gn = 0;
    for (int i = 0; i < ncp; ) {
        int end = fn(cp, i, ncp);
        if (end <= i) end = i + 1;
        if (i) got[gn++] = '|';
        memcpy(got + gn, text + off[i], off[end] - off[i]);
        gn += off[end] - off[i];
        i = end;
    }
    got[gn] = 0;
    if (strcmp(got, spec) != 0) {
        fprintf(stderr, "%s: \"%s\" split as \"%s\", the reference as \"%s\"\n",
                family, text, got, spec);
        assert(0);
    }
}

static void test_boundaries_match_the_references(void) {
    static const struct { const char *name; split_fn fn; } fams[] = {
        { "llama3", split_llama3 }, { "qwen2", split_qwen2 },
        { "o200k", llama4_split_next }, { "tekken", tekken_split_next },
        { "smollm", smollm_split_next },
    };
    for (size_t f = 0; f < sizeof fams / sizeof *fams; f++) {
        const char *n = fams[f].name;
        split_fn fn = fams[f].fn;
        expect_split(n, fn, "°)");          // the lab's case: one punctuation run
        expect_split(n, fn, "(°)");         // and with the bracket before it
        expect_split(n, fn, "€|5");         // a currency sign is not a letter
        expect_split(n, fn, "x|²");         // a superscript digit is \p{N}
        expect_split(n, fn, "café|)");      // Latin-1 letters still join their word
        expect_split(n, fn, "\xEF\xBF\xBD)");      // U+FFFD is a symbol, not a letter
    }
    // digits: \p{N}{1,3} for llama3 and o200k, one at a time for qwen2 and smollm
    expect_split("o200k", llama4_split_next, "(|25|°)");
    expect_split("qwen2", split_qwen2, "(|2|5|°)");
    expect_split("llama3", split_llama3, "²³¹");
    expect_split("o200k", llama4_split_next, "²³¹");
    expect_split("qwen2", split_qwen2, "²|³|¹");
    // the case-splitting families read µ as lowercase and A as uppercase
    expect_split("o200k", llama4_split_next, "µ|A");
    expect_split("tekken", tekken_split_next, "µ|A");
    expect_split("llama3", split_llama3, "µA");
    // the lab's second case: a punctuation run through U+FFFD ends before the
    // letters, so ".txt" after it is not one pre-token
    expect_split("o200k", llama4_split_next, " \"\xEF\xBF\xBD.|txt|\"");
    expect_split("llama3", split_llama3, " \"\xEF\xBF\xBD.|txt|\"");
    // a guillemet leads its word where the regex allows a leading non-letter
    expect_split("llama3", split_llama3, "«word|»");
    // a whitespace run that ends at a digit group: llama-bpe's single pass
    // leaves the last space to the digit, MiniCPM5's digits were cut out
    // first, so the run stays whole (openbmb/MiniCPM5-2B tokenizer.json:
    // "  1" is ["  ", "1"]); elsewhere the two rules agree
    expect_split("llama3", split_llama3, " | |1");
    expect_split("minicpm5", split_minicpm5, "  |1");
    expect_split("minicpm5", split_minicpm5, " \t |1");
    expect_split("minicpm5", split_minicpm5, "123|456|7");
    expect_split("minicpm5", split_minicpm5, " | x");
    // spark2_5 (XHToken tokenizer.json): every digit alone, a newline on its
    // own where afmoe joins the run, a punctuation run without trailing
    // newlines, ASCII punctuation leading ASCII letters, CJK runs apart
    expect_split("spark25", split_spark25, "1|2|3|4");
    expect_split("spark25", split_spark25, "a|\n|\n|b");
    expect_split("spark25", split_spark25, "!|\n");
    expect_split("spark25", split_spark25, "byte|-level");
    expect_split("spark25", split_spark25, "  |1");
    expect_split("spark25", split_spark25, "x|\xe4\xb8\xad\xe6\x96\x87|y");
    expect_split("o200k", llama4_split_next, "«word|»");
    expect_split("smollm", smollm_split_next, "«|word|»");
    // the classes beyond the two blocks (meta-llama/Llama-3.2, mistralai's
    // Mistral-Nemo and openai/gpt-oss tokenizer.json files)
    static const struct { const char *name; split_fn fn; } wide[] = {
        { "llama3", split_llama3 }, { "tekken", tekken_split_next }, { "o200k", llama4_split_next },
    };
    for (size_t f = 0; f < sizeof wide / sizeof *wide; f++) {
        expect_split(wide[f].name, wide[f].fn, "\xEF\xBC\x9A\xE2\x80\x9C");  // full-width colon, opening quote
        expect_split(wide[f].name, wide[f].fn, "\xEF\xBC\x89\n");             // full-width ) and a newline
        expect_split(wide[f].name, wide[f].fn, "\xEF\xBB\xBF//");              // U+FEFF before a comment
        expect_split(wide[f].name, wide[f].fn, "x|\xD9\xA3");                   // an Arabic-Indic digit is \p{N}
        expect_split(wide[f].name, wide[f].fn, "\xD4\xB1\xD5\xA2");           // Armenian capital, small
        expect_split(wide[f].name, wide[f].fn, "\xE6\x97\xA5\xE6\x9C\xAC|\xEF\xBC\x9A\xE2\x80\x9C");
    }
    expect_split("llama3", split_llama3, "a|\xD2\x83");                     // \p{L}+ ends at a mark
    expect_split("tekken", tekken_split_next, "a\xD2\x83");                 // tekken's classes hold \p{M}
    expect_split("tekken", tekken_split_next, "\xD5\xA1|\xD4\xB2");         // Armenian small, capital
    expect_split("o200k", llama4_split_next, "\xD5\xA1|\xD4\xB2");
    expect_split("llama3", split_llama3, "\xD5\xA1\xD4\xB2");
    expect_split("llama3", split_llama3, "x|\xD9\xA0\xD9\xA1\xD9\xA2|\xD9\xA3");
    expect_split("tekken", tekken_split_next, "x|\xD9\xA0|\xD9\xA1|\xD9\xA2|\xD9\xA3");
    // the case alternatives in the regex's order: U* gives back a caseless
    // letter so an L can follow (" 亚洲AV"), and a mark tries the run with
    // and without being the leading character (mistralai's Mistral-Nemo and
    // openai/gpt-oss tokenizer.json files)
    expect_split("o200k", llama4_split_next, " \xE4\xBA\x9A\xE6\xB4\xB2|AV");
    expect_split("o200k", llama4_split_next, "\xE4\xBA\x9A\xE6\xB4\xB2|AV");
    expect_split("o200k", llama4_split_next, "\xD2\x83|AV");
    expect_split("o200k", llama4_split_next, "\xD2\x83|!!");
    expect_split("o200k", llama4_split_next, "x\xD2\x83|AV");
    expect_split("o200k", llama4_split_next, "a|Bc");
    expect_split("o200k", llama4_split_next, "ABc|D");
    expect_split("tekken", tekken_split_next, " \xE4\xBA\x9A\xE6\xB4\xB2|AV");
    expect_split("tekken", tekken_split_next, "\xE4\xBA\x9A\xE6\xB4\xB2|AV");
    expect_split("tekken", tekken_split_next, "\xD2\x83|AV");
    expect_split("tekken", tekken_split_next, "\xD2\x83|!!");
    expect_split("tekken", tekken_split_next, "x\xD2\x83|AV");
    expect_split("tekken", tekken_split_next, "a|Bc");
    expect_split("tekken", tekken_split_next, "ABc|D");
    // afmoe (arcee-ai/Trinity-Nano-Preview): \p{Nd} groups by three, other
    // numbers are a piece of their own; a format character leads a letter run
    expect_split("afmoe", afmoe_split_next, "1|\xC2\xB2");
    expect_split("afmoe", afmoe_split_next, "12|\xC2\xB2|3");
    expect_split("afmoe", afmoe_split_next, "\xC2\xB2\xC2\xB3");
    expect_split("afmoe", afmoe_split_next, "1|234|567");
    expect_split("afmoe", afmoe_split_next, "\xEF\xBB\xBFusing");
    expect_split("afmoe", afmoe_split_next, "\xC2\xADt");
    expect_split("afmoe", afmoe_split_next, "a|\xE2\x80\x8C" "b");
    expect_split("afmoe", afmoe_split_next, "x| |\xC2\xBD|y");
    // spark2_5 (XHToken/Spark-X2.6-4B): the same lead rule
    expect_split("spark25", split_spark25, "\xE2\x80\x8C\xD9\x87\xD8\xA7\xDB\x8C");
    expect_split("spark25", split_spark25, "\xE1\xA0\x8E\xE1\xA0\xA0");
    expect_split("spark25", split_spark25, "1|\xC2\xB2");
    expect_split("spark25", split_spark25, "\xEF\xBB\xBFusing");
    expect_split("spark25", split_spark25, "a|,b");
    // a gap no alternative matches is one piece, as tokenizers' Isolated
    // split keeps it; a Thai run is its own group, ending where CJK starts
    expect_split("afmoe", afmoe_split_next, "\xE2\x80\x8B\xE2\x80\x8B");
    expect_split("afmoe", afmoe_split_next, "\b\b");
    expect_split("afmoe", afmoe_split_next, "\xE0\xB8\x97\xE0\xB8\xB5\xE0\xB9\x88");
    expect_split("afmoe", afmoe_split_next, "\xE0\xB8\x81\xE0\xB8\x82|\xE4\xB8\xAD");
    expect_split("afmoe", afmoe_split_next, "x|\xE2\x80\x8B\xE2\x80\x8B|!");
    expect_split("afmoe", afmoe_split_next, "\xD2\x83|!");
    expect_split("afmoe", afmoe_split_next, "\xD2\x83\xD2\x83|!");
    expect_split("spark25", split_spark25, "\xE2\x80\x8B\xE2\x80\x8B");
    expect_split("spark25", split_spark25, "\b\b");
    expect_split("spark25", split_spark25, "\xE0\xB8\x97\xE0\xB8\xB5\xE0\xB9\x88");
    expect_split("spark25", split_spark25, "\xE0\xB8\x81\xE0\xB8\x82|\xE4\xB8\xAD");
    expect_split("spark25", split_spark25, "x|\xE2\x80\x8B\xE2\x80\x8B|!");
    expect_split("spark25", split_spark25, "\xD2\x83|!");
    expect_split("spark25", split_spark25, "\xD2\x83\xD2\x83|!");
}

int main(void) {
    test_classes_match_unicode();
    test_boundaries_match_the_references();
    puts("tokenizer classes ok");
    return 0;
}
