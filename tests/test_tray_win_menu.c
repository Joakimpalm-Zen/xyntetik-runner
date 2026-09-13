// Windows only: the tray's popup menu carries the core's UTF-8 labels as
// UTF-16, and its icon is a 32-bit alpha bitmap of the rasterized ensö. The ANSI entry points read `Configure default runner…` and the
// `●` of the managed row in the console code page, and the menu showed
// each byte of those glyphs as its own character. The anchor is the wide
// literal the menu must read back, spelled by code point and independent
// of the conversion under test: GetMenuStringW returns what the user sees.
//
//     gcc -I src tests/test_tray_win_menu.c src/tray.c src/instances.c \
//         src/json.c src/compat.c -o test-tray-win-menu.exe -lgdi32 ...
#ifdef _WIN32
#include "../src/tray_win.c"

#include <stdio.h>
#include <string.h>
#include <wchar.h>

static int fails;

static void expect_row(HMENU m, int pos, const wchar_t *want, const char *what) {
    wchar_t got[512] = L"";
    int n = GetMenuStringW(m, (UINT)pos, got, 512, MF_BYPOSITION);
    if (n <= 0 || wcscmp(got, want) != 0) {
        fprintf(stderr, "FAIL: %s: row %d reads %d chars", what, pos, n);
        for (int i = 0; got[i]; i++) fprintf(stderr, " %04x", (unsigned)got[i]);
        fprintf(stderr, "\n");
        fails++;
    }
}

int main(void) {
    tray_item items[4];
    memset(items, 0, sizeof items);
    items[0].kind = TRAY_K_ACTION;
    snprintf(items[0].label, sizeof items[0].label, "Configure default runner\xe2\x80\xa6");
    items[1].kind = TRAY_K_LABEL;
    snprintf(items[1].label, sizeof items[1].label, "\xe2\x97\x8f serve  \xc2\xb7  :8080  \xc2\xb7  pid 42");
    items[2].kind = TRAY_K_SUB_BEGIN;   // the submenu title is row 1's label
    items[3].kind = TRAY_K_SUB_END;
    HMENU m = menu_from_items(items, 4);
    if (!m) { fprintf(stderr, "FAIL: no menu\n"); return 1; }
    expect_row(m, 0, L"Configure default runner\x2026", "the ellipsis is one character");
    expect_row(m, 1, L"\x25cf serve  \x00b7  :8080  \x00b7  pid 42",
               "the managed-instance mark and the middle dots survive as glyphs");
    // the submenu title was set through ModifyMenuW: still the same text
    MENUITEMINFOW info = { .cbSize = sizeof info, .fMask = MIIM_SUBMENU };
    if (!GetMenuItemInfoW(m, 1, TRUE, &info) || !info.hSubMenu) {
        fprintf(stderr, "FAIL: row 1 did not become a submenu\n");
        fails++;
    }
    DestroyMenu(m);

    // The icon: a 32-bit colour bitmap whose alpha channel is the glyph's
    // coverage, so the taskbar shows the ensö and nothing else. Version 1
    // handed the shell an opaque 16 px black square with the glyph on it.
    // Read the bitmap back through the same API the shell uses.
    for (int st = 0; st < 3; st++) {
        const int px = 24;
        HICON ic = grid_icon_px((tray_icon_state)st, px);
        if (!ic) { fprintf(stderr, "FAIL: no icon for state %d\n", st); fails++; continue; }
        ICONINFO info;
        if (!GetIconInfo(ic, &info) || !info.hbmColor) {
            fprintf(stderr, "FAIL: icon %d carries no colour bitmap\n", st); fails++; DestroyIcon(ic); continue;
        }
        BITMAP bm;
        GetObject(info.hbmColor, sizeof bm, &bm);
        if (bm.bmBitsPixel != 32 || bm.bmWidth != px || bm.bmHeight != px) {
            fprintf(stderr, "FAIL: icon %d colour bitmap is %d bpp %ldx%ld, want 32 bpp %dx%d\n",
                    st, (int)bm.bmBitsPixel, (long)bm.bmWidth, (long)bm.bmHeight, px, px);
            fails++;
        }
        BITMAPINFO bi;
        memset(&bi, 0, sizeof bi);
        bi.bmiHeader.biSize = sizeof bi.bmiHeader;
        bi.bmiHeader.biWidth = px; bi.bmiHeader.biHeight = -px;   // top-down
        bi.bmiHeader.biPlanes = 1; bi.bmiHeader.biBitCount = 32;
        bi.bmiHeader.biCompression = BI_RGB;
        unsigned char *bits = calloc((size_t)px * px * 4, 1);
        HDC dc = GetDC(NULL);
        int rows = GetDIBits(dc, info.hbmColor, 0, (UINT)px, bits, &bi, DIB_RGB_COLORS);
        ReleaseDC(NULL, dc);
        if (rows != px) { fprintf(stderr, "FAIL: icon %d bits unreadable (%d rows)\n", st, rows); fails++; }
        int corner = bits[3], solid = 0, partial = 0;
        for (int i = 0; i < px * px; i++) {
            int a = bits[(size_t)i * 4 + 3];
            if (a == 255) solid++;
            else if (a > 0) partial++;
        }
        if (corner != 0 || !solid || !partial) {
            fprintf(stderr, "FAIL: icon %d alpha: corner %d, %d solid, %d partial pixels\n",
                    st, corner, solid, partial);
            fails++;
        }
        free(bits);
        DeleteObject(info.hbmColor);
        if (info.hbmMask) DeleteObject(info.hbmMask);
        DestroyIcon(ic);
    }
    // the shell's small-icon size is what the tray renders at
    if (icon_px() < 16) { fprintf(stderr, "FAIL: icon size %d\n", icon_px()); fails++; }
    unsigned rgb = glyph_rgb();
    if (rgb != 0xFFFFFFu && rgb != 0x000000u) { fprintf(stderr, "FAIL: glyph colour %06x\n", rgb); fails++; }

    if (fails) return 1;
    printf("tray win menu: ok\n");
    printf("tray win icon: ok\n");
    return 0;
}
#else
#include <stdio.h>
int main(void) { printf("tray win menu: skipped (not Windows)\n"); return 0; }
#endif
