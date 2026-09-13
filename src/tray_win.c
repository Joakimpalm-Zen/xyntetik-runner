// Windows tray backend: Shell_NotifyIcon + a hidden message-only window.
// The runner stays a console-subsystem binary (no -mwindows — that would
// break every CLI mode); when launched with --tray from Explorer the
// console is released with FreeConsole so no black window lingers.
#ifdef _WIN32
#include "tray.h"

#include <windows.h>
#include <shellapi.h>
#include <commdlg.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TRAY_MAX_ITEMS 128
#define WM_TRAY_CALLBACK (WM_APP + 1)
#define TRAY_ICON_ID 1
#define MENU_ID_BASE 1000

static NOTIFYICONDATAA g_nid;
static HWND g_hwnd;
static tray_item g_items[TRAY_MAX_ITEMS];
static int g_nitems;

// ---------------------------------------------------------------- the icon
// The ensö, rasterized by the core (tray_glyph_render: the macOS drawing's
// geometry, coverage-antialiased, premultiplied BGRA on a transparent
// background) into a 32-bit DIB with an alpha channel, at the size the
// shell asks for. Version 1 painted a 16 px white-on-black square with GDI
// strokes: on a light taskbar the square showed, on any display above 100%
// the shell scaled the 16 px up, and the strokes had no antialiasing at all.
//
// White on the dark taskbar, black on the light one: Windows has no template
// image the shell recolours, so the colour is read from the personalization
// key the shell itself reads, on every refresh (a theme change while the
// tray runs is picked up within a badge tick).
static unsigned glyph_rgb(void) {
    HKEY k;
    DWORD light = 0, n = sizeof light, type = 0;
    if (RegOpenKeyExA(HKEY_CURRENT_USER,
                      "Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize",
                      0, KEY_READ, &k) == ERROR_SUCCESS) {
        if (RegQueryValueExA(k, "SystemUsesLightTheme", NULL, &type,
                             (BYTE *)&light, &n) != ERROR_SUCCESS || type != REG_DWORD)
            light = 0;
        RegCloseKey(k);
    }
    return light ? 0x000000u : 0xFFFFFFu;
}

// The notification area's small icon size for this process's DPI: 16 px at
// 100%, 20 at 125%, 24 at 150%, 32 at 200%. Meaningful only once the process
// is DPI aware (tray_platform_run sets that before creating its window);
// an unaware process is told 16 and the shell stretches it.
static int icon_px(void) {
    int px = GetSystemMetrics(SM_CXSMICON);
    return px >= 8 ? px : 16;
}

// A 32-bit top-down DIB the glyph is rendered into; the caller owns it.
static HBITMAP glyph_dib(tray_icon_state st, int px, unsigned rgb, void **bits_out) {
    BITMAPV5HEADER bi;
    memset(&bi, 0, sizeof bi);
    bi.bV5Size = sizeof bi;
    bi.bV5Width = px;
    bi.bV5Height = -px;                 // top-down, the raster's row order
    bi.bV5Planes = 1;
    bi.bV5BitCount = 32;
    bi.bV5Compression = BI_BITFIELDS;
    bi.bV5RedMask = 0x00FF0000; bi.bV5GreenMask = 0x0000FF00;
    bi.bV5BlueMask = 0x000000FF; bi.bV5AlphaMask = 0xFF000000;
    void *bits = NULL;
    HBITMAP bm = CreateDIBSection(NULL, (BITMAPINFO *)&bi, DIB_RGB_COLORS, &bits, NULL, 0);
    if (!bm || !bits) { if (bm) DeleteObject(bm); return NULL; }
    if (!tray_glyph_render(st, px, rgb, (unsigned char *)bits)) { DeleteObject(bm); return NULL; }
    if (bits_out) *bits_out = bits;
    return bm;
}

static HICON grid_icon_px(tray_icon_state st, int px) {
    HBITMAP color = glyph_dib(st, px, glyph_rgb(), NULL);
    if (!color) return NULL;
    // The AND mask is all zeros (nothing masked): with a 32-bit colour
    // bitmap the shell composites by the alpha channel, and the mask only
    // has to exist. 1bpp rows are word-aligned.
    size_t stride = (((size_t)px + 15) / 16) * 2;
    unsigned char *mask_bits = calloc(stride * (size_t)px, 1);
    HBITMAP mask = mask_bits ? CreateBitmap(px, px, 1, 1, mask_bits) : NULL;
    HICON icon = NULL;
    if (mask) {
        ICONINFO ii = { TRUE, 0, 0, mask, color };
        icon = CreateIconIndirect(&ii);
        DeleteObject(mask);
    }
    free(mask_bits);
    DeleteObject(color);
    return icon;
}

static HICON grid_icon(tray_icon_state st) {
    return grid_icon_px(st, icon_px());
}

// Design-review seam (see tray.h). Writes 32-bit top-down BMPs with an
// alpha channel (a V4 header names the masks), because they need no encoder
// and every Windows viewer opens them; viewers that ignore alpha show the
// glyph on black, the premultiplied colour.
bool tray_platform_icon_dump(const char *dir, int px) {
    const char *names[] = { "idle", "loaded", "running" };
    if (px < 8 || px > 512) return false;
    bool ok = true;
    for (int i = 0; i < 3 && ok; i++) {
        size_t nbytes = (size_t)px * (size_t)px * 4;
        unsigned char *bits = malloc(nbytes);
        if (!bits || !tray_glyph_render((tray_icon_state)i, px, 0xFFFFFFu, bits)) {
            free(bits);
            ok = false;
            break;
        }
        BITMAPV4HEADER bi;
        memset(&bi, 0, sizeof bi);
        bi.bV4Size = sizeof bi;
        bi.bV4Width = px;
        bi.bV4Height = -px;             // top-down, as rendered
        bi.bV4Planes = 1;
        bi.bV4BitCount = 32;
        bi.bV4V4Compression = BI_BITFIELDS;
        bi.bV4RedMask = 0x00FF0000; bi.bV4GreenMask = 0x0000FF00;
        bi.bV4BlueMask = 0x000000FF; bi.bV4AlphaMask = 0xFF000000;
        bi.bV4CSType = 0x73524742;      // "sRGB"
        char path[1200];
        snprintf(path, sizeof path, "%s\\tray-%s.bmp", dir, names[i]);
        FILE *f = fopen(path, "wb");
        if (!f) {
            ok = false;
        } else {
            BITMAPFILEHEADER fh = {0};
            fh.bfType = 0x4D42;   // "BM"
            fh.bfOffBits = sizeof fh + sizeof bi;
            fh.bfSize = fh.bfOffBits + (DWORD)nbytes;
            fwrite(&fh, sizeof fh, 1, f);
            fwrite(&bi, sizeof bi, 1, f);
            fwrite(bits, 1, nbytes, f);
            fclose(f);
            printf("wrote %s\n", path);
        }
        free(bits);
    }
    return ok;
}

static void set_icon(void) {
    HICON ic = grid_icon(tray_icon());
    g_nid.hIcon = ic;
    Shell_NotifyIconA(NIM_MODIFY, &g_nid);
    DestroyIcon(ic);
}

// -------------------------------------------------------------- popup menu

// The core writes its labels in UTF-8 (the ● that marks the managed
// instance, the … on every row that opens something). The ANSI menu entry
// points read them in the console code page and drew each byte of those
// glyphs as its own character. Every label goes through the wide entry
// points, converted here; a label that will not convert is shown as its
// raw bytes rather than dropped, so a row is never missing from the menu.
static const wchar_t *menu_wide(const char *utf8, wchar_t *buf, int cap) {
    int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8, -1, buf, cap);
    if (n > 0) return buf;
    n = MultiByteToWideChar(CP_ACP, 0, utf8, -1, buf, cap);
    if (n > 0) return buf;
    buf[0] = 0;
    return buf;
}

// The menu for a row list, separated from the core call so a test can hand
// it labels and read them back with GetMenuStringW. Item ids count from
// MENU_ID_BASE by row index, which the command handler reverses.
static HMENU menu_from_items(const tray_item *items, int n) {
    HMENU root = CreatePopupMenu();
    HMENU cur = root;
    HMENU stack_parent = NULL;
    int last_pos = -1;
    wchar_t wide[512];

    for (int i = 0; i < n; i++) {
        const tray_item *t = &items[i];
        switch (t->kind) {
        case TRAY_K_SEP:
            AppendMenuW(cur, MF_SEPARATOR, 0, NULL);
            break;
        case TRAY_K_SUB_BEGIN: {
            HMENU sub = CreatePopupMenu();
            if (last_pos >= 0)
                ModifyMenuW(cur, (UINT)last_pos, MF_BYPOSITION | MF_POPUP | MF_STRING,
                            (UINT_PTR)sub, menu_wide(items[i - 1].label, wide, 512));
            stack_parent = cur;
            cur = sub;
            break;
        }
        case TRAY_K_SUB_END:
            if (stack_parent) { cur = stack_parent; stack_parent = NULL; }
            break;
        default: {
            UINT flags = MF_STRING;
            UINT_PTR id = (UINT_PTR)(MENU_ID_BASE + i);
            if (t->kind == TRAY_K_LABEL) { flags |= MF_GRAYED; id = 0; }
            if (t->kind == TRAY_K_CHECK && t->checked) flags |= MF_CHECKED;
            AppendMenuW(cur, flags, id, menu_wide(t->label, wide, 512));
            last_pos = GetMenuItemCount(cur) - 1;
            break;
        }
        }
    }
    return root;
}

static HMENU build_menu(void) {
    g_nitems = tray_menu_build(g_items, TRAY_MAX_ITEMS);
    return menu_from_items(g_items, g_nitems);
}

static void pick_model(void) {
    char file[1024] = "";
    OPENFILENAMEA ofn = { .lStructSize = sizeof ofn };
    ofn.hwndOwner = g_hwnd;
    ofn.lpstrFilter = "GGUF models (*.gguf)\0*.gguf\0All files\0*.*\0";
    ofn.lpstrFile = file;
    ofn.nMaxFile = sizeof file;
    ofn.lpstrTitle = "Choose a GGUF model for the desktop-managed runner";
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
    if (GetOpenFileNameA(&ofn))
        tray_menu_act(TRAY_ACT_PICK_MODEL, 0, file);
    else
        tray_menu_act(TRAY_ACT_PICK_MODEL, 0, NULL);
}

static LRESULT CALLBACK wndproc(HWND h, UINT msg, WPARAM w, LPARAM l) {
    switch (msg) {
    case WM_TRAY_CALLBACK:
        if (LOWORD(l) == WM_RBUTTONUP || LOWORD(l) == WM_LBUTTONUP) {
            HMENU m = build_menu();
            POINT pt;
            GetCursorPos(&pt);
            SetForegroundWindow(h);  // required or the menu won't dismiss
            UINT cmd = (UINT)TrackPopupMenu(m,
                TPM_RETURNCMD | TPM_NONOTIFY | TPM_RIGHTBUTTON,
                pt.x, pt.y, 0, h, NULL);
            DestroyMenu(m);
            if (cmd >= MENU_ID_BASE) {
                tray_item *t = &g_items[cmd - MENU_ID_BASE];
                if (t->action == TRAY_ACT_PICK_MODEL)
                    pick_model();
                else
                    tray_menu_act(t->action, t->arg, NULL);
            }
            if (tray_should_quit()) PostQuitMessage(0);
            else set_icon();
        }
        return 0;
    case WM_TIMER:
        if (tray_should_quit()) PostQuitMessage(0);
        else set_icon();
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcA(h, msg, w, l);
}

// -------------------------------------------------------------- autostart
// HKCU\Software\Microsoft\Windows\CurrentVersion\Run, value "XyntetikTray".

#define RUN_KEY "Software\\Microsoft\\Windows\\CurrentVersion\\Run"
#define RUN_VAL "XyntetikTray"
#define RUN_VAL_OLD "GridcoreTray"  // pre-rename value, migrated on sight

// GetModuleFileNameA returns 0 on failure (leaving the buffer untouched) and
// on truncation does not NUL-terminate before Windows 10 1607. Either way an
// unchecked buffer is uninitialized or unterminated stack, and both call sites
// below persist it into the autostart command — a corrupt Run value that would
// fail every login. Refuse to write one rather than register garbage.
static bool self_exe(char *out, DWORD cap) {
    DWORD n = GetModuleFileNameA(NULL, out, cap);
    return n != 0 && n < cap;
}

// One-time migration of the pre-rename autostart value (Gridcore ->
// Xyntetik): re-register under the new value with the current executable and
// delete the old one, so autostart survives the rename without leaving two
// registrations behind.
static void migrate_old_autostart(void) {
    HKEY k;
    if (RegOpenKeyExA(HKEY_CURRENT_USER, RUN_KEY, 0, KEY_READ | KEY_SET_VALUE,
                      &k) != ERROR_SUCCESS)
        return;
    if (RegQueryValueExA(k, RUN_VAL_OLD, NULL, NULL, NULL, NULL) == ERROR_SUCCESS) {
        char exe[1024], cmd[1100];
        if (self_exe(exe, sizeof exe)) {
            snprintf(cmd, sizeof cmd, "\"%s\" --tray", exe);
            RegSetValueExA(k, RUN_VAL, 0, REG_SZ,
                           (const BYTE *)cmd, (DWORD)strlen(cmd) + 1);
            RegDeleteValueA(k, RUN_VAL_OLD);
        }
    }
    RegCloseKey(k);
}

bool tray_platform_autostart_get(void) {
    migrate_old_autostart();
    HKEY k;
    if (RegOpenKeyExA(HKEY_CURRENT_USER, RUN_KEY, 0, KEY_READ, &k) != ERROR_SUCCESS)
        return false;
    bool present = RegQueryValueExA(k, RUN_VAL, NULL, NULL, NULL, NULL) == ERROR_SUCCESS;
    RegCloseKey(k);
    return present;
}

bool tray_platform_autostart_set(bool on) {
    HKEY k;
    if (RegOpenKeyExA(HKEY_CURRENT_USER, RUN_KEY, 0, KEY_SET_VALUE, &k) != ERROR_SUCCESS)
        return false;
    bool ok;
    if (on) {
        char exe[1024], cmd[1100];
        if (!self_exe(exe, sizeof exe)) {
            RegCloseKey(k);
            return false;
        }
        snprintf(cmd, sizeof cmd, "\"%s\" --tray", exe);
        ok = RegSetValueExA(k, RUN_VAL, 0, REG_SZ,
                            (const BYTE *)cmd, (DWORD)strlen(cmd) + 1) == ERROR_SUCCESS;
    } else {
        LONG rc = RegDeleteValueA(k, RUN_VAL);
        ok = rc == ERROR_SUCCESS || rc == ERROR_FILE_NOT_FOUND;
    }
    RegCloseKey(k);
    return ok;
}

// -------------------------------------------------------------- main loop

int tray_platform_run(void) {
    migrate_old_autostart();
    FreeConsole();  // detach from any console we were launched from

    // Per-monitor DPI awareness, before any window exists: the shell then
    // asks this process for an icon at the size it will draw (SM_CXSMICON
    // follows the DPI) instead of stretching a 16 px one. Resolved at run
    // time so the binary still starts on a Windows without the entry point.
    HMODULE user32 = GetModuleHandleA("user32.dll");
    typedef BOOL (WINAPI *set_ctx_fn)(HANDLE);
    set_ctx_fn set_ctx = user32
        ? (set_ctx_fn)(void *)GetProcAddress(user32, "SetProcessDpiAwarenessContext") : NULL;
    if (set_ctx) set_ctx((HANDLE)-4);   // DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2

    WNDCLASSA wc = { .lpfnWndProc = wndproc,
                     .hInstance = GetModuleHandleA(NULL),
                     .lpszClassName = "XyntetikTrayWnd" };
    RegisterClassA(&wc);
    g_hwnd = CreateWindowA(wc.lpszClassName, "xyntetik-tray", 0, 0, 0, 0, 0,
                           HWND_MESSAGE, NULL, wc.hInstance, NULL);
    if (!g_hwnd) return 1;

    memset(&g_nid, 0, sizeof g_nid);
    g_nid.cbSize = sizeof g_nid;
    g_nid.hWnd = g_hwnd;
    g_nid.uID = TRAY_ICON_ID;
    g_nid.uFlags = NIF_ICON | NIF_MESSAGE | NIF_TIP;
    g_nid.uCallbackMessage = WM_TRAY_CALLBACK;
    g_nid.hIcon = grid_icon(tray_icon());
    snprintf(g_nid.szTip, sizeof g_nid.szTip, "xyntetik-runner");
    Shell_NotifyIconA(NIM_ADD, &g_nid);

    SetTimer(g_hwnd, 1, 5000, NULL);  // badge refresh while menu is closed

    MSG msg;
    while (GetMessageA(&msg, NULL, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }

    Shell_NotifyIconA(NIM_DELETE, &g_nid);
    DestroyWindow(g_hwnd);
    return 0;
}

#endif // _WIN32
