// Fallback backend for platforms without a tray implementation (Linux in
// v1). Keeps --tray a defined flag everywhere with an honest error.
#include "tray.h"
#include <stdio.h>

// how many times a tray reached its run loop: test_tray_core reads it to see a
// tray refused before it ran (R9.5)
int tray_stub_runs;

int tray_platform_run(void) {
    tray_stub_runs++;
    fprintf(stderr, "error: --tray is not available on this platform "
                    "(macOS and Windows only in v1)\n");
    return 1;
}

bool tray_platform_autostart_get(void) { return false; }

// no offscreen renderer in the stub; the seam reports failure rather than
// writing nothing and claiming success
bool tray_platform_icon_dump(const char *dir, int px) {
    (void)dir; (void)px;
    return false;
}
bool tray_platform_autostart_set(bool on) { (void)on; return false; }
