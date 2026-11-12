// The macOS tray's menu rebuild gives back the items it made (R6.68.5).
//
// tray_macos.m was built without ARC and released nothing: menuNeedsUpdate:
// made every NSMenuItem and every submenu with alloc/init (+1), handed them to
// a menu that retains them again, and never let go of its own reference, so
// each open of the tray menu leaked the whole menu it replaced. The file is now
// built with -fobjc-arc. This test (itself ARC, for zeroing weak references)
// rebuilds the menu, keeps a weak reference to every item and submenu the
// first rebuild made, rebuilds again, and demands that all of them are gone.
//
// HOME points at an empty folder first: the menu reads the instance registry
// under it, and must find nothing of the person running the test.
#import <AppKit/AppKit.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
}

#define MAX_SEEN 512

int main(void) {
    char home[] = "/tmp/test-tray-macos-arc-XXXXXX";
    if (!mkdtemp(home)) { perror("mkdtemp"); return 2; }
    setenv("HOME", home, 1);
    @autoreleasepool {
        Class cls = NSClassFromString(@"TrayDelegate");
        ck(cls != nil, "the tray's menu delegate is linked in");
        if (!cls) return 1;
        id<NSMenuDelegate> del = [[cls alloc] init];
        NSMenu *menu = [[NSMenu alloc] init];
        __weak id seen[MAX_SEEN];
        int n = 0;
        @autoreleasepool {
            [del menuNeedsUpdate:menu];
            for (NSMenuItem *mi in menu.itemArray) {
                // AppKit may share separator items; they are not the tray's to free
                if (mi.isSeparatorItem || n >= MAX_SEEN) continue;
                seen[n++] = mi;
                if (mi.submenu && n < MAX_SEEN) {
                    seen[n++] = mi.submenu;
                    for (NSMenuItem *si in mi.submenu.itemArray)
                        if (!si.isSeparatorItem && n < MAX_SEEN) seen[n++] = si;
                }
            }
            ck(n > 0, "the rebuilt menu has items");
            [del menuNeedsUpdate:menu];   // replaces every item of the first rebuild
        }
        int alive = 0;
        for (int i = 0; i < n; i++)
            if (seen[i]) alive++;
        if (alive) fprintf(stderr, "%d of %d items and submenus outlived the rebuild\n", alive, n);
        ck(alive == 0, "a rebuild gives back every item and submenu the previous one made");
        ck(menu.itemArray.count > 0, "the second rebuild has its own items");
        if (!g_fail) fprintf(stderr, "tray macOS ARC ok: %d items and submenus given back\n", n);
    }
    rmdir(home);
    return g_fail;
}
