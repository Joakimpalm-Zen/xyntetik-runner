// macOS menu-bar backend: NSStatusItem + NSMenu. Rendering only — every
// menu is built by tray_menu_build() at open time and clicks go straight
// back through tray_menu_act(). Accessory activation policy: no Dock icon,
// no menu bar of our own.
#import <AppKit/AppKit.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#include "tray.h"

#include <errno.h>
#include <fcntl.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

extern char **environ;

#define TRAY_MAX_ITEMS 128

static NSStatusItem *g_status;

// ---------------------------------------------------------------- the icon
// The Xyntetik ensö, drawn in code as a template image so macOS recolors it
// for light and dark menu bars. Three states, matching the three the core
// can actually distinguish, and matching the brand marks one to one (the
// site's marks/ set: an open ring, the spark at its end, the Runner streaks):
//
//   IDLE     the bare ensö                     nothing loaded
//   LOADED   ensö + the spark at its end       model resident, waiting
//   RUNNING  the full Runner mark: ensö,       inference in flight
//            spark and three motion streaks
//
// Geometry is the 100-unit mark scaled to an 18-pt box (0.18) so the menu-bar
// glyph and the site's mark are the same drawing: ring radius 40 -> 7.2, gap
// at the top between the mark's end points (115 degrees upper-left, 60
// degrees upper-right, in y-up terms), the spark on the upper-right end, the
// streaks at the mark's three rows.
static NSImage *core_icon_px(tray_icon_state st, CGFloat px) {
    const CGFloat s = px / 18.0;   // geometry below is authored in 18-pt units
    NSImage *img = [NSImage imageWithSize:NSMakeSize(px, px)
                                  flipped:NO
                           drawingHandler:^BOOL(NSRect rect) {
        (void)rect;
        [[NSColor blackColor] setFill];
        [[NSColor blackColor] setStroke];
        const CGFloat cx = 9.0 * s, cy = 9.0 * s;
        NSPoint c = NSMakePoint(cx, cy);

        // the ring: the long way round from 115 to 60 degrees, open at the top
        const CGFloat r = 7.0 * s;
        NSBezierPath *ring = [NSBezierPath bezierPath];
        ring.lineWidth = 1.7 * s;
        ring.lineCapStyle = NSLineCapStyleRound;
        [ring appendBezierPathWithArcWithCenter:c radius:r
                                     startAngle:115.0 endAngle:60.0];
        [ring stroke];

        if (st == TRAY_ICON_IDLE) return YES;

        // the spark: a dot on the ring's upper-right end
        const CGFloat ex = cx + r * cos(60.0 * M_PI / 180.0);
        const CGFloat ey = cy + r * sin(60.0 * M_PI / 180.0);
        const CGFloat dr = 1.35 * s;
        [[NSBezierPath bezierPathWithOvalInRect:
            NSMakeRect(ex - dr, ey - dr, 2 * dr, 2 * dr)] fill];

        if (st == TRAY_ICON_RUNNING) {
            // the Runner streaks: three rows, the middle one longest, each a
            // round-capped stroke tapering nowhere (a template image has no
            // gradient to fade them with, so they are plain motion lines)
            struct { CGFloat y, x0, x1, w; } rows[] = {
                { cy + 2.07 * s, cx - 4.1 * s, cx + 2.3 * s, 0.95 * s },
                { cy,            cx - 5.2 * s, cx + 4.0 * s, 1.2 * s },
                { cy - 2.07 * s, cx - 3.8 * s, cx + 1.6 * s, 0.95 * s },
            };
            for (int i = 0; i < 3; i++) {
                NSBezierPath *ln = [NSBezierPath bezierPath];
                ln.lineWidth = rows[i].w;
                ln.lineCapStyle = NSLineCapStyleRound;
                [ln moveToPoint:NSMakePoint(rows[i].x0, rows[i].y)];
                [ln lineToPoint:NSMakePoint(rows[i].x1, rows[i].y)];
                [ln stroke];
            }
        }
        return YES;
    }];
    [img setTemplate:YES];
    return img;
}

static NSImage *core_icon(tray_icon_state st) { return core_icon_px(st, 18.0); }

bool tray_platform_icon_dump(const char *dir, int px) {
    @autoreleasepool {
        const char *names[] = { "idle", "loaded", "running" };
        for (int i = 0; i < 3; i++) {
            // template images carry no colour, so paint the glyph onto an
            // opaque white ground — otherwise the PNG is black-on-transparent
            // and unreviewable in most viewers
            NSImage *glyph = core_icon_px((tray_icon_state)i, px);
            NSBitmapImageRep *rep = [[NSBitmapImageRep alloc]
                initWithBitmapDataPlanes:NULL pixelsWide:px pixelsHigh:px
                            bitsPerSample:8 samplesPerPixel:4 hasAlpha:YES
                                 isPlanar:NO colorSpaceName:NSDeviceRGBColorSpace
                              bytesPerRow:0 bitsPerPixel:0];
            [NSGraphicsContext saveGraphicsState];
            NSGraphicsContext.currentContext =
                [NSGraphicsContext graphicsContextWithBitmapImageRep:rep];
            [[NSColor whiteColor] setFill];
            NSRectFill(NSMakeRect(0, 0, px, px));
            [glyph drawInRect:NSMakeRect(0, 0, px, px)];
            [NSGraphicsContext restoreGraphicsState];

            NSData *png = [rep representationUsingType:NSBitmapImageFileTypePNG
                                            properties:@{}];
            char path[1200];
            snprintf(path, sizeof path, "%s/tray-%s.png", dir, names[i]);
            if (![png writeToFile:[NSString stringWithUTF8String:path]
                       atomically:YES])
                return false;
            printf("wrote %s\n", path);
        }
    }
    return true;
}

// ------------------------------------------------------------- menu bridge

@interface TrayDelegate : NSObject <NSMenuDelegate>
- (void)clicked:(NSMenuItem *)sender;
@end

@implementation TrayDelegate

- (void)menuNeedsUpdate:(NSMenu *)menu {
    [menu removeAllItems];
    static tray_item items[TRAY_MAX_ITEMS];
    int n = tray_menu_build(items, TRAY_MAX_ITEMS);

    NSMenu *cur = menu;
    NSMenuItem *last = nil;
    for (int i = 0; i < n; i++) {
        tray_item *t = &items[i];
        switch (t->kind) {
        case TRAY_K_SEP:
            [cur addItem:[NSMenuItem separatorItem]];
            break;
        case TRAY_K_SUB_BEGIN: {
            NSMenu *sub = [[NSMenu alloc] init];
            sub.autoenablesItems = NO;
            if (last) {
                // AppKit refuses to open a submenu hanging off a disabled
                // item — the parent row must be enabled (it has no action,
                // so clicking it still does nothing)
                last.enabled = YES;
                [cur setSubmenu:sub forItem:last];
            }
            cur = sub;
            break;
        }
        case TRAY_K_SUB_END:
            cur = menu;
            break;
        default: {
            NSMenuItem *mi = [[NSMenuItem alloc]
                initWithTitle:[NSString stringWithUTF8String:t->label]
                       action:(t->kind == TRAY_K_LABEL ? nil : @selector(clicked:))
                keyEquivalent:@""];
            mi.target = (t->kind == TRAY_K_LABEL) ? nil : self;
            mi.enabled = (t->kind != TRAY_K_LABEL);
            mi.tag = i;
            mi.representedObject = @[ @(t->action), @(t->arg) ];
            if (t->kind == TRAY_K_CHECK)
                mi.state = t->checked ? NSControlStateValueOn : NSControlStateValueOff;
            [cur addItem:mi];
            last = mi;
            break;
        }
        }
    }
    g_status.button.image = core_icon(tray_icon());
}

- (void)clicked:(NSMenuItem *)sender {
    NSArray *ra = sender.representedObject;
    int action = [ra[0] intValue];
    long arg = [ra[1] longValue];

    if (action == TRAY_ACT_PICK_MODEL) {
        NSOpenPanel *panel = [NSOpenPanel openPanel];
        panel.canChooseFiles = YES;
        panel.canChooseDirectories = NO;
        panel.allowsMultipleSelection = NO;
        UTType *gguf = [UTType typeWithFilenameExtension:@"gguf"];
        if (gguf) panel.allowedContentTypes = @[ gguf ];
        panel.message = @"Choose a GGUF model for the desktop-managed runner";
        [NSApp activateIgnoringOtherApps:YES];
        if ([panel runModal] == NSModalResponseOK && panel.URL)
            tray_menu_act(action, 0, panel.URL.path.UTF8String);
        else
            tray_menu_act(action, 0, NULL);
    } else {
        tray_menu_act(action, arg, NULL);
    }

    if (tray_should_quit())
        [NSApp stop:nil];
    else
        g_status.button.image = core_icon(tray_icon());
}

- (void)tick:(NSTimer *)timer {
    if (tray_should_quit()) { [NSApp stop:nil]; return; }
    g_status.button.image = core_icon(tray_icon());
}

@end

// -------------------------------------------------------------- autostart
// LaunchAgent at ~/Library/LaunchAgents/ai.xyntetik.runner.tray.plist.
// Presence of the file IS the state; no launchctl bookkeeping in v1.

static void agent_path(char *out, size_t cap) {
    const char *home = getenv("HOME");
    snprintf(out, cap, "%s/Library/LaunchAgents/ai.xyntetik.runner.tray.plist",
             home ? home : ".");
}

static void unload_old_agent(void) {
    char *argv[] = {
        (char *)"launchctl", (char *)"remove",
        (char *)"ai.gridcore.runner.tray", NULL,
    };
    posix_spawn_file_actions_t fa;
    if (posix_spawn_file_actions_init(&fa) != 0) return;
    posix_spawn_file_actions_addopen(&fa, 1, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_adddup2(&fa, 1, 2);

    pid_t pid;
    int rc = posix_spawnp(&pid, argv[0], &fa, NULL, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    if (rc != 0) return;

    while (waitpid(pid, NULL, 0) < 0 && errno == EINTR) {}
}

// One-time migration of the pre-rename agent (Gridcore -> Xyntetik). The old
// label must be unloaded before the new one registers, or two trays can end
// up installed; autostart state is preserved by re-creating the agent under
// the new label.
static void migrate_old_agent(void) {
    static bool done = false;
    if (done) return;
    done = true;
    const char *home = getenv("HOME");
    char op[1024];
    snprintf(op, sizeof op,
             "%s/Library/LaunchAgents/ai.gridcore.runner.tray.plist",
             home ? home : ".");
    FILE *f = fopen(op, "rb");
    if (!f) return;
    fclose(f);
    unload_old_agent();
    remove(op);
    tray_platform_autostart_set(true);
}

bool tray_platform_autostart_get(void) {
    migrate_old_agent();
    char p[1024];
    agent_path(p, sizeof p);
    FILE *f = fopen(p, "rb");
    if (f) fclose(f);
    return f != NULL;
}

bool tray_platform_autostart_set(bool on) {
    char p[1024];
    agent_path(p, sizeof p);
    if (!on) return remove(p) == 0;

    char exe[1200];
    uint32_t sz = sizeof exe;
    extern int _NSGetExecutablePath(char *, uint32_t *);
    if (_NSGetExecutablePath(exe, &sz) != 0) return false;

    FILE *f = fopen(p, "wb");
    if (!f) return false;
    fprintf(f,
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\""
        " \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
        "<plist version=\"1.0\"><dict>\n"
        "  <key>Label</key><string>ai.xyntetik.runner.tray</string>\n"
        "  <key>ProgramArguments</key><array>\n"
        "    <string>%s</string>\n"
        "    <string>--tray</string>\n"
        "  </array>\n"
        "  <key>RunAtLoad</key><true/>\n"
        "</dict></plist>\n",
        exe);
    return fclose(f) == 0;
}

// -------------------------------------------------------------- main loop

int tray_platform_run(void) {
    migrate_old_agent();
    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];

        TrayDelegate *del = [[TrayDelegate alloc] init];
        g_status = [[NSStatusBar systemStatusBar]
            statusItemWithLength:NSSquareStatusItemLength];
        g_status.button.image = core_icon(tray_icon());
        g_status.button.toolTip = @"xyntetik-runner";

        NSMenu *menu = [[NSMenu alloc] init];
        menu.autoenablesItems = NO;
        menu.delegate = del;
        g_status.menu = menu;

        // badge refresh while the menu is closed (menuNeedsUpdate only
        // fires on open)
        [NSTimer scheduledTimerWithTimeInterval:5.0
                                         target:del
                                       selector:@selector(tick:)
                                       userInfo:nil
                                        repeats:YES];

        [NSApp run];
        [[NSStatusBar systemStatusBar] removeStatusItem:g_status];
    }
    return 0;
}
