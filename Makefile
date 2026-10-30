CC      ?= cc
# Probe by RUNNING the interpreter, not by looking it up: on Windows a
# python3 "app execution alias" stub exists on PATH but exits non-zero with
# a Store advert, so `command -v python3` picks an interpreter that cannot run.
# Prefer an interpreter that can run the mandatory pytest gates; only fall
# back to a merely runnable one so non-test targets retain their stdlib-only
# dependency contract and `test-python-deps` can give one precise error.
PYTHON  ?= $(shell if python3 -c "import pytest" >/dev/null 2>&1; then echo python3; \
                    elif python -c "import pytest" >/dev/null 2>&1; then echo python; \
                    elif python3 -c "" >/dev/null 2>&1; then echo python3; \
                    else echo python; fi)
# gnu11 (not c11): strict ISO mode hides M_PI and POSIX symbols on glibc/MinGW
# -march=native unlocks the AVX2/FMA/F16C dot kernels in quants.c on x86;
# other ISAs (ARM macs) compile the scalar fallbacks
CFLAGS  ?= -O3 -ffast-math -std=gnu11 -Wall -Wextra -Wno-unused-parameter -march=native
# Mandatory engine codegen, appended so it survives a hostile *environment*
# CFLAGS. A conda/distro toolchain that exports CFLAGS=-march=nocona -O2
# otherwise silently defeats the `?=` default above: __AVX2__ goes undefined,
# every AVX2/FMA/F16C dot kernel in quants.c is `#if`-compiled out, and the
# runner ships a SCALAR binary on AVX-512 hardware (measured: zero ymm/zmm
# instructions, ~6x slower end-to-end). Plain `+=` (NOT `override`) is exactly
# right here: it appends to an environment-set CFLAGS so the last -O/-march wins
# and the conda clobber is undone, but it is ignored for a *command-line* CFLAGS
# so the release build's portable `make CFLAGS="... -march=x86-64-v3"` pin is
# preserved (a release must not bake in the build host's -march=native).
# Cross-compile a local build with ARCH_FLAGS=-march=<target>. NB: do NOT name
# this RUNNER_ARCH — GitHub Actions sets RUNNER_ARCH=X64/ARM64 in the build
# environment, which a `?=` inherits and then leaks as a bogus bare compiler arg.
ARCH_FLAGS ?= -march=native
CFLAGS += -O3 -ffast-math -std=gnu11 $(ARCH_FLAGS)
LDFLAGS  = -lm -lpthread
# The build platform, decided once from the shell the build runs in. Windows
# is what that shell's uname reports there: the MSYS2 runtime prints
# MSYS_NT-<ver> with MSYSTEM unset and MINGW64_NT-<ver> or MINGW32_NT-<ver>
# under any other MSYSTEM (UCRT64 and CLANG64 included), Git for Windows
# MINGW64_NT-<ver>, Cygwin CYGWIN_NT-<ver>, the busybox-w32 uname Windows_NT.
# OS=Windows_NT, which Windows puts in a full environment, still counts for a
# make started without a uname (mingw32-make from cmd.exe) and on the command
# line. It is not enough alone: a bash.exe started directly gets a minimal
# environment without it, and the build took the POSIX branch, linking a
# `runner` with no .exe against -ldl, which MinGW does not have.
# tests/test_ci_contracts.py (PlatformDetection) holds each case.
UNAME_S := $(shell uname -s 2>/dev/null)
ifneq ($(filter MINGW% MSYS% CYGWIN% Windows_NT,$(UNAME_S))$(filter Windows_NT,$(OS)),)
BUILD_OS := Windows
else
BUILD_OS := $(UNAME_S)
endif
ifeq ($(BUILD_OS),Windows)
# -static: link winpthread/libgcc into the exe so it runs outside an MSYS2
# shell (otherwise it dies at load with STATUS_DLL_NOT_FOUND on libwinpthread-1.dll)
CFLAGS += -Werror=format-truncation -Werror=cpp
LDFLAGS += -lws2_32 -lpsapi -static   # psapi: QueryWorkingSetEx / GetProcessMemoryInfo
# tray: gdi32 (icon painting) and comdlg32 (GetOpenFileName) are not in the
# MinGW default lib set; shell32/advapi32 are but stay explicit for clarity
LDFLAGS += -lshell32 -lgdi32 -lcomdlg32 -ladvapi32
# The Python suites read and write UTF-8 (the runner's output, fixtures,
# records); without UTF-8 mode Windows decodes them as cp1252 and a test
# fails on whatever bytes the model happened to print. CI sets it per job;
# here a local `make test` gets it too.
export PYTHONUTF8 = 1
# runner.exe's application manifest (src/runner.manifest, linked through
# src/runner.rc) makes UTF-8 the process code page. windres is part of
# binutils, which the MSYS2 gcc package depends on.
WINDRES ?= windres
GPU_SRC  = src/cuda.c
GPU_BACKEND_DEF = -DRUNNER_GPU_CUDA
TRAY_SRC = src/tray.c src/tray_win.c
RUNNER_EXE = runner.exe
TEST_JSON_SCHEMA = test-json-schema.exe
TEST_SVAL_WALK = test-sval-walk.exe
TEST_TOKENIZER = test-tokenizer.exe
TEST_TEMPLATE = test-template.exe
TEST_PROMPT_MARKS = test-prompt-marks.exe
TEST_TOOLS = test-tools.exe
TEST_JSON_OOM = test-json-oom.exe
TEST_TOKENIZER_OOM = test-tokenizer-oom.exe
TEST_TOK_CLASSES = test-tokenizer-classes.exe
TEST_TEMPLATE_OOM = test-template-oom.exe
TEST_TEMPLATE_FAMILIES = test-template-families.exe
TEST_SCHEMA_OOM = test-schema-oom.exe
TEST_SAMPLER = test-sampler.exe
TEST_SHARED = test-shared-weights.exe
TEST_MLOCK = test-mlock.exe
TEST_BATCH = test-batch.exe
DIFFTOK = difftok.exe
TEST_BIND = test-bind.exe
else ifeq ($(BUILD_OS),Darwin)
GPU_SRC  = src/metal.m
GPU_BACKEND_DEF = -DRUNNER_GPU_METAL
# Compiler-specific correctness annotations must not silently disappear.  The
# Mamba-2 recurrence, in particular, relies on precise FP semantics and an
# ignored attribute lets both sides of its identity gate agree on the wrong
# arithmetic regime.
CFLAGS += -Werror=nan-infinity-disabled -Werror=unknown-attributes
LDFLAGS += -framework Metal -framework Foundation
# AppKit only on Darwin, only for the tray backend; UniformTypeIdentifiers
# for the non-deprecated NSOpenPanel file filter
LDFLAGS += -framework AppKit -framework UniformTypeIdentifiers
TRAY_SRC = src/tray.c src/tray_macos.m
RUNNER_EXE = runner
TEST_JSON_SCHEMA = test-json-schema
TEST_SVAL_WALK = test-sval-walk
TEST_TOKENIZER = test-tokenizer
TEST_TEMPLATE = test-template
TEST_PROMPT_MARKS = test-prompt-marks
TEST_TOOLS = test-tools
TEST_JSON_OOM = test-json-oom
TEST_TOKENIZER_OOM = test-tokenizer-oom
TEST_TOK_CLASSES = test-tokenizer-classes
TEST_TEMPLATE_OOM = test-template-oom
TEST_TEMPLATE_FAMILIES = test-template-families
TEST_SCHEMA_OOM = test-schema-oom
TEST_SAMPLER = test-sampler
TEST_SHARED = test-shared-weights
TEST_MLOCK = test-mlock
TEST_BATCH = test-batch
DIFFTOK = difftok
TEST_BIND = test-bind
else
GPU_SRC  = src/cuda.c
GPU_BACKEND_DEF = -DRUNNER_GPU_CUDA
LDFLAGS += -ldl
TRAY_SRC = src/tray.c src/tray_stub.c
RUNNER_EXE = runner
TEST_JSON_SCHEMA = test-json-schema
TEST_SVAL_WALK = test-sval-walk
TEST_TOKENIZER = test-tokenizer
TEST_TEMPLATE = test-template
TEST_PROMPT_MARKS = test-prompt-marks
TEST_TOOLS = test-tools
TEST_JSON_OOM = test-json-oom
TEST_TOKENIZER_OOM = test-tokenizer-oom
TEST_TOK_CLASSES = test-tokenizer-classes
TEST_TEMPLATE_OOM = test-template-oom
TEST_TEMPLATE_FAMILIES = test-template-families
TEST_SCHEMA_OOM = test-schema-oom
TEST_SAMPLER = test-sampler
TEST_SHARED = test-shared-weights
TEST_MLOCK = test-mlock
TEST_BATCH = test-batch
DIFFTOK = difftok
TEST_BIND = test-bind
endif

# Workers for the root pytest (make test-python PYTEST_WORKERS=auto): empty
# runs pytest in one process, as a local `make test` always has.
PYTEST_WORKERS ?=
PYTEST_XDIST = $(if $(PYTEST_WORKERS),-n $(PYTEST_WORKERS) --dist loadfile,)

# Which GPU backend this build links. It was previously visible only to
# test-split-guard, so source could not tell CUDA from Metal -- and the two
# differ in what their attention kernels can address (see RUNNER_KV_RING).
# `override` is intentional and narrowly scoped to this required identity:
# tagged releases set CFLAGS on make's command line, which otherwise suppresses
# an ordinary `+=` and would compile out the Metal safety refusal.
override CFLAGS += $(GPU_BACKEND_DEF)

# T3 build (opt-in, `make T3=1`): the portable bit-exact configuration from
# docs/portable-bitexact-2026-09-05.md. Strict float (no fast-math, no
# contraction), portable transcendental functions and canonical-order dot
# kernels, so the decode path produces the same bytes on arm64, x86-64 and
# riscv64 (measured 165/165 positions on three ISAs). Costs 7-17% decode;
# the default build keeps fast-math. `override` for the same reason as the
# backend define: a release sets CFLAGS on the command line. Objects land in
# their own .build/ directory because BUILD_ID hashes the final flags.
T3 ?= 0
ifeq ($(T3),1)
override CFLAGS := $(filter-out -ffast-math,$(CFLAGS)) -fno-fast-math -ffp-contract=off \
                   -DRUNNER_PORTABLE_MATH -DRUNNER_CANON_KERNELS -DRUNNER_T3_BUILD
endif

# Objects are keyed by the exact compiler command prefix (see the object
# layer below). Defined here, right after CFLAGS is final and BEFORE any rule
# names $(OBJDIR): a target line expands immediately, and an empty BUILD_ID
# at that point would have named `.build//quants.o`, leaving the real quant
# object to the generic pattern rule -- with -ffast-math. test-makefile-sane
# caught exactly that.
BUILD_ID := $(firstword $(shell printf '%s' '$(CC) $(CFLAGS) $(LDFLAGS)' | cksum))
OBJDIR = .build/$(BUILD_ID)

# same .exe suffix rule as every other test binary, without repeating the
# three-way platform branch above
TEST_PREFIX = $(TEST_BATCH:test-batch%=test-prefix%)
TEST_RECURRENT = $(TEST_BATCH:test-batch%=test-recurrent-rewind%)
TEST_PENALTY_WINDOW = $(TEST_BATCH:test-batch%=test-penalty-window%)
TEST_ROPE_YARN = $(TEST_BATCH:test-batch%=test-rope-yarn%)
TEST_ATTN_SCALE = $(TEST_BATCH:test-batch%=test-attn-scale%)
TEST_DOT_ROUTE = $(TEST_BATCH:test-batch%=test-dot-route%)
TEST_REQUEST_STOP = $(TEST_BATCH:test-batch%=test-request-stop%)
TEST_HOST_HEADER = $(TEST_BATCH:test-batch%=test-host-header%)
TEST_GRAMMAR_FF = $(TEST_BATCH:test-batch%=test-grammar-ff%)
TEST_LOOKUP_DRAFT = $(TEST_BATCH:test-batch%=test-lookup-draft%)
TEST_CHOICE_MASS = $(TEST_BATCH:test-batch%=test-choice-mass%)
TEST_VRAMREG = $(TEST_BATCH:test-batch%=test-vram-registry%)
TEST_UPLOAD_REPORT = $(TEST_BATCH:test-batch%=test-upload-report%)
TEST_KV_TOL = $(TEST_BATCH:test-batch%=test-kv-tol%)
TEST_KV_FP4 = $(TEST_BATCH:test-batch%=test-kv-fp4%)
TEST_METAL_KVFP4 = $(TEST_BATCH:test-batch%=test-metal-kvfp4%)
TEST_QUANTS_SIMD = $(TEST_BATCH:test-batch%=test-quants-simd%)
TEST_INSTANCES = $(TEST_BATCH:test-batch%=test-instances%)
TEST_METAL_ADMISSION = $(TEST_BATCH:test-batch%=test-metal-admission%)
TEST_METAL_TENSOR = $(TEST_BATCH:test-batch%=test-metal-tensor%)
TEST_TRAY_CORE = $(TEST_BATCH:test-batch%=test-tray-core%)
TEST_TRAY_WIN = $(TEST_BATCH:test-batch%=test-tray-win-menu%)
TEST_UTF8_CONSOLE = $(TEST_BATCH:test-batch%=test-utf8-console%)
TEST_TC_TOL = $(TEST_BATCH:test-batch%=test-tc-tol%)
TEST_I8_TOL = $(TEST_BATCH:test-batch%=test-i8-tol%)
TEST_LORA_GRAD = $(TEST_BATCH:test-batch%=test-lora-grad%)
TEST_DPO_GRAD = $(TEST_BATCH:test-batch%=test-dpo-grad%)
TEST_KD_GRAD = $(TEST_BATCH:test-batch%=test-kd-grad%)
TEST_MVT = $(TEST_BATCH:test-batch%=test-mvt%)
TEST_TRAIN_GPU = $(TEST_BATCH:test-batch%=test-train-gpu%)
TEST_MVCANON = $(TEST_BATCH:test-batch%=test-mvcanon%)
TEST_MV_TOL = $(TEST_BATCH:test-batch%=test-mv-tol%)
TEST_ATTN_TOL = $(TEST_BATCH:test-batch%=test-attn-tol%)
TEST_GPU_ID = $(TEST_BATCH:test-batch%=test-gpu-identity%)
TEST_MOE_MM_AB = $(TEST_BATCH:test-batch%=test-moe-mm-ab%)
TEST_BATCH_ID = $(TEST_BATCH:test-batch%=test-batch-identity%)
TEST_SPEC_ID = $(TEST_BATCH:test-batch%=test-spec-identity%)
TEST_ATTN_SPLIT = $(TEST_BATCH:test-batch%=test-attn-split-bin%)
TEST_MOE_TOL = $(TEST_BATCH:test-batch%=test-moe-tol%)
TEST_MOE_ROUTER = $(TEST_BATCH:test-batch%=test-moe-router%)
TEST_PAGING_WARN = $(TEST_BATCH:test-batch%=test-paging-warn%)
TEST_AUTOFIT = $(TEST_BATCH:test-batch%=test-autofit%)
TEST_RESP_SM = $(TEST_BATCH:test-batch%=test-responses-sm%)
TEST_BUDGET = $(TEST_BATCH:test-batch%=test-prompt-budget%)
TEST_ATTRIB = $(TEST_BATCH:test-batch%=test-tool-attribution%)
TEST_STOP_CONSTRAINT = $(TEST_BATCH:test-batch%=test-stop-constraint%)
# not a test: the runner-side driver scripts/template-conformance.py renders
# through. Built here rather than by the script so it links the SAME object
# set the server does (see the template-conformance target near the bottom).
TMPL_CONF_RENDER = $(TEST_BATCH:test-batch%=template-conformance-render%)
# test_responses_sm drives the framer through a POSIX socketpair(); winsock
# has none, so on Windows the suite skips it LOUDLY (it runs in Linux CI and
# on the POSIX dev boxes) rather than shimming the transport under the test.
# test_tool_attribution reads its refusals back off the same kind of pair, so
# it skips on the same terms.
# test_tray_win_menu is the inverse case: it reads the popup menu back through
# the Win32 API (GetMenuStringW), which exists nowhere else, so POSIX skips
# it loudly and the Windows job and the Windows box run it.
ifeq ($(BUILD_OS),Windows)
TEST_RESP_SM_DEP =
TEST_RESP_SM_RUN = @echo "skip: test-responses-sm (POSIX socketpair; covered by Linux CI)"
TEST_ATTRIB_DEP =
TEST_ATTRIB_RUN = @echo "skip: test-tool-attribution (POSIX socketpair; covered by Linux CI)"
TEST_MSG_OOM_DEP =
TEST_MSG_OOM_RUN = @echo "skip: test-prompt-oom (POSIX socketpair; covered by Linux CI)"
TEST_TRAY_WIN_DEP = $(TEST_TRAY_WIN)
TEST_TRAY_WIN_RUN = ./$(TEST_TRAY_WIN)
TEST_UTF8_CONSOLE_DEP = $(TEST_UTF8_CONSOLE)
TEST_UTF8_CONSOLE_RUN = ./$(TEST_UTF8_CONSOLE)
else
TEST_RESP_SM_DEP = $(TEST_RESP_SM)
TEST_RESP_SM_RUN = ./$(TEST_RESP_SM)
TEST_ATTRIB_DEP = $(TEST_ATTRIB)
TEST_ATTRIB_RUN = ./$(TEST_ATTRIB)
TEST_MSG_OOM_DEP = $(TEST_MSG_OOM)
TEST_MSG_OOM_RUN = ./$(TEST_MSG_OOM)
TEST_TRAY_WIN_DEP =
TEST_TRAY_WIN_RUN = @echo "skip: test-tray-win-menu (Win32 menu readback; covered by the Windows job)"
TEST_UTF8_CONSOLE_DEP =
TEST_UTF8_CONSOLE_RUN = @echo "skip: test-utf8-console (Windows console code page; covered by the Windows job)"
endif
TEST_QUANTIZE = $(TEST_BATCH:test-batch%=test-quantize%)
TEST_QUANTIZE_OOM = $(TEST_BATCH:test-batch%=test-quantize-oom%)
TEST_VRAM_ROLLBACK = $(TEST_BATCH:test-batch%=test-vram-rollback%)
TEST_GGUF_GETTERS = $(TEST_BATCH:test-batch%=test-gguf-getters%)
TEST_HFHUB = $(TEST_BATCH:test-batch%=test-hfhub%)
TEST_GGUF_SPLIT = $(TEST_BATCH:test-batch%=test-gguf-split-load%)
TEST_PARSE = $(TEST_BATCH:test-batch%=test-parse%)
TEST_ENVELOPE = $(TEST_BATCH:test-batch%=test-envelope%)
TEST_SHA256 = $(TEST_BATCH:test-batch%=test-sha256%)
TEST_ED25519 = $(TEST_BATCH:test-batch%=test-ed25519%)
TEST_MLDSA = $(TEST_BATCH:test-batch%=test-mldsa%)
TEST_PMATH = $(TEST_BATCH:test-batch%=test-pmath%)
TEST_FPCLASS = $(TEST_BATCH:test-batch%=test-fpclass%)
TEST_ACT = $(TEST_BATCH:test-batch%=test-act%)
TEST_ECDSA = $(TEST_BATCH:test-batch%=test-ecdsa%)
TEST_WATERMARK = $(TEST_BATCH:test-batch%=test-watermark%)
TEST_METAL_OWNERSHIP = $(TEST_BATCH:test-batch%=test-metal-ownership%)
TEST_METAL_SHADERS = $(TEST_BATCH:test-batch%=test-metal-shaders%)
TEST_METAL_KQUANTS = $(TEST_BATCH:test-batch%=test-metal-kquants%)
TEST_MODEL_LOAD_FAILURE = $(TEST_BATCH:test-batch%=test-model-load-failure%)
TEST_MODEL_LOAD_OOM = $(TEST_BATCH:test-batch%=test-model-load-oom%)
TEST_MOE_HOST_BRIDGE = $(TEST_BATCH:test-batch%=test-moe-host-bridge%)
TEST_LORA_CHAINS = $(TEST_BATCH:test-batch%=test-lora-chains%)
TEST_MVT_CHAINS = $(TEST_BATCH:test-batch%=test-mvt-chains%)
TEST_DN_RECURRENCE = $(TEST_BATCH:test-batch%=test-dn-recurrence%)
TEST_ATTN_BW = $(TEST_BATCH:test-batch%=test-attn-bw%)
TEST_THREAD_DEFAULT = $(TEST_BATCH:test-batch%=test-thread-default%)

# Every module header, so a change to any of them rebuilds. runner.h was one
# file until 0.1.5; the split (RNR-018) would otherwise have quietly narrowed
# what `make` considers a dependency.
HDR = $(wildcard src/*.h)

# Quant arithmetic is the numerical identity boundary.  Keep it in a separate
# translation unit with fast-math explicitly disabled; the rest of the engine
# retains -ffast-math.  The final -fno-fast-math also defeats a hostile or
# distro-supplied CFLAGS that enabled it before our filtered flags.
QUANTS_OBJ = $(OBJDIR)/quants.o
QUANTS_CFLAGS = $(filter-out -ffast-math,$(CFLAGS)) -fno-fast-math

# The offline GGUF quantizer (--quantize) is quant arithmetic too: its Q8_0
# (and q4_0/q3_k) output must be the canonical, bit-exact ggml result, so a
# runner-produced file is byte-identical to llama.cpp's for the same input.
# Under -ffast-math the reciprocal id=1/d and the roundf boundary drifted,
# tipping ~1-code-per-block differences vs ggml on identical input while the
# fp16 block scale still matched. Build it fast-math-free, like quants.o.
QUANTIZE_OBJ = $(OBJDIR)/quantize.o

# These used to carry FORCE (rebuild on every invocation) because the
# requested CC/CFLAGS were not encoded anywhere, and reusing a native
# developer object in a portable release or cross build would be a
# correctness bug. The object directory is now keyed by CC and CFLAGS (see
# the object layer below; QUANTS_CFLAGS derives from CFLAGS), so a different
# flag set is a different object and FORCE, which had each of `make test`'s
# 17 sub-makes recompile both objects and relink `runner`, is not needed.
#
# .DEFAULT_GOAL, because this rule sits ABOVE `runner` and a rule is make's
# default goal by position: without the pin, bare `make` built ONE object
# file and exited 0. Local flows always name a target, so only CI's bare
# `make` jobs (windows smoke, consumer-compatibility) caught it.
.DEFAULT_GOAL := runner
$(QUANTS_OBJ): src/quants.c $(HDR)
	mkdir -p $(dir $@)
	$(CC) $(QUANTS_CFLAGS) -I src -c src/quants.c -o $@

$(QUANTIZE_OBJ): src/quantize.c $(HDR)
	mkdir -p $(dir $@)
	$(CC) $(QUANTS_CFLAGS) -I src -c src/quantize.c -o $@

# ---------------------------------------------------------------- object layer
# Every test binary used to recompile the engine from source (model.c alone
# 49 times per `make test`, ~13 of its 19 CI minutes). The shared engine
# sources are now compiled ONCE with exactly $(CFLAGS) and the standard test
# rules link the objects: the same flags, the same code, one compile.
# What stays on sources, deliberately: `runner` itself and its -D test
# variants (SRC), the sanitizer builds, the QUANTS_CFLAGS kernels, the tray
# core (-DRUNNER_TEST_TRAY_HTTP) and the -O0 interface stub -- every rule
# whose flags differ from $(CFLAGS) keeps compiling what it links.
#
# The objects live under a directory keyed by CC and CFLAGS (a cksum of the
# exact command prefix), so a native developer build, a portable release
# build and a cross build never share an object: a different flag set is a
# different directory, not a timestamp comparison that GNU make 3.81 (the
# macOS default) resolves to whole seconds. The hazard the quant objects'
# old FORCE guarded is answered by construction rather than by rebuilding
# them in every one of `make test`'s 17 sub-makes.
ENGINE_OBJ_NAMES = gguf compat instances tokenizer model sample vramreg template \
                   jsonmode schema schema_tools engine json envelope ed25519 ecdsa oms http \
                   registry provenance respstore receipts watermark kvsnap session scheduler completion api_responses api_anthropic server hfhub
ENGINE_OBJ = $(ENGINE_OBJ_NAMES:%=$(OBJDIR)/%.o) $(MLDSA_OBJ)
# the schema compiler and validator, and the native tool grammars built on them
SCHEMA_SRC = src/schema.c src/schema_tools.c
SCHEMA_OBJ = $(OBJDIR)/schema.o $(OBJDIR)/schema_tools.o
ifeq ($(GPU_SRC),src/metal.m)
GPU_OBJ = $(OBJDIR)/metal.o
else
GPU_OBJ = $(OBJDIR)/cuda.o
endif

$(OBJDIR)/%.o: src/%.c $(HDR)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I src -c $< -o $@

# ML-DSA-44 (FIPS 204) receipt signatures: the wrapper plus the vendored
# pq-crystals reference under src/mldsa/ (integer-only code, so the float
# flags of the engine objects do not touch its results)
MLDSA_SRC = src/mldsa.c $(wildcard src/mldsa/*.c)
MLDSA_OBJ = $(OBJDIR)/mldsa.o $(patsubst src/mldsa/%.c,$(OBJDIR)/mldsa_%.o,$(wildcard src/mldsa/*.c))
$(OBJDIR)/mldsa_%.o: src/mldsa/%.c $(wildcard src/mldsa/*.h) src/mldsa.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I src -c $< -o $@

# cuda.c embeds kernels_ptx.h, metal.m the shader headers: a regenerated
# header must rebuild the backend object (the same rule `runner` carries)
$(OBJDIR)/cuda.o: src/cuda.c $(HDR) src/kernels_ptx.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I src -c $< -o $@
$(OBJDIR)/metal.o: src/metal.m $(HDR) src/kernels_metal.h src/kernels_tensor_metal.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I src -c $< -o $@

SRC = src/gguf.c src/compat.c $(QUANTS_OBJ) src/instances.c src/tokenizer.c src/model.c src/sample.c src/dpo.c \
      src/vramreg.c src/npy.c \
      src/template.c src/jsonmode.c $(SCHEMA_SRC) $(QUANTIZE_OBJ) src/engine.c src/decide.c src/clef.c src/json.c src/envelope.c src/ed25519.c $(MLDSA_SRC) src/ecdsa.c src/oms.c src/http.c src/registry.c src/provenance.c src/lineage.c src/respstore.c src/receipts.c src/watermark.c src/kvsnap.c src/session.c src/scheduler.c src/completion.c src/api_responses.c src/api_anthropic.c src/server.c src/hfhub.c src/bundle.c \
      src/main.c $(GPU_SRC) $(TRAY_SRC)

# Cross-compile a static riscv64 binary with zig (no toolchain install;
# https://ziglang.org, `ZIG=path/to/zig`): the bit-exact configuration from
# docs/portable-bitexact-2026-09-05.md, strict float, portable math and
# canonical kernels, CUDA backend compiled in but dormant, tray stubbed.
# Runs under qemu-riscv64 (user mode) or on an RVA20+ board; not a release
# target. `make cross-riscv64 && qemu-riscv64 ./runner-riscv64 --help`.
ZIG ?= zig
CROSS_TARGET ?= riscv64-linux-musl
CROSS_CFLAGS = -O3 -fno-fast-math -ffp-contract=off -std=gnu11 -Wno-unused-parameter \
               -DRUNNER_GPU_CUDA -DRUNNER_PORTABLE_MATH -DRUNNER_CANON_KERNELS
CROSS_SRC = $(filter-out $(GPU_SRC) $(TRAY_SRC),$(filter %.c,$(SRC))) \
            src/quants.c src/quantize.c src/cuda.c src/tray.c src/tray_stub.c
cross-riscv64:
	@rm -rf .build/cross-$(CROSS_TARGET) && mkdir -p .build/cross-$(CROSS_TARGET)
	@for f in $(CROSS_SRC); do \
	  $(ZIG) cc -target $(CROSS_TARGET) $(CROSS_CFLAGS) -I src -c $$f \
	    -o .build/cross-$(CROSS_TARGET)/$$(basename $$f .c).o || exit 1; done
	$(ZIG) cc -target $(CROSS_TARGET) -static .build/cross-$(CROSS_TARGET)/*.o \
	  -o runner-$(CROSS_TARGET:%-linux-musl=%) -lm -lpthread
	@echo "runner-$(CROSS_TARGET:%-linux-musl=%): $$(wc -c < runner-$(CROSS_TARGET:%-linux-musl=%)) bytes"
.PHONY: cross-riscv64

# kernels_ptx.h is embedded into the binary by cuda.c — a pull that changes
# ONLY the regenerated PTX header must rebuild, or benchmarks silently run
# yesterday's kernels (this bit a publication run on 2026-07-29).
# Reuse the flag-keyed objects that focused tests already link. Special
# quantization and crypto objects keep their existing compilation rules.
RUNNER_OBJ = $(patsubst src/%.c,$(OBJDIR)/%.o,$(filter-out $(MLDSA_SRC),$(filter %.c,$(SRC)))) \
             $(patsubst src/%.m,$(OBJDIR)/%.o,$(filter %.m,$(SRC))) \
             $(QUANTS_OBJ) $(QUANTIZE_OBJ) $(MLDSA_OBJ)
# The Windows executable's resources: the UTF-8 application manifest. A build
# without it is a runner that reads a non-ASCII -p prompt in the ANSI code
# page (cp1252 on a Western European system, "?" for what that cannot
# spell), so a missing windres stops the build instead of quietly dropping
# the manifest.
ifeq ($(BUILD_OS),Windows)
RUNNER_RES_OBJ = $(OBJDIR)/win_manifest.o
RUNNER_OBJ += $(RUNNER_RES_OBJ)
$(RUNNER_RES_OBJ): src/runner.rc src/runner.manifest
	@mkdir -p $(dir $@)
	@command -v $(WINDRES) > /dev/null 2>&1 || { \
	  echo "error: '$(WINDRES)' not found. It embeds runner.exe's UTF-8 manifest (src/runner.rc)," >&2; \
	  echo "       without which non-ASCII prompts, paths and environment values are misread." >&2; \
	  echo "       Install binutils (MSYS2 UCRT64: pacman -S mingw-w64-ucrt-x86_64-binutils)" >&2; \
	  echo "       or name the resource compiler with WINDRES=<path>." >&2; exit 1; }
	$(WINDRES) -I src -O coff -i src/runner.rc -o $@
# The same executable without that manifest: what an older Windows, or one
# whose manifest was stripped, runs, in the machine's ANSI code page.
# tests/test_utf8_io.py holds it to the same tokens, paths and cache as
# runner.exe, which is what tests compat.c's own wide reads.
RUNNER_NO_MANIFEST = runner-no-manifest.exe
$(RUNNER_NO_MANIFEST): $(filter-out $(RUNNER_RES_OBJ),$(RUNNER_OBJ)) $(HDR) src/kernels_ptx.h
	$(CC) $(CFLAGS) $(filter-out $(RUNNER_RES_OBJ),$(RUNNER_OBJ)) -o $@ $(LDFLAGS)
else
RUNNER_NO_MANIFEST =
endif
$(OBJDIR)/%.o: src/%.m $(HDR)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I src -c $< -o $@
# The executable is keyed too: switching back from T3 must not keep a newer
# T3 runner merely because the default objects are older. Comparison keeps no-op
# builds cheap while selecting the exact configured executable on every call.
# The copy lands under a temporary name and is renamed over the old file, so
# it is a new file and never the old one rewritten: on macOS, a process still
# running from the old file (a server left up) gets every later launch of a
# file rewritten under it SIGKILLed at exec, which every test then reports as
# "runner exited during startup", returncode -9, no log.
$(OBJDIR)/$(RUNNER_EXE): $(RUNNER_OBJ) $(HDR) src/kernels_ptx.h src/kernels_tensor_metal.h
	$(CC) $(CFLAGS) $(RUNNER_OBJ) -o $@ $(LDFLAGS)
runner: $(OBJDIR)/$(RUNNER_EXE) FORCE
	@$(PYTHON) -c "import filecmp,os,shutil; src='$(OBJDIR)/$(RUNNER_EXE)'; dst='$(RUNNER_EXE)'; same=os.path.exists(dst) and filecmp.cmp(src,dst,shallow=False); same or (shutil.copy2(src,dst+'.tmp'), os.replace(dst+'.tmp',dst))"

# Local negative controls for scripts/write-stall.py. These compile from the
# same sources and embedded kernels as runner; they are not release artifacts.
runner-no-write-timeout: $(SRC) $(HDR) src/kernels_ptx.h
	$(CC) $(CFLAGS) -DRUNNER_TEST_NO_WRITE_TIMEOUT $(SRC) -o $@ $(LDFLAGS)

runner-sigpipe-default: $(SRC) $(HDR) src/kernels_ptx.h
	$(CC) $(CFLAGS) -DRUNNER_TEST_SIGPIPE_DEFAULT $(SRC) -o $@ $(LDFLAGS)

# Every translation unit, including quantization, must carry instrumentation.
# An optimized object linked into this binary silently loses its sanitizer gate.
SAN_FLAGS = -O0 -g -fsanitize=address,undefined -fno-sanitize-recover=undefined \
            -fno-omit-frame-pointer -fno-fast-math -std=gnu11 -Wall $(GPU_BACKEND_DEF)
SAN_SRC = $(filter-out $(QUANTS_OBJ) $(QUANTIZE_OBJ),$(SRC)) src/quants.c src/quantize.c
debug: $(SAN_SRC) $(HDR)
	$(CC) $(SAN_FLAGS) $(SAN_SRC) -o runner-debug $(LDFLAGS)

# $(CFLAGS), not a hand-rolled flag list: schema bounds compile in the same
# -ffast-math configuration as the shipped binary. Building this test without
# those flags once gated behavior in a configuration that does not ship.
$(TEST_JSON_SCHEMA): tests/test_json_schema.c $(OBJDIR)/json.o $(OBJDIR)/jsonmode.o $(SCHEMA_OBJ) $(HDR)
	$(CC) $(CFLAGS) -I src tests/test_json_schema.c $(OBJDIR)/json.o $(OBJDIR)/jsonmode.o $(SCHEMA_OBJ) -o $@ -lm

$(TEST_SVAL_WALK): tests/test_sval_walk.c $(OBJDIR)/json.o $(OBJDIR)/jsonmode.o $(SCHEMA_OBJ) $(HDR)
	$(CC) $(CFLAGS) -I src tests/test_sval_walk.c $(OBJDIR)/json.o $(OBJDIR)/jsonmode.o $(SCHEMA_OBJ) -o $@ -lm

# quants.c is needed for the ggml_type_* helpers gguf.c links against; CFLAGS
# (not the plainer flags above) so the AVX2 paths match a real build
TEST_TOK_SRC = tests/test_tokenizer.c $(OBJDIR)/gguf.o $(OBJDIR)/tokenizer.o $(OBJDIR)/compat.o $(QUANTS_OBJ)
$(TEST_TOKENIZER): $(TEST_TOK_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_TOK_SRC) -o $@ -lm

# the two merge implementations must agree on every input; see the test's header
TEST_TOK_MERGE = $(TEST_BATCH:test-batch%=test-tokenizer-merge%)
TEST_TOK_MERGE_SRC = tests/test_tokenizer_merge.c $(OBJDIR)/gguf.o $(OBJDIR)/tokenizer.o \
                     $(OBJDIR)/compat.o $(QUANTS_OBJ)
$(TEST_TOK_MERGE): $(TEST_TOK_MERGE_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_TOK_MERGE_SRC) -o $@ -lm

$(TEST_GGUF_SPLIT): tests/test_gguf_split_load.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o $(QUANTS_OBJ) $(HDR)
	$(CC) $(CFLAGS) -I src tests/test_gguf_split_load.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o $(QUANTS_OBJ) -o $@ $(LDFLAGS)

# difftok: tokenizer differential harness. Not part of `make test` -- it needs a
# real multi-GB model GGUF, which models/ is gitignored for. scripts/difftok.py
# builds it on demand and compares against the HuggingFace reference.
DIFFTOK_SRC = tests/difftok.c $(OBJDIR)/gguf.o $(OBJDIR)/tokenizer.o $(OBJDIR)/compat.o $(QUANTS_OBJ)
$(DIFFTOK): $(DIFFTOK_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(DIFFTOK_SRC) -o $@ -lm

TEST_TMPL_SRC = tests/test_template.c $(OBJDIR)/gguf.o $(OBJDIR)/tokenizer.o $(OBJDIR)/template.o $(SCHEMA_OBJ) $(OBJDIR)/jsonmode.o \
                $(OBJDIR)/json.o $(OBJDIR)/compat.o $(QUANTS_OBJ)
$(TEST_TEMPLATE): $(TEST_TMPL_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_TMPL_SRC) -o $@ -lm

TEST_PROMPT_MARKS_SRC = tests/test_prompt_marks.c $(OBJDIR)/gguf.o $(OBJDIR)/tokenizer.o $(OBJDIR)/template.o $(SCHEMA_OBJ) $(OBJDIR)/jsonmode.o \
                $(OBJDIR)/json.o $(OBJDIR)/compat.o $(QUANTS_OBJ)
$(TEST_PROMPT_MARKS): $(TEST_PROMPT_MARKS_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_PROMPT_MARKS_SRC) -o $@ -lm

# the strict tool envelope is only meaningful if the schema engine enforces
# it, so schema.c/jsonmode.c compile in and the tests drive the real validator
TEST_TOOLS_SRC = tests/test_tools.c $(OBJDIR)/gguf.o $(OBJDIR)/tokenizer.o $(OBJDIR)/template.o \
                 $(SCHEMA_OBJ) $(OBJDIR)/jsonmode.o $(OBJDIR)/json.o $(OBJDIR)/compat.o $(QUANTS_OBJ)
$(TEST_TOOLS): $(TEST_TOOLS_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_TOOLS_SRC) -o $@ -lm

# sampler presets and the greedy/penalty contract need no model, so the test
# links src/sample.c alone
$(TEST_SAMPLER): tests/test_sampler.c $(OBJDIR)/sample.o $(HDR)
	$(CC) $(CFLAGS) -I src tests/test_sampler.c $(OBJDIR)/sample.o -o $@ -lm

# compiles src/json.c directly into the test with instrumented allocators
$(TEST_JSON_OOM): tests/test_json_oom.c src/json.c src/json.h
	$(CC) $(CFLAGS) -I src tests/test_json_oom.c -o $@ -lm

# compiles src/tokenizer.c into the test with instrumented allocators; gguf.c
# and friends link normally so their allocations stay outside the failure window
TEST_TOK_OOM_SRC = tests/test_tokenizer_oom.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o $(QUANTS_OBJ)
$(TEST_TOKENIZER_OOM): $(TEST_TOK_OOM_SRC) src/tokenizer.c $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_TOK_OOM_SRC) -o $@ -lm
# The code-point classes the BPE pre-tokenizers split on, against the Unicode
# charts; it includes src/tokenizer.c to reach the static classifiers.
TEST_TOK_CLASSES_SRC = tests/test_tokenizer_classes.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o $(QUANTS_OBJ)
$(TEST_TOK_CLASSES): $(TEST_TOK_CLASSES_SRC) src/tokenizer.c $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_TOK_CLASSES_SRC) -o $@ -lm

# same shape for the renderer: json.c and template.c are compiled into the test
# with instrumented allocators, tokenizer.c/gguf.c link normally
TEST_TMPL_OOM_SRC = tests/test_template_oom.c $(OBJDIR)/tokenizer.o $(OBJDIR)/gguf.o \
                    $(SCHEMA_OBJ) $(OBJDIR)/jsonmode.o $(OBJDIR)/compat.o $(QUANTS_OBJ)
$(TEST_TEMPLATE_OOM): $(TEST_TMPL_OOM_SRC) src/template.c src/json.c $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_TMPL_OOM_SRC) -o $@ -lm

# Every family's render contract at once: empty conversations, tools and
# thinking reach, the tool protocol, and capacity at every buffer size.
TEST_TMPL_FAMILIES_SRC = tests/test_template_families.c $(OBJDIR)/gguf.o $(OBJDIR)/tokenizer.o $(OBJDIR)/template.o \
                $(SCHEMA_OBJ) $(OBJDIR)/jsonmode.o $(OBJDIR)/json.o $(OBJDIR)/compat.o $(QUANTS_OBJ)
$(TEST_TEMPLATE_FAMILIES): $(TEST_TMPL_FAMILIES_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_TMPL_FAMILIES_SRC) -o $@ -lm

# schema.c, schema_tools.c and json.c compile into the test: enum/const
# literals are serialised through jv_dump, so builder failures are schema
# failure paths
$(TEST_SCHEMA_OOM): tests/test_schema_oom.c $(SCHEMA_OBJ) $(OBJDIR)/json.o $(OBJDIR)/jsonmode.o $(HDR)
	$(CC) $(CFLAGS) -I src tests/test_schema_oom.c $(OBJDIR)/jsonmode.o -o $@ -lm

# shared model weights: needs the real model + backend, so it links the same
# sources the runner does minus the CLI/server front end
TEST_SHARED_SRC = tests/test_shared_weights.c src/gguf.c src/compat.c \
                  $(QUANTS_OBJ) src/model.c src/vramreg.c $(GPU_SRC)
$(TEST_SHARED): $(TEST_SHARED_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_SHARED_SRC) -o $@ $(LDFLAGS)

# --mlock over the real load path: which ranges a load wires, the rollback of
# a lock that fails halfway, and a shared lock's lifetime, measured against
# the kernel's locked-memory limit. The fixture's per-layer token table sits
# between hot ranges, several pages wide at either page size.
TEST_MLOCK_SRC = tests/test_mlock.c src/gguf.c src/compat.c \
                 $(QUANTS_OBJ) src/model.c src/vramreg.c $(GPU_SRC)
$(TEST_MLOCK): $(TEST_MLOCK_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_MLOCK_SRC) -o $@ $(LDFLAGS)
test-es-mlock.gguf: scripts/make-test-model.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-model.py --eseries 3,256 test-es-mlock.gguf

# ASAN_MODEL selects the fixture for the sharing / split / no-identity gates.
# It was UNSET, so all three silently fell back to the 370 KB test.gguf —
# which is physically incapable of moving a VRAM budget, overflowing a
# file-size field, or forcing an eviction. The gates were not weak; their input
# was, and a green run said nothing about that. Prefer a real model when one is
# on the box; fall back to the fixture otherwise, and SAY WHICH either way, so
# a pass is never mistaken for coverage it did not have.
ASAN_MODEL ?= $(firstword $(wildcard models/SmolLM2-135M-Instruct-Q8_0.gguf \
                                     models/e2b-q40.gguf))

.PHONY: fixture-scale-note
fixture-scale-note:
	@if [ -z "$(ASAN_MODEL)" ]; then \
	  echo "note: gates below run against the 370 KB test.gguf — too small to"; \
	  echo "      exercise VRAM budgets, eviction or file-size limits. Set"; \
	  echo "      ASAN_MODEL=<real .gguf> to make those contracts testable."; \
	else \
	  echo "note: gates below run against $(ASAN_MODEL)"; \
	fi

# same test under ASan/UBSan: the free-exactly-once half of it only fails
# loudly here. Kept out of `make test` because a sanitized model load is slow.
test-shared-asan: $(TEST_SHARED_SRC) $(HDR) test.gguf fixture-scale-note
	$(CC) -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer \
	    -std=gnu11 -Wall -I src $(filter-out $(QUANTS_OBJ),$(TEST_SHARED_SRC)) src/quants.c -o test-shared-asan-bin $(LDFLAGS)
	RUNNER_TEST_GPU_OFF=1 LSAN_OPTIONS=suppressions=tests/lsan.supp \
	    ./test-shared-asan-bin $(ASAN_MODEL)

# Prove the sharing gate is not vacuous, at fixture scale and on any backend.
# With the file identity unavailable both instances load privately, so
# test-shared-weights MUST go red — and it must go red for the sharing reason,
# not something incidental. A green run here means the gate can no longer
# detect lost sharing, which is the state it was in for its whole life before
# 2026-08-04 (see the CHANGELOG entries for the shared-weights split fix).
test-shared-noid: $(TEST_SHARED) fixture-scale-note
	@set -e; \
	if RUNNER_TEST_NO_FILE_ID=1 ./$(TEST_SHARED) $(ASAN_MODEL) > shared-noid.out 2>&1; then \
		echo "FAIL: test-shared-weights passed with no file identity —"; \
		echo "      the sharing gate cannot detect lost sharing and is vacuous."; \
		cat shared-noid.out; exit 1; \
	fi; \
	grep -q "cannot be keyed by file identity" shared-noid.out || { \
		echo "FAIL: nothing on stderr named the lost file identity"; \
		cat shared-noid.out; exit 1; }; \
	grep -q "instances share one layer array" shared-noid.out || { \
		echo "FAIL: red, but not for the lost-sharing reason"; \
		cat shared-noid.out; exit 1; }; \
	echo "shared-weights no-identity gate ok (red as required, and said why)"

# file identity at real-checkpoint size: model_file_identity() must key a
# 5 GB file, because losing the identity is what silently unshares weights
# (the fixture-scale gates can never see the >2 GB stat() cliff that did it)
TEST_FILE_ID = $(TEST_BATCH:test-batch%=test-file-identity%)
TEST_FILE_ID_SRC = tests/test_file_identity.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o \
                   $(QUANTS_OBJ) $(OBJDIR)/model.o $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_FILE_ID): $(TEST_FILE_ID_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_FILE_ID_SRC) -o $@ $(LDFLAGS)

# A bare `runner` in a terminal starts the tray; a bare `runner` anywhere
# else must NOT — a GUI event loop on a piped stdin would hang every script
# that ever probes the binary, including this test. Piping stdin here is
# therefore both the test setup and the property under test. --no-tray is the
# documented spelling of the same suppression and must stay a known flag.
# The binary must carry the kernels currently in the tree. check-generated.py
# compares src/kernels_metal.h to src/kernels.metal, but neither can see a
# STALE BINARY, and a kernel measured against one looks exactly like an honest
# result: on 2026-08-07 a q4_K change measured as no effect and a q5_K change
# as +4.6%, both against builds that did not contain them. make rebuilds
# correctly on a header change; the hazard is an A/B whose two builds land in
# the same second, where make's newer-than test keeps the old binary.
# Skips cleanly on a build with no embedded shader source (CUDA/CPU-only).
test-shader-embed: runner
	@set -e; \
	command -v $(PYTHON) >/dev/null 2>&1 || { \
		echo "shader embed: skip (no $(PYTHON) on this box)"; exit 0; }; \
	caps=$$(./runner --caps 2>/dev/null); \
	have=$$(printf '%s' "$$caps" | $(PYTHON) -c "import sys,json; g=json.load(sys.stdin).get('gpu') or {}; print(g.get('shader_source_sha256') or '')"); \
	if [ -z "$$have" ]; then echo "shader embed: skip (no embedded shader source in this build)"; exit 0; fi; \
	want=$$($(PYTHON) -c "import hashlib;print(hashlib.sha256(open('src/kernels.metal',encoding='utf-8').read().encode()).hexdigest())"); \
	if [ "$$have" != "$$want" ]; then \
		echo "FAIL: runner was built from different Metal shaders than src/kernels.metal"; \
		echo "  binary: $$have"; echo "  source: $$want"; \
		echo "  re-run scripts/embed-metal.py and rebuild"; exit 1; fi; \
	echo "shader embed ok (binary carries src/kernels.metal, $${have}...)"

test-bare-invocation: runner
	@set -e; \
	rc=0; out=$$(echo "" | ./runner 2>&1) || rc=$$?; \
	echo "$$out" | grep -q "usage:" || { \
		echo "FAIL: bare runner with piped stdin did not print usage"; exit 1; }; \
	[ $$rc -ne 0 ] || { echo "FAIL: bare runner exited 0 without doing anything"; exit 1; }; \
	out=$$(echo "" | ./runner --no-tray 2>&1) || true; \
	echo "$$out" | grep -q "unknown option" && { \
		echo "FAIL: --no-tray is not a recognized flag"; exit 1; }; \
	echo "$$out" | grep -q "usage:" || { \
		echo "FAIL: bare --no-tray did not print usage"; exit 1; }; \
	echo "bare invocation ok (non-tty gets usage, --no-tray recognized)"

# RI-6: requested help is an ANSWER, not a diagnostic. `runner --help` is how
# a script or a package manifest discovers this binary's interface, so it goes
# to stdout and exits 0, stderr stays reserved for things that went wrong, and
# the help flags appear in the list they head. The parity gate below then keeps
# that list and the manual (MANUAL.md) from drifting apart.
test-help-interface: runner
	@set -e; \
	out=$$(./runner --help 2>/dev/null); \
	echo "$$out" | grep -q "usage:" || { \
		echo "FAIL: --help wrote no usage to stdout"; exit 1; }; \
	err=$$(./runner --help 2>&1 >/dev/null); \
	[ -z "$$err" ] || { \
		echo "FAIL: --help wrote to stderr: $$err"; exit 1; }; \
	./runner --help >/dev/null 2>&1 || { \
		echo "FAIL: --help exited non-zero"; exit 1; }; \
	echo "$$out" | grep -q -- "-h, --help" || { \
		echo "FAIL: --help omits -h/--help from its own option list"; exit 1; }; \
	err=$$(./runner --definitely-not-a-flag 2>&1 >/dev/null || true); \
	echo "$$err" | grep -q "unknown option" || { \
		echo "FAIL: an unknown option did not report on stderr"; exit 1; }; \
	rc=0; ./runner --definitely-not-a-flag >/dev/null 2>&1 || rc=$$?; \
	[ $$rc -ne 0 ] || { \
		echo "FAIL: an unknown option exited 0"; exit 1; }; \
	$(PYTHON) scripts/help-parity.py --binary ./$(RUNNER_EXE) --readme MANUAL.md; \
	echo "help interface ok (stdout, exit 0, flags listed, MANUAL.md in parity)"

# split-guard harness: same link as the shared-weights test — the guard lives
# in the GPU registry, so it needs the real backend
# NOT `test-split-guard%`: on POSIX $(TEST_BATCH) is `test-batch`, so that
# substitution produced `test-split-guard` — colliding with the .PHONY test
# target of the same name below. make then dropped this compile recipe
# ("overriding commands"), reported a circular self-dependency, and the guard
# binary was never built. The guard whose comment says "delete the guard and
# this goes red" could not go red, and `make test` never referenced it at all.
# Windows was unaffected only because `test-batch.exe` yields a distinct name.
TEST_SPLIT_GUARD = $(TEST_BATCH:test-batch%=test-split-guard-bin%)
TEST_SPLIT_GUARD_SRC = tests/test_split_guard.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o \
                       $(QUANTS_OBJ) $(OBJDIR)/model.o $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_SPLIT_GUARD): $(TEST_SPLIT_GUARD_SRC) $(HDR)
	$(CC) $(CFLAGS) $(GPU_BACKEND_DEF) -I src $(TEST_SPLIT_GUARD_SRC) -o $@ $(LDFLAGS)

# The split guard must be able to fire: a no-identity load of an already-
# resident path, forced to a different split, must be reported loudly. The
# harness self-skips (exit 0, says so) without a GPU backend; when it does
# run, the report line is the gate — delete the guard and this goes red.
test-split-guard: $(TEST_SPLIT_GUARD) test.gguf
	@set -e; \
	./$(TEST_SPLIT_GUARD) $(ASAN_MODEL) > split-guard.out 2>&1; \
	if grep -q "^skip:" split-guard.out; then cat split-guard.out; exit 0; fi; \
	grep -q "re-decided the CPU/GPU split without a file identity" split-guard.out || { \
		echo "FAIL: forced split disagreement produced no loud report"; \
		cat split-guard.out; exit 1; }; \
	echo "split guard ok (no-identity split disagreement is reported loudly)"

# batched decode: same sources as the shared-weights test (real model +
# backend), because the property under test is a backend property
TEST_BATCH_SRC = tests/test_batch.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o \
                 $(QUANTS_OBJ) $(OBJDIR)/model.o $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_BATCH): $(TEST_BATCH_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_BATCH_SRC) -o $@ $(LDFLAGS)

# the loopback-only bind. Links nothing: it reads src/server.c and src/main.c
# and interrogates the built ./runner, so a bind address introduced anywhere --
# constant, flag, env var -- trips it. Depends on `runner` because the CLI half
# of the check needs the shipped binary rather than a comment about it.
$(TEST_BIND): tests/test_bind.c src/server.c src/main.c
	$(CC) $(CFLAGS) -I src tests/test_bind.c -o $@

$(TEST_HOST_HEADER): tests/test_host_header.c $(OBJDIR)/http.o $(OBJDIR)/json.o $(OBJDIR)/compat.o $(HDR)
	$(CC) $(CFLAGS) -I src tests/test_host_header.c $(OBJDIR)/http.o $(OBJDIR)/json.o \
		$(OBJDIR)/compat.o -o $@ $(LDFLAGS)

# forkable KV prefixes: needs the real model, the real tokenizer and the real
# engine, because the property under test is that a forked cache produces the
# same logits the model would have produced by prefilling
TEST_PREFIX_SRC = tests/test_prefix.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o $(QUANTS_OBJ) \
                  $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/sample.o $(OBJDIR)/jsonmode.o \
                  $(SCHEMA_OBJ) $(OBJDIR)/json.o $(OBJDIR)/engine.o $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_PREFIX): $(TEST_PREFIX_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_PREFIX_SRC) -o $@ $(LDFLAGS)

# recurrent-state cache seam (SSM tracer 4): same real-object link set as the
# prefix gate, because the property under test is the same — bit-identical
# logits — but on the recurrent (SSM) fold that a rewind must snapshot/restore.
TEST_RECURRENT_SRC = tests/test_recurrent_rewind.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o $(QUANTS_OBJ) \
                  $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/sample.o $(OBJDIR)/jsonmode.o \
                  $(SCHEMA_OBJ) $(OBJDIR)/json.o $(OBJDIR)/engine.o $(OBJDIR)/vramreg.o $(GPU_OBJ)
TEST_PENALTY_WINDOW_SRC = tests/test_penalty_window.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o $(QUANTS_OBJ) \
                  $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/sample.o $(OBJDIR)/jsonmode.o \
                  $(SCHEMA_OBJ) $(OBJDIR)/json.o $(OBJDIR)/engine.o $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_PENALTY_WINDOW): $(TEST_PENALTY_WINDOW_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_PENALTY_WINDOW_SRC) -o $@ $(LDFLAGS)

# YaRN's frequency table against the publishers' reference ramps (gpt-oss
# unrounded, transformers' default rounded); the same link set as the
# penalty-window gate, since model_load is the interface under test
TEST_ROPE_YARN_SRC = tests/test_rope_yarn.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o $(QUANTS_OBJ) \
                  $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/sample.o $(OBJDIR)/jsonmode.o \
                  $(SCHEMA_OBJ) $(OBJDIR)/json.o $(OBJDIR)/engine.o $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_ROPE_YARN): $(TEST_ROPE_YARN_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_ROPE_YARN_SRC) -o $@ $(LDFLAGS)

# per-family attention scales against the published configs (Gemma 3 27B's
# query_pre_attn_scalar is not its head width); same link set as above
TEST_ATTN_SCALE_SRC = $(TEST_ROPE_YARN_SRC:tests/test_rope_yarn.c=tests/test_attn_scale.c)
$(TEST_ATTN_SCALE): $(TEST_ATTN_SCALE_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_ATTN_SCALE_SRC) -o $@ $(LDFLAGS)

# the CPU dot route gate: dispatch of the one-row and small-batch matvec, and
# a same-process ratio canary for the multi-column dot (tests/test_dot_route.c)
TEST_DOT_ROUTE_SRC = $(TEST_ROPE_YARN_SRC:tests/test_rope_yarn.c=tests/test_dot_route.c)
$(TEST_DOT_ROUTE): $(TEST_DOT_ROUTE_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_DOT_ROUTE_SRC) -o $@ $(LDFLAGS)

$(TEST_RECURRENT): $(TEST_RECURRENT_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_RECURRENT_SRC) -o $@ $(LDFLAGS)

# Request cancellation at the engine's prefill/decode poll boundaries.  The
# scheduler is included so its private batched loop is exercised without
# widening that loop into a public test API; the generated recurrent fixture
# makes stale-fold reuse observable as byte divergence.
TEST_REQUEST_STOP_SRC = tests/test_request_stop.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o \
                  $(QUANTS_OBJ) $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/sample.o \
                  $(OBJDIR)/jsonmode.o $(SCHEMA_OBJ) $(OBJDIR)/json.o $(OBJDIR)/engine.o \
                  $(OBJDIR)/vramreg.o $(OBJDIR)/http.o $(GPU_OBJ)
$(TEST_REQUEST_STOP): $(TEST_REQUEST_STOP_SRC) src/scheduler.c $(HDR) test-ornith.gguf test-ornith-draft.gguf
	$(CC) $(CFLAGS) -I src $(TEST_REQUEST_STOP_SRC) -o $@ $(LDFLAGS)

# grammar fast-forward: same full-engine link as the prefix test — the gate
# is byte identity of a real constrained generation with the walk on and off
TEST_GRAMMAR_FF_SRC = tests/test_grammar_ff.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o \
                  $(QUANTS_OBJ) $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/sample.o \
                  $(OBJDIR)/jsonmode.o $(SCHEMA_OBJ) $(OBJDIR)/json.o $(OBJDIR)/engine.o \
                  $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_GRAMMAR_FF): $(TEST_GRAMMAR_FF_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_GRAMMAR_FF_SRC) -o $@ $(LDFLAGS)

# the decision record's legal mass on hand-written logits (one engine step
# under an enum): the same full-engine link as the fast-forward gate
TEST_CHOICE_MASS_SRC = tests/test_choice_mass.c $(filter-out tests/test_grammar_ff.c,$(TEST_GRAMMAR_FF_SRC))
$(TEST_CHOICE_MASS): $(TEST_CHOICE_MASS_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_CHOICE_MASS_SRC) -o $@ $(LDFLAGS)

# The prompt-lookup search pinned against hand-computed proposals (the
# absolute anchor of --draft-lookup; the CLI identity gate is in
# tests/test_draft_lookup.py). Links the engine, so it takes engine.c's deps.
TEST_LOOKUP_DRAFT_SRC = tests/test_lookup_draft.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o \
                  $(QUANTS_OBJ) $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/sample.o \
                  $(OBJDIR)/jsonmode.o $(SCHEMA_OBJ) $(OBJDIR)/json.o $(OBJDIR)/engine.o \
                  $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_LOOKUP_DRAFT): $(TEST_LOOKUP_DRAFT_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_LOOKUP_DRAFT_SRC) -o $@ $(LDFLAGS)

# the cross-process VRAM registry. Links only vramreg.c and compat.c: the
# free-VRAM figure arrives through a callback, so the whole module is drivable
# with synthetic numbers and the test needs no GPU, no model and no driver --
# which is what lets it run in CI.
$(TEST_VRAMREG): tests/test_vram_registry.c $(OBJDIR)/vramreg.o $(OBJDIR)/compat.o $(HDR)
	$(CC) $(CFLAGS) -I src tests/test_vram_registry.c $(OBJDIR)/vramreg.o $(OBJDIR)/compat.o -o $@ $(LDFLAGS)

# weight-upload progress policy (src/upload_report.h), header only, no device
$(TEST_UPLOAD_REPORT): tests/test_upload_report.c src/upload_report.h
	$(CC) $(CFLAGS) -I src tests/test_upload_report.c -o $@ $(LDFLAGS)

# q8 KV tolerance gate: needs the tokenizer too, because it teacher-forces a
# fixed piece of real text rather than synthetic token ids
TEST_KV_TOL_SRC = tests/test_kv_tol.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o $(QUANTS_OBJ) \
                  $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_KV_TOL): $(TEST_KV_TOL_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_KV_TOL_SRC) -o $@ $(LDFLAGS)

# the --kv fp4 cache codec (UE4M3 ceiling scale + E2M1 nibbles) against its
# inverse and the dequantised dot; the reference every backend's cache bytes
# are held to
$(TEST_KV_FP4): tests/test_kv_fp4.c $(QUANTS_OBJ) $(HDR)
	$(CC) $(CFLAGS) -I src tests/test_kv_fp4.c $(QUANTS_OBJ) -o $@ $(LDFLAGS)

# the CUDA KV store kernel against the CPU encoders, byte for byte (the twin
# of test-metal-kv-fp4): loads the driver API dynamically and JITs the
# committed PTX, so it builds without a CUDA toolkit and skips without a driver
TEST_CUDA_KVFP4 = $(TEST_BATCH:test-batch%=test-cuda-kvfp4%)
$(TEST_CUDA_KVFP4): tests/test_cuda_kvfp4.c src/kernels_ptx.h $(QUANTS_OBJ) $(HDR)
	$(CC) $(CFLAGS) -I src tests/test_cuda_kvfp4.c $(QUANTS_OBJ) -o $@ $(LDFLAGS)
test-cuda-kv-fp4: $(TEST_CUDA_KVFP4)
	./$(TEST_CUDA_KVFP4)

# SIMD (AVX2/NEON) dot and dequant kernels vs an independent double-precision
# reference; also pins q8_quant_row byte-identical to its scalar definition
TEST_QUANTS_SIMD_SRC = tests/test_quants_simd.c $(QUANTS_OBJ)
# The codebook decode primitives the CUDA kernels compile (src/iq_decode.h),
# executed on the host against the tables in quants_iq_grids.h, exhaustively.
# QUANTS_CFLAGS for the same reason as the quant tests: this is arithmetic
# the numerical identity rests on.
TEST_IQ_DECODE = $(TEST_BATCH:test-batch%=test-iq-decode%)
$(TEST_IQ_DECODE): tests/test_iq_decode.c src/iq_decode.h src/quants_iq_grids.h $(QUANTS_OBJ)
	$(CC) $(QUANTS_CFLAGS) -I src tests/test_iq_decode.c $(QUANTS_OBJ) -o $@ -lm -lpthread
# The same for the tensor-core stagers of the K-quants and legacy quants
# without a raw-register loader (src/kq_stage.h): every element of random and
# corner blocks against dequant_row(), bit for bit.
TEST_KQ_STAGE = $(TEST_BATCH:test-batch%=test-kq-stage%)
$(TEST_KQ_STAGE): tests/test_kq_stage.c src/kq_stage.h src/iq_decode.h $(QUANTS_OBJ)
	$(CC) $(QUANTS_CFLAGS) -I src tests/test_kq_stage.c $(QUANTS_OBJ) -o $@ -lm -lpthread

# QUANTS_CFLAGS, not CFLAGS: this test carries inline scalar REFERENCE
# implementations of the quant kernels, and a reference compiled under
# -ffast-math is checking the confined -fno-fast-math engine against a
# differently-rounded definition of itself. Caught on ARM the day the
# int8 activation quantizer (built on x86) met the fast-math confinement
# (built the same night): identical scalar source, two float regimes,
# constructed half-way ties diverged. The verifier must share the
# engine's float semantics or it verifies nothing.
$(TEST_QUANTS_SIMD): $(TEST_QUANTS_SIMD_SRC) $(HDR)
	$(CC) $(QUANTS_CFLAGS) -I src $(TEST_QUANTS_SIMD_SRC) -o $@ -lm -lpthread

# Canonical-order kernels (RUNNER_CANON_KERNELS, experiment R12.1): the
# SIMD kernel on this host must equal the tree definition bit for bit. quants.c
# is compiled here with the define, in the engine's strict float regime, not
# taken from the shared object (which is built without it).
TEST_CANON_KERNELS = $(TEST_BATCH:test-batch%=test-canon-kernels%)
$(TEST_CANON_KERNELS): tests/test_canon_kernels.c src/quants.c $(HDR)
	$(CC) $(QUANTS_CFLAGS) -ffp-contract=off -DRUNNER_CANON_KERNELS -I src \
	    tests/test_canon_kernels.c src/quants.c -o $@ -lm -lpthread

# discovery registry: pure-C, runs against a private HOME/APPDATA
TEST_INSTANCES_SRC = tests/test_instances.c $(OBJDIR)/instances.o $(OBJDIR)/json.o $(OBJDIR)/compat.o
$(TEST_INSTANCES): $(TEST_INSTANCES_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_INSTANCES_SRC) -o $@ -lm

# the same registry under injected allocation failure. instances.c is compiled
# INTO the test (macro-substituted allocators), so it is deliberately absent
# from the source list here.
TEST_INSTANCES_OOM = $(TEST_BATCH:test-batch%=test-instances-oom%)
$(TEST_INSTANCES_OOM): tests/test_instances_oom.c $(OBJDIR)/instances.o $(OBJDIR)/json.o $(OBJDIR)/compat.o $(HDR)
	$(CC) $(CFLAGS) -I src tests/test_instances_oom.c $(OBJDIR)/json.o $(OBJDIR)/compat.o -o $@ -lm

# Pure policy seam: simulates a Mac whose model is larger than one MTLBuffer
# but still inside the aggregate Metal working set.
$(TEST_METAL_ADMISSION): tests/test_metal_admission.c src/metal_admission.h
	$(CC) $(CFLAGS) -I src tests/test_metal_admission.c -o $@

# links the stub backend on EVERY platform: this gate tests the portable
# core (menu model, managed spawn/stop, quit semantics), not the GUI.
# ws2_32: the core's /v1/models enrichment uses sockets on Windows
ifeq ($(BUILD_OS),Windows)
TRAY_TEST_LIBS = -lws2_32
else
TRAY_TEST_LIBS =
endif
TEST_TRAY_CORE_SRC = tests/test_tray_core.c src/tray.c src/tray_stub.c \
                     src/instances.c src/json.c src/compat.c
$(TEST_TRAY_CORE): $(TEST_TRAY_CORE_SRC) $(HDR)
	$(CC) $(CFLAGS) -DRUNNER_TEST_TRAY_HTTP -I src $(TEST_TRAY_CORE_SRC) \
		-o $@ -lm $(TRAY_TEST_LIBS)

# The Win32 backend's menu, read back through GetMenuStringW: the test
# includes src/tray_win.c and links the core it calls into. Windows only;
# the POSIX suite skips it by name (see TEST_TRAY_WIN_RUN).
TEST_TRAY_WIN_SRC = tests/test_tray_win_menu.c src/tray.c src/instances.c \
                    src/json.c src/compat.c
$(TEST_TRAY_WIN): $(TEST_TRAY_WIN_SRC) src/tray_win.c $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_TRAY_WIN_SRC) -o $@ -lm \
		-lws2_32 -lpsapi -lshell32 -lgdi32 -lcomdlg32 -ladvapi32

# plat_utf8_startup and the console's output code page: set to UTF-8 for a
# console on stdout or stderr, left alone when both are redirected, restored
# at exit. Windows only, like the menu readback above (TEST_UTF8_CONSOLE_RUN).
$(TEST_UTF8_CONSOLE): tests/test_utf8_console.c $(OBJDIR)/compat.o $(HDR)
	$(CC) $(CFLAGS) -I src tests/test_utf8_console.c $(OBJDIR)/compat.o -o $@ $(LDFLAGS)

# TC tolerance gate: same shape as the q8-KV gate — teacher-forced logits,
# top-1 + bounded-deviation criteria, per (type, arch) via the model argument
# NaN/Inf detection for the gates, compiled fast-math-free (QUANTS_CFLAGS)
# because a fast-math TU folds every such test to false: see
# tests/finite_check.c.
FINITE_CHECK_OBJ = $(OBJDIR)/finite_check.o
$(FINITE_CHECK_OBJ): tests/finite_check.c tests/finite_check.h
	mkdir -p $(dir $@)
	$(CC) $(QUANTS_CFLAGS) -I tests -c tests/finite_check.c -o $@

TEST_TC_TOL_SRC = tests/test_tc_tol.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o $(QUANTS_OBJ) \
                  $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/vramreg.o $(GPU_OBJ) \
                  $(FINITE_CHECK_OBJ)
$(TEST_TC_TOL): $(TEST_TC_TOL_SRC) $(HDR) tests/finite_check.h
	$(CC) $(CFLAGS) -I src -I tests $(TEST_TC_TOL_SRC) -o $@ $(LDFLAGS)

# adaptation D3: finite-difference gradient gate over the full-coverage
# adapter fixture (rank-2 pairs on every hooked projection of every layer)
TEST_LORA_GRAD_SRC = tests/test_lora_grad.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o \
                  $(QUANTS_OBJ) $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/vramreg.o \
                  $(GPU_OBJ)
$(TEST_LORA_GRAD): $(TEST_LORA_GRAD_SRC) $(HDR) test.gguf test-lora.full.gguf test-q8.gguf test-lora-q8.full.gguf test-qk.gguf test-lora-qk.full.gguf test-nope.gguf test-lora-nope.full.gguf test-prope.gguf test-lora-prope.full.gguf test-prope-neox.gguf test-lora-prope-neox.full.gguf test-ornith.gguf test-lora-ornith.full.gguf test-clef.gguf test-lora-clef.full.gguf test-granite.gguf test-lora-granite.full.gguf test-muse.gguf test-lora-muse.full.gguf test-gemma3.gguf test-lora-gemma3.full.gguf test-gemma4.gguf test-lora-gemma4.full.gguf test-g4v.gguf test-lora-g4v.full.gguf test-es8.gguf test-lora-es8.full.gguf test-moe-train.moe4-train.gguf test-lora-moe4.full.gguf test-moe-train.moe4-train-nonorm.gguf test-lora-moe4n.full.gguf
	$(CC) $(CFLAGS) -I src $(TEST_LORA_GRAD_SRC) -o $@ $(LDFLAGS)

# Every fixture generator imports this module for the GGUF byte encoding, so a
# fixture is stale when it changes: it is a prerequisite beside each generator.
GGUF_FIXTURE_LIB = scripts/gguf_fixture.py

test-lora.full.gguf: test.gguf scripts/make-test-lora.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-lora.py test.gguf test-lora

# DPO gate: the pairwise objective and its gradient, checked three ways
# (independent loss formula, the grad = coeff*(gCE_w - gCE_l) identity, and a
# directional derivative against central differences of the actual loss)
# before any training wall clock is spent on the objective.
TEST_DPO_GRAD_SRC = tests/test_dpo_grad.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o \
                  $(QUANTS_OBJ) $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o \
                  $(OBJDIR)/dpo.o $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_DPO_GRAD): $(TEST_DPO_GRAD_SRC) $(HDR) test.gguf test-lora.full.gguf
	$(CC) $(CFLAGS) -I src $(TEST_DPO_GRAD_SRC) -o $@ $(LDFLAGS)

# R8.15 gate: the distillation objective (KL to a teacher's top-k plus a tail
# bucket) against the weighted cross-entropy it must equal for a one-hot
# teacher, an independent loss formula, and central differences of the loss,
# on the plain, logit-scaled (granite) and soft-capped (gemma4) heads
TEST_KD_GRAD_SRC = tests/test_kd_grad.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o \
                  $(QUANTS_OBJ) $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o \
                  $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_KD_GRAD): $(TEST_KD_GRAD_SRC) $(HDR) test.gguf test-lora.full.gguf test-granite.gguf test-lora-granite.full.gguf test-gemma4.gguf test-lora-gemma4.full.gguf
	$(CC) $(CFLAGS) -I src $(TEST_KD_GRAD_SRC) -o $@ $(LDFLAGS)

test-lora-q8.full.gguf: test-q8.gguf scripts/make-test-lora.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-lora.py test-q8.gguf test-lora-q8

# D8 slice 1: device transposed matvec vs the CPU trainer's chain (skips
# without a CUDA device; the three fixture runs cover F32, Q8_0 and BF16)
TEST_MVT_SRC = tests/test_mvt.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o $(QUANTS_OBJ) \
                  $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_MVT): $(TEST_MVT_SRC) $(HDR) test.gguf test-q8.gguf test-bf16.gguf
	$(CC) $(CFLAGS) -I src $(TEST_MVT_SRC) -o $@ $(LDFLAGS)

# R8.19: the training context's products for every weight type on random
# blocks: the transposed product the CPU chain's bytes, the forward product
# batch-invariant (skips without a CUDA device)
TEST_TRAIN_GPU_SRC = tests/test_train_gpu.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o $(QUANTS_OBJ) \
                  $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_TRAIN_GPU): $(TEST_TRAIN_GPU_SRC) $(HDR) tests/dn_ref.h tests/attn_ref.h
	$(CC) $(CFLAGS) -I src $(TEST_TRAIN_GPU_SRC) -o $@ $(LDFLAGS)

# The canonical-order forward (R8.7.1 slice 1). quants.c is compiled in HERE
# with RUNNER_CANON_KERNELS and the strict float regime -- not taken from the
# shared object, which is built without it and would answer a different
# question -- for the same reason test-canon-kernels does it that way.
TEST_MVCANON_SRC = tests/test_mvcanon.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o \
                   $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/vramreg.o \
                   $(GPU_OBJ)
$(TEST_MVCANON): $(TEST_MVCANON_SRC) src/quants.c $(HDR) test.gguf test-q8.gguf
	$(CC) $(CFLAGS) -ffp-contract=off -DRUNNER_CANON_KERNELS -I src \
	    $(TEST_MVCANON_SRC) src/quants.c -o $@ $(LDFLAGS)

test-yarn.gguf: scripts/make-test-model.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-model.py --yarn 32,4096 test-yarn.gguf

# qwen4exp (R4.26.1, experimental): hyper-connection streams, the Engram
# n-gram PLE block, Gated DeltaNet + full-attention blocks, routed + shared
# experts. Random weights; the gate is "loads, binds every tensor, the CPU
# forward runs and every logit is finite". The reference anchor is the
# llama.cpp and fp32 comparisons in R4.26.3, not this.
test-qwen4exp.gguf: scripts/make-test-qwen4exp.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-qwen4exp.py test-qwen4exp.gguf

# gemma3 at the 27B's block count and at the 1B's, both with heads narrower
# than n_embd / n_head so the two attention-scale rules disagree
test-gemma3-62.gguf: scripts/make-test-model.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-model.py --gemma3 --layers 62 --head-dim 8 test-gemma3-62.gguf
test-gemma3-26.gguf: scripts/make-test-model.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-model.py --gemma3 --layers 26 --head-dim 8 test-gemma3-26.gguf

test-qk.gguf: scripts/make-test-model.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-model.py --qk-norm test-qk.gguf

# NoPE every layer with a live attention temperature (floor 1): the LoRA
# backward must skip the rotation and apply the temperature's adjoint
test-nope.gguf: scripts/make-test-model.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-model.py test-nope.gguf --attn-knobs 1,0.5,1

test-lora-nope.full.gguf: test-nope.gguf scripts/make-test-lora.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-lora.py test-nope.gguf test-lora-nope

# R8.16: partial-dimension rope, half of each 16-wide head rotated, in the
# interleaved-pair (llama) and the split-half (neox, qwen3) layouts
test-prope.gguf: scripts/make-test-model.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-model.py test-prope.gguf --rope-dim 8

test-lora-prope.full.gguf: test-prope.gguf scripts/make-test-lora.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-lora.py test-prope.gguf test-lora-prope

test-prope-neox.gguf: scripts/make-test-model.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-model.py test-prope-neox.gguf --arch qwen3 --rope-dim 8

test-lora-prope-neox.full.gguf: test-prope-neox.gguf scripts/make-test-lora.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-lora.py test-prope-neox.gguf test-lora-prope-neox

# R8.9.4 dense-shape variants in the backward: granite's muP scalars
# (embedding, fixed attention, residual and divided logit scale, tied
# embeddings) and muse-glimmer's attention output gate, QK and sandwich
# norms, 3-local:1-global sliding pattern with NoPE full layers, scaled and
# softcapped logits
test-granite.gguf: scripts/make-test-model.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-model.py --granite test-granite.gguf

test-lora-granite.full.gguf: test-granite.gguf scripts/make-test-lora.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-lora.py test-granite.gguf test-lora-granite

test-muse.gguf: scripts/make-test-model.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-model.py --muse-glimmer test-muse.gguf

test-lora-muse.full.gguf: test-muse.gguf scripts/make-test-lora.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-lora.py test-muse.gguf test-lora-muse

# R17.3: router-sensitive MoE fixtures (written together by one run) and
# their attention-only adapters, for the LoRA backward's strict FD gate
test-moe-train.moe4-train.gguf: scripts/make-test-moe.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-moe.py test-moe-train > /dev/null
test-moe-train.moe4-train-nonorm.gguf: test-moe-train.moe4-train.gguf
test-lora-moe4.full.gguf: test-moe-train.moe4-train.gguf scripts/make-test-lora.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-lora.py test-moe-train.moe4-train.gguf test-lora-moe4 > /dev/null
test-lora-moe4n.full.gguf: test-moe-train.moe4-train-nonorm.gguf scripts/make-test-lora.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-lora.py test-moe-train.moe4-train-nonorm.gguf test-lora-moe4n > /dev/null

# R8.9.5: gemma3's GELU (not SiLU) gated FFN, the first refusal gemma3 and
# gemma4 hit under --train. No SWA keys are written -- that adjoint is
# already covered by the muse fixture above -- so this isolates ACT_GELU.
test-gemma3.gguf: scripts/make-test-model.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-model.py --gemma3 test-gemma3.gguf

test-lora-gemma3.full.gguf: test-gemma3.gguf scripts/make-test-lora.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-lora.py test-gemma3.gguf test-lora-gemma3

# R8.9.6: gemma-4 shapes in the backward. Dense (weightless V norm), the
# absent-V layers gemma-4's full attention ships (V is the raw K
# projection), and the E-series (shared KV, per-layer embeddings; 3 sharing
# layers, PLE width 8: wide enough to exercise every adjoint, narrow enough
# that the gate's directional derivative resolves it).
test-gemma4.gguf: scripts/make-test-model.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-model.py --arch gemma4 test-gemma4.gguf

test-lora-gemma4.full.gguf: test-gemma4.gguf scripts/make-test-lora.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-lora.py test-gemma4.gguf test-lora-gemma4

test-g4v.gguf: scripts/make-test-model.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-model.py --arch gemma4 --drop-v 1 test-g4v.gguf

test-lora-g4v.full.gguf: test-g4v.gguf scripts/make-test-lora.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-lora.py test-g4v.gguf test-lora-g4v

test-es8.gguf: scripts/make-test-model.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-model.py --eseries 3,8 --drop-kv shared test-es8.gguf

test-lora-es8.full.gguf: test-es8.gguf scripts/make-test-lora.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-lora.py test-es8.gguf test-lora-es8

test-qkw.gguf: scripts/make-test-model.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-model.py --qk-norm --wide test-qkw.gguf

test-lora-qk.full.gguf: test-qk.gguf scripts/make-test-lora.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-lora.py test-qk.gguf test-lora-qk

# fused-int8 CPU dot tolerance gate: the CPU twin of the TC gate — teacher-
# forced logits, 0/64 top-1 flips + bounded deviation, per (type, model) via
# the model argument. CPU-only, so it needs no GPU backend to be meaningful.
TEST_I8_TOL_SRC = tests/test_i8_tol.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o src/quants.c \
                  $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_I8_TOL): $(TEST_I8_TOL_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_I8_TOL_SRC) -o $@ $(LDFLAGS)

# tolerance-gated Metal decode matvec: the GPU twin of the i8 gate — teacher-
# forced logits, 0/64 top-1 flips + bounded deviation, per (type, model) via
# the model argument. Self-skips on backends with no fast matvec kernel.
TEST_MV_TOL_SRC = tests/test_mv_tol.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o $(QUANTS_OBJ) \
                  $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_MV_TOL): $(TEST_MV_TOL_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_MV_TOL_SRC) -o $@ $(LDFLAGS)

# cooperative-KV attention read: same shape as the fast-matvec gate, for the
# other reassociating Metal route. Self-skips where it never dispatches.
TEST_ATTN_TOL_SRC = tests/test_attn_tol.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o $(QUANTS_OBJ) \
                    $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_ATTN_TOL): $(TEST_ATTN_TOL_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_ATTN_TOL_SRC) -o $@ $(LDFLAGS)

# CPU/GPU byte identity at LOGIT precision. The text-comparison gates
# (test-metal-kquant and friends) are sound on real models and blind on toy
# fixtures, where greedy argmax absorbs large numeric differences -- so a
# fixture-scale backend feature could be wrong and pass everything. This
# compares the logit vectors themselves.
TEST_CUDA_TEARDOWN = $(TEST_BATCH:test-batch%=test-cuda-teardown%)
TEST_CUDA_TEARDOWN_SRC = tests/test_cuda_teardown.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o $(QUANTS_OBJ) \
                  $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_CUDA_TEARDOWN): $(TEST_CUDA_TEARDOWN_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src -I tests $(TEST_CUDA_TEARDOWN_SRC) -o $@ $(LDFLAGS)

TEST_GPU_ID_SRC = tests/test_gpu_identity.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o $(QUANTS_OBJ) \
                  $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/vramreg.o $(GPU_OBJ) \
                  $(FINITE_CHECK_OBJ)
$(TEST_GPU_ID): $(TEST_GPU_ID_SRC) $(HDR) tests/finite_check.h
	$(CC) $(CFLAGS) -I src -I tests $(TEST_GPU_ID_SRC) -o $@ $(LDFLAGS)

# GPU-matvec vs GPU-grouped-MMA on the house fidelity columns (the routing
# half of the account is scripts/moe-mm-flips.py). Same link as gpu-identity.
TEST_MOE_MM_AB_SRC = tests/test_moe_mm_ab.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o $(QUANTS_OBJ) \
                     $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_MOE_MM_AB): $(TEST_MOE_MM_AB_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_MOE_MM_AB_SRC) -o $@ $(LDFLAGS)

# Metal microbatch decode vs sequential: byte identity or bust, with an
# engagement check so a declined batch cannot pass by comparing the
# sequential path against itself.
TEST_BATCH_ID_SRC = tests/test_batch_identity.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o $(QUANTS_OBJ) \
                    $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_BATCH_ID): $(TEST_BATCH_ID_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_BATCH_ID_SRC) -o $@ $(LDFLAGS)

# Speculative verify rows vs solo decode steps: byte identity on every row of
# rounds 2..16 wide with rewinds (tests/test_spec_identity.c).
TEST_SPEC_ID_SRC = tests/test_spec_identity.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o $(QUANTS_OBJ) \
                   $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_SPEC_ID): $(TEST_SPEC_ID_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_SPEC_ID_SRC) -o $@ $(LDFLAGS)

# Split attention (R3.1.7): the (head, chunk) / (head, slice) split is
# bit-identical to the head-parallel path at 1-7 threads, forced and
# automatic, solo and batched, on every cache kind and attention variant:
# f16 and fp4 (test.gguf), q8_0 and k8v4 (head width 32), heads cut into
# two 32-channel slices (width 64, every cache kind) and three 16-channel
# ones (width 48, f16 and fp4), a width no slice divides (muse, 24), a
# sliding-window ring, tied-V (gemma4-hetero) and attention sinks (gpt-oss).
TEST_ATTN_SPLIT_SRC = tests/test_attn_split.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o \
                      $(QUANTS_OBJ) $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o \
                      $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_ATTN_SPLIT): $(TEST_ATTN_SPLIT_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_ATTN_SPLIT_SRC) -o $@ $(LDFLAGS)

# the cases go to the binary as a list file (@FILE, see check_list in the
# test): twenty file:mode arguments on one command line are what Defender's
# behaviour monitoring takes for malware on Windows
ATTN_SPLIT_CASES = test.gguf test.gguf:fp4 attn-split-hd32.gguf \
	attn-split-hd32.gguf:q8 attn-split-hd32.gguf:k8v4 test-muse.gguf \
	attn-split-hd64.gguf attn-split-hd64.gguf:q8 attn-split-hd64.gguf:fp4 \
	attn-split-hd64.gguf:k8v4 attn-split-hd48.gguf attn-split-hd48.gguf:fp4 \
	attn-split-swa.gguf:ring attn-split-g4h.gguf:tied \
	attn-split-moe.gptoss-top1.gguf
test-attn-split: $(TEST_ATTN_SPLIT) test.gguf test-muse.gguf
	@set -e; \
	$(PYTHON) scripts/make-test-model.py --head-dim 32 attn-split-hd32.gguf > /dev/null; \
	$(PYTHON) scripts/make-test-model.py --head-dim 64 attn-split-hd64.gguf > /dev/null; \
	$(PYTHON) scripts/make-test-model.py --head-dim 48 attn-split-hd48.gguf > /dev/null; \
	$(PYTHON) scripts/make-test-model.py --swa 32,2 attn-split-swa.gguf > /dev/null; \
	$(PYTHON) scripts/make-test-model.py --gemma4-hetero attn-split-g4h.gguf > /dev/null; \
	$(PYTHON) scripts/make-test-moe.py attn-split-moe > /dev/null; \
	printf '%s\n' $(ATTN_SPLIT_CASES) > attn-split.list; \
	./$(TEST_ATTN_SPLIT) @attn-split.list 2> attn-split.err || { \
		tail -40 attn-split.err; exit 1; }; \
	rm -f attn-split.list

# fused-vs-eager MoE routing tolerance: same full-engine link as tc-tol, and
# the same self-skipping shape (no GPU / not MoE / no full offload / the fused
# router never engaged all skip rather than pass)
TEST_MOE_TOL_SRC = tests/test_moe_tol.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o $(QUANTS_OBJ) \
                  $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_MOE_TOL): $(TEST_MOE_TOL_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_MOE_TOL_SRC) -o $@ $(LDFLAGS)

TEST_MOE_ROUTER_SRC = tests/test_moe_router.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o $(QUANTS_OBJ) \
                  $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_MOE_ROUTER): $(TEST_MOE_ROUTER_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_MOE_ROUTER_SRC) -o $@ $(LDFLAGS)

# residency-warning wording: needs the loader (the hot-set estimate has to
# agree with the actual tensor set), so it takes the same link as the two above
TEST_PAGING_WARN_SRC = tests/test_paging_warn.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o $(QUANTS_OBJ) \
                  $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_PAGING_WARN): $(TEST_PAGING_WARN_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_PAGING_WARN_SRC) -o $@ $(LDFLAGS)
# reservation auto-fit arithmetic: no model file, no GPU, no fixture -- the
# regime it covers is unreachable on a dev machine, so it is fed real 7B/24 GB
# numbers directly. See the header of tests/test_autofit.c.
TEST_AUTOFIT_SRC = tests/test_autofit.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o $(QUANTS_OBJ) \
                  $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_AUTOFIT): $(TEST_AUTOFIT_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_AUTOFIT_SRC) -o $@ $(LDFLAGS)
# the Responses framing state machine, driven directly over a socketpair.
# Includes completion.c (the framer is static there) and links the engine
# around it — the runner's object set minus main.c, server.c and the file the
# test itself includes.
TEST_RESP_SM_SRC = tests/test_responses_sm.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o \
                  $(QUANTS_OBJ) $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/sample.o \
                  $(OBJDIR)/jsonmode.o $(SCHEMA_OBJ) $(OBJDIR)/json.o $(OBJDIR)/engine.o \
                  $(OBJDIR)/template.o $(OBJDIR)/vramreg.o $(OBJDIR)/http.o $(OBJDIR)/envelope.o $(OBJDIR)/ed25519.o $(MLDSA_OBJ) $(OBJDIR)/ecdsa.o $(OBJDIR)/oms.o $(OBJDIR)/registry.o $(OBJDIR)/provenance.o $(OBJDIR)/respstore.o $(OBJDIR)/receipts.o $(OBJDIR)/watermark.o $(OBJDIR)/kvsnap.o $(OBJDIR)/scheduler.o $(OBJDIR)/decide.o $(OBJDIR)/clef.o $(GPU_OBJ)
$(TEST_RESP_SM): $(TEST_RESP_SM_SRC) src/completion.c $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_RESP_SM_SRC) -o $@ $(LDFLAGS)

# a client "stop" sequence under constrained output. Same link as the framing
# test above and for the same reason -- gen_ctx and stop_feed are static in
# completion.c, and the property under test is that the engine's validator and
# that sink agree on what the client received, so neither side may be stubbed.
TEST_STOP_CONSTRAINT_SRC = tests/test_stop_constraint.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o \
                  $(QUANTS_OBJ) $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/sample.o \
                  $(OBJDIR)/jsonmode.o $(SCHEMA_OBJ) $(OBJDIR)/json.o $(OBJDIR)/engine.o \
                  $(OBJDIR)/template.o $(OBJDIR)/vramreg.o $(OBJDIR)/http.o $(OBJDIR)/envelope.o $(OBJDIR)/ed25519.o $(MLDSA_OBJ) $(OBJDIR)/ecdsa.o $(OBJDIR)/oms.o $(OBJDIR)/registry.o $(OBJDIR)/provenance.o $(OBJDIR)/respstore.o $(OBJDIR)/receipts.o $(OBJDIR)/watermark.o $(OBJDIR)/kvsnap.o \
                  $(OBJDIR)/scheduler.o $(OBJDIR)/decide.o $(OBJDIR)/clef.o $(GPU_OBJ)
$(TEST_STOP_CONSTRAINT): $(TEST_STOP_CONSTRAINT_SRC) src/completion.c $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_STOP_CONSTRAINT_SRC) -o $@ $(LDFLAGS)

# the per-request prompt BUDGET, on all three tool-calling surfaces. The
# renderer truncates silently at its cap, so a buffer estimate that misses a
# term drops the tail of the prompt -- the user's own turn -- with no error.
# handle_chat is static in server.c, so that TU is #included by the test and
# left out of the link; completion.c is out too, because the test captures the
# prompt instead of generating from it.
TEST_BUDGET_SRC = tests/test_prompt_budget.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o \
                  $(QUANTS_OBJ) $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/sample.o \
                  $(OBJDIR)/jsonmode.o $(SCHEMA_OBJ) $(OBJDIR)/json.o $(OBJDIR)/engine.o \
                  $(OBJDIR)/template.o $(OBJDIR)/vramreg.o $(OBJDIR)/http.o $(OBJDIR)/envelope.o $(OBJDIR)/ed25519.o $(MLDSA_OBJ) $(OBJDIR)/ecdsa.o $(OBJDIR)/oms.o $(OBJDIR)/registry.o $(OBJDIR)/provenance.o $(OBJDIR)/respstore.o $(OBJDIR)/receipts.o $(OBJDIR)/watermark.o $(OBJDIR)/kvsnap.o \
                  $(OBJDIR)/scheduler.o $(OBJDIR)/decide.o $(OBJDIR)/clef.o $(OBJDIR)/api_responses.o $(OBJDIR)/api_anthropic.o \
                  $(OBJDIR)/session.o $(GPU_OBJ)
$(TEST_BUDGET): $(TEST_BUDGET_SRC) src/server.c $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_BUDGET_SRC) -o $@ $(LDFLAGS)

# a replayed tool result must name the function it came from, on all three
# tool-calling surfaces. Same recipe as the budget test -- handle_chat is
# static in server.c, so that TU is #included by the test and left out of the
# link -- plus a socketpair, because half of what is asserted is the 400 a
# request that cannot be attributed gets back.
TEST_ATTRIB_SRC = tests/test_tool_attribution.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o \
                  $(QUANTS_OBJ) $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/sample.o \
                  $(OBJDIR)/jsonmode.o $(SCHEMA_OBJ) $(OBJDIR)/json.o $(OBJDIR)/engine.o \
                  $(OBJDIR)/template.o $(OBJDIR)/vramreg.o $(OBJDIR)/http.o $(OBJDIR)/envelope.o $(OBJDIR)/ed25519.o $(MLDSA_OBJ) $(OBJDIR)/ecdsa.o $(OBJDIR)/oms.o $(OBJDIR)/registry.o $(OBJDIR)/provenance.o $(OBJDIR)/respstore.o $(OBJDIR)/receipts.o $(OBJDIR)/watermark.o $(OBJDIR)/kvsnap.o \
                  $(OBJDIR)/scheduler.o $(OBJDIR)/decide.o $(OBJDIR)/clef.o $(OBJDIR)/api_responses.o $(OBJDIR)/api_anthropic.o \
                  $(OBJDIR)/session.o $(GPU_OBJ)
$(TEST_ATTRIB): $(TEST_ATTRIB_SRC) src/server.c $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_ATTRIB_SRC) -o $@ $(LDFLAGS)

# The same inbound translations under injected allocation failure. Same source
# set minus the three files that DO the translating -- server.c, api_responses.c
# and api_anthropic.c are compiled INTO the test with substituted allocators.
TEST_MSG_OOM = $(TEST_BATCH:test-batch%=test-prompt-oom%)
TEST_MSG_OOM_SRC = tests/test_prompt_oom.c \
                   $(filter-out $(OBJDIR)/api_anthropic.o $(OBJDIR)/api_responses.o \
                                $(OBJDIR)/json.o,\
                     $(TEST_ATTRIB_SRC:tests/test_tool_attribution.c=))
$(TEST_MSG_OOM): $(TEST_MSG_OOM_SRC) src/server.c src/api_anthropic.c \
                 src/api_responses.c src/json.c $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_MSG_OOM_SRC) -o $@ $(LDFLAGS)

# The runner side of the template-conformance gate. Same recipe as the two
# tests above, and for the same reason: the flattening from an OpenAI request
# to the renderer's flat turns is handle_chat's, it is static in server.c, and
# a harness that RE-IMPLEMENTED it would be comparing jinja against a copy of
# runner rather than against runner. So server.c is #included and the
# generation path is stubbed out of the link -- the prompt is the artifact,
# nothing is generated from it.
#
# api_responses.c / api_anthropic.c are additionally left out: only server.c's
# route table refers to them, the gate never reaches it, and dropping them
# keeps the add_generation_prompt interception (see the driver's header
# comment) confined to one translation unit.
TMPL_CONF_RENDER_SRC = scripts/template-conformance-render.c $(OBJDIR)/gguf.o \
                  $(OBJDIR)/compat.o $(QUANTS_OBJ) $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o \
                  $(OBJDIR)/sample.o $(OBJDIR)/jsonmode.o $(SCHEMA_OBJ) $(OBJDIR)/json.o \
                  $(OBJDIR)/engine.o $(OBJDIR)/template.o $(OBJDIR)/vramreg.o $(OBJDIR)/http.o \
                  $(OBJDIR)/envelope.o $(OBJDIR)/ed25519.o $(MLDSA_OBJ) $(OBJDIR)/ecdsa.o $(OBJDIR)/oms.o $(OBJDIR)/registry.o $(OBJDIR)/provenance.o $(OBJDIR)/respstore.o $(OBJDIR)/receipts.o $(OBJDIR)/watermark.o $(OBJDIR)/kvsnap.o $(OBJDIR)/scheduler.o $(OBJDIR)/decide.o $(OBJDIR)/clef.o $(OBJDIR)/session.o $(GPU_OBJ)
$(TMPL_CONF_RENDER): $(TMPL_CONF_RENDER_SRC) src/server.c $(HDR)
	$(CC) $(CFLAGS) -I src $(TMPL_CONF_RENDER_SRC) -o $@ $(LDFLAGS)

# server_run twice in one process: the property a once-per-process global can
# hide forever, because nothing ever asks the state to come back.
TEST_RESTART = $(TEST_BATCH:test-batch%=test-server-restart%)
TEST_RESTART_SRC = tests/test_server_restart.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o \
                   $(QUANTS_OBJ) $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/sample.o \
                   $(OBJDIR)/jsonmode.o $(SCHEMA_OBJ) $(OBJDIR)/json.o $(OBJDIR)/engine.o \
                   $(OBJDIR)/template.o $(OBJDIR)/vramreg.o $(OBJDIR)/http.o $(OBJDIR)/envelope.o $(OBJDIR)/ed25519.o $(MLDSA_OBJ) $(OBJDIR)/ecdsa.o $(OBJDIR)/oms.o $(OBJDIR)/registry.o $(OBJDIR)/provenance.o $(OBJDIR)/respstore.o $(OBJDIR)/receipts.o $(OBJDIR)/watermark.o $(OBJDIR)/kvsnap.o \
                   $(OBJDIR)/scheduler.o $(OBJDIR)/completion.o $(OBJDIR)/decide.o $(OBJDIR)/clef.o $(OBJDIR)/api_responses.o \
                   $(OBJDIR)/api_anthropic.o $(OBJDIR)/session.o $(OBJDIR)/server.o $(GPU_OBJ)
$(TEST_RESTART): $(TEST_RESTART_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_RESTART_SRC) -o $@ $(LDFLAGS)

# /v1/capabilities is answered on the accept thread and dereferences the
# resident model; a slot thread frees that model on any swap. Only ASan makes
# the read of freed bytes loud, so this gate is BUILT sanitized rather than
# added to `make test` -- same shape as test-shared-asan above. Kept out of the
# default run because it starts a real server and drives it for a few seconds.
TEST_SWAP_RACE_SRC = tests/test_swap_race.c src/gguf.c src/compat.c \
                     src/quants.c src/tokenizer.c src/model.c src/sample.c \
                     src/jsonmode.c $(SCHEMA_SRC) src/json.c src/engine.c \
                     src/template.c src/vramreg.c src/http.c src/envelope.c src/ed25519.c $(MLDSA_SRC) src/ecdsa.c src/oms.c src/registry.c src/provenance.c src/lineage.c src/respstore.c src/receipts.c src/watermark.c src/kvsnap.c \
                     src/scheduler.c src/completion.c src/api_responses.c \
                     src/api_anthropic.c src/session.c src/server.c src/decide.c src/clef.c $(GPU_SRC)
# Parallel slots share one tokenizer and encode on their own threads; under
# ThreadSanitizer any per-call state left in the shared struct is a report,
# and TSan exits non-zero on one. Not in `make test` (MinGW has no TSan); CI
# runs it beside test-swap-race.
TEST_TOKENIZER_RACE_SRC = tests/test_tokenizer_race.c src/tokenizer.c src/gguf.c \
                          src/compat.c src/quants.c
test-tokenizer-race: $(TEST_TOKENIZER_RACE_SRC) $(HDR) test.gguf
	$(CC) -O1 -g -fsanitize=thread -fno-omit-frame-pointer -fno-fast-math \
	    -std=gnu11 -Wall -Wno-unused-const-variable $(GPU_BACKEND_DEF) -I src \
	    $(TEST_TOKENIZER_RACE_SRC) -o test-tokenizer-race-bin -lm -lpthread
	TSAN_OPTIONS=halt_on_error=1 ./test-tokenizer-race-bin test.gguf

test-swap-race: $(TEST_SWAP_RACE_SRC) $(HDR) test.gguf
	$(CC) -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer \
	    -fno-fast-math -std=gnu11 -Wall -I src $(TEST_SWAP_RACE_SRC) \
	    -o test-swap-race-bin $(LDFLAGS)
	ASAN_OPTIONS=detect_leaks=0 ./test-swap-race-bin test.gguf

# the weight-residency platform layer: mlock, mincore, major faults, available
# RAM. compat.c only -- these are platform shims, not engine code.
TEST_RESIDENCY = $(TEST_BATCH:test-batch%=test-residency%)
$(TEST_RESIDENCY): tests/test_residency.c $(OBJDIR)/compat.o $(HDR)
	$(CC) $(CFLAGS) -I src tests/test_residency.c $(OBJDIR)/compat.o -o $@ $(LDFLAGS)

# what this process may use: the host's CPUs and RAM narrowed by the cgroup
# limits (cpu.max, memory.max), read from fixture trees the test writes
TEST_CAPACITY = $(TEST_BATCH:test-batch%=test-capacity%)
$(TEST_CAPACITY): tests/test_capacity.c $(OBJDIR)/compat.o $(HDR)
	$(CC) $(CFLAGS) -I src tests/test_capacity.c $(OBJDIR)/compat.o -o $@ $(LDFLAGS)

# the device turn is FIFO, not just exclusive. scheduler.c is #included by the
# test (the turnstile is static) so it is NOT linked here.
TEST_SCHED_TURN = $(TEST_BATCH:test-batch%=test-sched-turn%)
TEST_SCHED_TURN_SRC = tests/test_sched_turn.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o \
                      $(QUANTS_OBJ) $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/sample.o \
                      $(OBJDIR)/jsonmode.o $(SCHEMA_OBJ) $(OBJDIR)/json.o $(OBJDIR)/engine.o \
                      $(OBJDIR)/template.o $(OBJDIR)/vramreg.o $(OBJDIR)/http.o $(OBJDIR)/envelope.o $(OBJDIR)/ed25519.o $(MLDSA_OBJ) $(OBJDIR)/ecdsa.o $(OBJDIR)/oms.o $(OBJDIR)/registry.o $(OBJDIR)/provenance.o $(OBJDIR)/respstore.o $(OBJDIR)/receipts.o $(OBJDIR)/watermark.o $(OBJDIR)/kvsnap.o $(GPU_OBJ)
$(TEST_SCHED_TURN): $(TEST_SCHED_TURN_SRC) src/scheduler.c $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_SCHED_TURN_SRC) -o $@ $(LDFLAGS)

# snapshot persistence: the round trip, and the refusals that matter more
TEST_PFX_PERSIST = $(TEST_BATCH:test-batch%=test-prefix-persist%)
TEST_PFX_PERSIST_SRC = tests/test_prefix_persist.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o \
                       $(QUANTS_OBJ) $(OBJDIR)/tokenizer.o $(OBJDIR)/model.o $(OBJDIR)/sample.o \
                       $(OBJDIR)/jsonmode.o $(SCHEMA_OBJ) $(OBJDIR)/json.o $(OBJDIR)/engine.o \
                       $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_PFX_PERSIST): $(TEST_PFX_PERSIST_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_PFX_PERSIST_SRC) -o $@ $(LDFLAGS)

TEST_QUANTIZE_SRC = tests/test_quantize.c $(QUANTIZE_OBJ) $(OBJDIR)/gguf.o \
                    $(OBJDIR)/compat.o $(QUANTS_OBJ) $(OBJDIR)/json.o
$(TEST_QUANTIZE): $(TEST_QUANTIZE_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_QUANTIZE_SRC) -o $@ $(LDFLAGS)

# Every allocation of a file transformation failed in turn, and the refusals
# between its stages (tests/test_quantize_oom.c): quantize.c is compiled into
# the test with its allocators substituted, under QUANTS_CFLAGS as quantize.o
# is, so it links what quantize.o would and not quantize.o itself.
TEST_QUANTIZE_OOM_SRC = tests/test_quantize_oom.c $(OBJDIR)/gguf.o \
                        $(OBJDIR)/compat.o $(QUANTS_OBJ) $(OBJDIR)/json.o
$(TEST_QUANTIZE_OOM): $(TEST_QUANTIZE_OOM_SRC) src/quantize.c $(HDR)
	$(CC) $(QUANTS_CFLAGS) -I src $(TEST_QUANTIZE_OOM_SRC) -o $@ $(LDFLAGS)

# vramreg.c is #included (calloc-hooked) by the test, so it is not linked here
$(TEST_VRAM_ROLLBACK): tests/test_vram_rollback.c $(OBJDIR)/compat.o $(HDR)
	$(CC) $(CFLAGS) -I src tests/test_vram_rollback.c $(OBJDIR)/compat.o -o $@ $(LDFLAGS)

$(TEST_GGUF_GETTERS): tests/test_gguf_getters.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o $(QUANTS_OBJ) $(HDR)
	$(CC) $(CFLAGS) -I src tests/test_gguf_getters.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o $(QUANTS_OBJ) -o $@ $(LDFLAGS)

$(TEST_PARSE): tests/test_parse.c $(OBJDIR)/compat.o src/compat.h
	$(CC) $(CFLAGS) -I src tests/test_parse.c $(OBJDIR)/compat.o -o $@ $(LDFLAGS)

$(TEST_HFHUB): tests/test_hfhub.c $(OBJDIR)/hfhub.o $(OBJDIR)/envelope.o $(OBJDIR)/ed25519.o $(MLDSA_OBJ) $(OBJDIR)/json.o $(OBJDIR)/compat.o src/hfhub.h
	$(CC) $(CFLAGS) -I src tests/test_hfhub.c $(OBJDIR)/hfhub.o $(OBJDIR)/envelope.o $(OBJDIR)/ed25519.o $(MLDSA_OBJ) $(OBJDIR)/json.o $(OBJDIR)/compat.o -o $@ $(LDFLAGS)

$(TEST_ENVELOPE): tests/test_envelope.c $(OBJDIR)/envelope.o $(OBJDIR)/ed25519.o $(MLDSA_OBJ) $(OBJDIR)/json.o $(OBJDIR)/compat.o src/envelope.h src/json.h src/runner.h
	$(CC) $(CFLAGS) -I src tests/test_envelope.c $(OBJDIR)/envelope.o $(OBJDIR)/ed25519.o $(MLDSA_OBJ) $(OBJDIR)/json.o $(OBJDIR)/compat.o -o $@ $(LDFLAGS)

# SHA-256 on every backend the host has (the CPU's instructions and the
# portable code) against reference digests (tests/test_sha256.c, envelope.c
# compiled in to reach the backend switch).
$(TEST_SHA256): tests/test_sha256.c src/envelope.c $(OBJDIR)/ed25519.o $(MLDSA_OBJ) $(OBJDIR)/json.o $(OBJDIR)/compat.o src/envelope.h src/json.h src/runner.h
	$(CC) $(CFLAGS) -I src tests/test_sha256.c $(OBJDIR)/ed25519.o $(MLDSA_OBJ) $(OBJDIR)/json.o $(OBJDIR)/compat.o -o $@ $(LDFLAGS)

# receipt and model-signature primitives: RFC 8032 / RFC 6979 known answers
$(TEST_ED25519): tests/test_ed25519.c $(OBJDIR)/ed25519.o src/ed25519.h
	$(CC) $(CFLAGS) -I src tests/test_ed25519.c $(OBJDIR)/ed25519.o -o $@ $(LDFLAGS)

# portable math (T3) against libm: accuracy bounds and special values; built
# without RUNNER_PORTABLE_MATH so both implementations are visible
# fpclass.h under the engine's own CFLAGS (-ffast-math): the class tests must
# hold on whatever compiler builds the runner (see the header)
$(TEST_FPCLASS): tests/test_fpclass.c src/fpclass.h
	$(CC) $(CFLAGS) -I src tests/test_fpclass.c -o $@ $(LDFLAGS)

# act.h's sigmoid and SiLU under the engine's own CFLAGS (-ffast-math): the
# plain forms return NaN below about -88 on x86-64 (see the header)
$(TEST_ACT): tests/test_act.c src/act.h src/fpclass.h
	$(CC) $(CFLAGS) -I src tests/test_act.c -o $@ $(LDFLAGS)

$(TEST_PMATH): tests/test_pmath.c src/pmath.h
	$(CC) -O2 -fno-fast-math -ffp-contract=off -std=gnu11 -Wall -I src tests/test_pmath.c -o $@ -lm

# ML-DSA-44: NIST ACVP FIPS 204 known answers (key generation from a seed,
# deterministic signature), then sign/verify/tamper on the wrapper
$(TEST_MLDSA): tests/test_mldsa.c tests/mldsa_kat.h $(MLDSA_OBJ) src/mldsa.h
	$(CC) $(CFLAGS) -I src -I tests tests/test_mldsa.c $(MLDSA_OBJ) -o $@ $(LDFLAGS)

$(TEST_WATERMARK): tests/test_watermark.c $(OBJDIR)/watermark.o $(OBJDIR)/envelope.o $(OBJDIR)/ed25519.o $(MLDSA_OBJ) $(OBJDIR)/json.o $(OBJDIR)/compat.o src/watermark.h
	$(CC) $(CFLAGS) -I src tests/test_watermark.c $(OBJDIR)/watermark.o $(OBJDIR)/envelope.o $(OBJDIR)/ed25519.o $(MLDSA_OBJ) $(OBJDIR)/json.o $(OBJDIR)/compat.o -o $@ $(LDFLAGS)

$(TEST_ECDSA): tests/test_ecdsa.c $(OBJDIR)/ecdsa.o $(OBJDIR)/envelope.o $(OBJDIR)/ed25519.o $(MLDSA_OBJ) $(OBJDIR)/json.o $(OBJDIR)/compat.o src/ecdsa.h
	$(CC) $(CFLAGS) -I src tests/test_ecdsa.c $(OBJDIR)/ecdsa.o $(OBJDIR)/envelope.o $(OBJDIR)/ed25519.o $(MLDSA_OBJ) $(OBJDIR)/json.o $(OBJDIR)/compat.o -o $@ $(LDFLAGS)

# quants.c joins for tpool_create/tpool_destroy: the test now also pins that an
# over-large -t is clamped to TP_MAX rather than silently discarded.
$(TEST_THREAD_DEFAULT): tests/test_thread_default.c $(OBJDIR)/compat.o src/compat.h $(QUANTS_OBJ)
	$(CC) $(CFLAGS) -I src tests/test_thread_default.c $(OBJDIR)/compat.o $(QUANTS_OBJ) -o $@ $(LDFLAGS)

TEST_MODEL_LOAD_FAILURE_SRC = tests/test_model_load_failure.c $(OBJDIR)/gguf.o \
                              $(OBJDIR)/compat.o $(QUANTS_OBJ) $(OBJDIR)/model.o \
                              $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_MODEL_LOAD_FAILURE): $(TEST_MODEL_LOAD_FAILURE_SRC) $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_MODEL_LOAD_FAILURE_SRC) -o $@ $(LDFLAGS)

# Every allocation of a model load failed in turn (tests/test_model_load_oom.c):
# model.c is compiled into the test with its allocators substituted, so it
# links every other engine object model.o would, and not model.o itself.
TEST_MODEL_LOAD_OOM_SRC = tests/test_model_load_oom.c $(OBJDIR)/gguf.o \
                          $(OBJDIR)/compat.o $(QUANTS_OBJ) \
                          $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_MODEL_LOAD_OOM): $(TEST_MODEL_LOAD_OOM_SRC) src/model.c $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_MODEL_LOAD_OOM_SRC) -o $@ $(LDFLAGS)

# The host MoE bridge (--cpu-moe on CUDA) against the CPU forward's FFN
# (tests/test_moe_host_bridge.c), built the same way: model.c compiled in.
TEST_MOE_HOST_BRIDGE_SRC = tests/test_moe_host_bridge.c $(OBJDIR)/gguf.o \
                           $(OBJDIR)/compat.o $(QUANTS_OBJ) \
                           $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_MOE_HOST_BRIDGE): $(TEST_MOE_HOST_BRIDGE_SRC) src/model.c $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_MOE_HOST_BRIDGE_SRC) -o $@ $(LDFLAGS)

# The adapter delta (lora_apply's blocked chains) against the serial loop it
# replaced (tests/test_lora_chains.c), model.c compiled in the same way, with
# -fno-associative-math: the test checks the source's fmaf chains, and gcc's
# -ffast-math may reassociate an fmaf chain into a vector reduction (it did
# at -march=native on a Zen 5, each loop its own way; the release ISAs and
# clang keep the chains). See the test's header.
TEST_LORA_CHAINS_SRC = tests/test_lora_chains.c $(OBJDIR)/gguf.o \
                       $(OBJDIR)/compat.o $(QUANTS_OBJ) \
                       $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_LORA_CHAINS): $(TEST_LORA_CHAINS_SRC) src/model.c $(HDR)
	$(CC) $(CFLAGS) -fno-associative-math -I src $(TEST_LORA_CHAINS_SRC) -o $@ $(LDFLAGS)

# The trainer's transposed product (matvec_t's tiles) against its per-element
# contract on every weight type (tests/test_mvt_chains.c), model.c compiled in.
TEST_MVT_CHAINS_SRC = tests/test_mvt_chains.c $(OBJDIR)/gguf.o \
                      $(OBJDIR)/compat.o $(QUANTS_OBJ) \
                      $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_MVT_CHAINS): $(TEST_MVT_CHAINS_SRC) src/model.c $(HDR)
	$(CC) $(CFLAGS) -I src $(TEST_MVT_CHAINS_SRC) -o $@ $(LDFLAGS)

# The CPU trainer's Gated DeltaNet recurrence (the recompute and the sweep
# through time) against a double-precision reference (tests/dn_ref.h, the
# device's reference too), model.c compiled in (tests/test_dn_recurrence.c).
TEST_DN_RECURRENCE_SRC = tests/test_dn_recurrence.c $(OBJDIR)/gguf.o \
                         $(OBJDIR)/compat.o $(QUANTS_OBJ) \
                         $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_DN_RECURRENCE): $(TEST_DN_RECURRENCE_SRC) src/model.c $(HDR) tests/dn_ref.h
	$(CC) $(CFLAGS) -I src $(TEST_DN_RECURRENCE_SRC) -o $@ $(LDFLAGS)

# The CPU trainer's attention backward against a double-precision reference
# (tests/attn_ref.h, the device's reference too), model.c compiled in.
TEST_ATTN_BW_SRC = tests/test_attn_bw.c $(OBJDIR)/gguf.o $(OBJDIR)/compat.o \
                   $(QUANTS_OBJ) $(OBJDIR)/vramreg.o $(GPU_OBJ)
$(TEST_ATTN_BW): $(TEST_ATTN_BW_SRC) src/model.c $(HDR) tests/attn_ref.h
	$(CC) $(CFLAGS) -I src $(TEST_ATTN_BW_SRC) -o $@ $(LDFLAGS)

# The CPU-only backend stub must be able to STAND IN for a real backend --
# that is the entire claim in its own header comment ("CUDA (cuda.c) and Metal
# (metal.m) implement this same interface"). It could not: gpu_max_working_set,
# gpu_tc_force and gpu_moe_eager_force are declared in gpu.h and were missing
# from it, so a link with this file in place of $(GPU_SRC) failed outright.
#
# Nothing in the Makefile builds src/gpu_none.c -- every platform branch picks
# cuda.c or metal.m -- which is exactly how it rotted, so the gate has to be
# the link itself. -O0 because what is under test is the interface, not the
# code generation, and this is the one build in the tree nobody else pays for.
CPU_STUB_SRC = $(filter-out $(GPU_SRC),$(SRC)) src/gpu_none.c
test-gpu-stub: $(CPU_STUB_SRC) $(HDR)
	$(CC) -O0 -std=gnu11 -Wall -Wextra -Wno-unused-parameter -I src \
	    $(CPU_STUB_SRC) -o runner-gpu-stub $(LDFLAGS)
	@./runner-gpu-stub --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); assert d.get('gpu') is None, d.get('gpu'); print('cpu-only backend stub ok (links a runner, and --caps reports no GPU)')"

test.gguf: scripts/make-test-model.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-model.py test.gguf

# tiny qwen35 recurrent (SSM) fixture — 3 recurrent + 1 attention layer — for
# the recurrent-state cache-seam gate (make-test-model.py cannot emit an SSM arch)
test-ornith.gguf: scripts/make-test-ornith.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-ornith.py test-ornith.gguf

# tiny lfm2 short-convolution hybrids, dense and MoE, for the same
# recurrent-state gates (the convolution ring is lfm2's whole state)
test-lfm2.gguf: scripts/make-test-lfm2.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-lfm2.py test-lfm2.gguf
test-lfm2-moe.gguf: scripts/make-test-lfm2.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-lfm2.py test-lfm2-moe.gguf --moe

# R8.16: full-coverage adapters on the qwen35 and clef fixtures (every
# DeltaNet projection, the fused Q/gate and K/V/output of the attention
# layer, every FFN) for the backward through the recurrence
test-lora-ornith.full.gguf: test-ornith.gguf scripts/make-test-lora.py $(GGUF_FIXTURE_LIB)
	LORA_TARGETS=qwen35 $(PYTHON) scripts/make-test-lora.py test-ornith.gguf test-lora-ornith

test-clef.gguf: scripts/make-test-clef.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-clef.py test-clef.gguf

test-lora-clef.full.gguf: test-clef.gguf scripts/make-test-lora.py $(GGUF_FIXTURE_LIB)
	LORA_TARGETS=qwen35 $(PYTHON) scripts/make-test-lora.py test-clef.gguf test-lora-clef

# same geometry/vocab, different weights: a draft model that genuinely diverges
# from the target, so the speculative fold-rollback gate exercises rejection
test-ornith-draft.gguf: scripts/make-test-ornith.py $(GGUF_FIXTURE_LIB)
	ORNITH_TEST_SEED=0x2b992ddf $(PYTHON) scripts/make-test-ornith.py test-ornith-draft.gguf

# The same fixture with its matmul weights stored Q8_0. test.gguf is F32, and
# F32 is the ONE case where a backend's batched matvec is its own batch-1
# kernel's twin by construction — so a batch gate run only on test.gguf proves
# nothing about any model anyone ships. That is not hypothetical: the CUDA
# decode microbatch stopped being bit-identical on every quantized model on
# 2026-07-28 and `make test` stayed green for three weeks
# (docs/cuda-microbatch-identity-2026-08-18.md). This fixture is what makes the
# gate mean something.
test-q8.gguf: scripts/make-test-model.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-model.py --quant q8_0 test-q8.gguf

# Q4_K reaches Metal's default fast decode matvec (k_mvf_q4_K, the owner's
# 2026-10-23 promotion), whose microbatch columns must be the solo step's
# bits: the batch and spec identity gates run on it. 256-wide rows, so the
# K-quant superblocks tile; written by the runner's own quantizer.
test-q4k.gguf: scripts/make-test-model.py $(GGUF_FIXTURE_LIB) runner
	$(PYTHON) scripts/make-test-model.py --wide test-q4k-src.gguf > /dev/null
	./$(RUNNER_EXE) -m test-q4k-src.gguf --quantize test-q4k.gguf --quant q4_k > /dev/null
	rm -f test-q4k-src.gguf

# BF16 reaches its own CUDA decode-microbatch twins. The F32 and Q8_0 runs do
# not execute k_gemvb_bf16_x4/_x8, so neither can hold their identity contract.
test-bf16.gguf: scripts/make-test-model.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-model.py --quant bf16 test-bf16.gguf

# Ornith/Qwen3.5 CPU tracer: a committed generator builds a tiny hybrid model
# with three recurrent DeltaNet blocks and one full-attention block.
test-ornith-cpu: runner
	$(PYTHON) -m pytest -q tests/test_ornith_cpu.py

test-apertus: runner
	$(PYTHON) -m pytest -q tests/test_apertus.py

test-moe: runner
	$(PYTHON) -m pytest -q tests/test_moe.py

# --prune-experts: stacked-layout MoE expert pruning in the quantize path.
test-prune-experts: runner
	$(PYTHON) -m pytest -q tests/test_prune_experts.py

test-weight-io-bench:
	$(PYTHON) -m pytest -q tests/test_weight_io_bench.py

$(TEST_METAL_SHADERS): tests/test_metal_shaders.m src/kernels_metal.h
	$(CC) -std=gnu11 -Wall -Wextra -Wno-unused-parameter -I src \
	    tests/test_metal_shaders.m -o $@ -framework Metal -framework Foundation

$(TEST_METAL_KVFP4): tests/test_metal_kvfp4.m src/kernels_metal.h $(QUANTS_OBJ) $(HDR)
	$(CC) -std=gnu11 -Wall -Wextra -Wno-unused-parameter -I src \
	    tests/test_metal_kvfp4.m $(QUANTS_OBJ) -o $@ -lm -lpthread \
	    -framework Metal -framework Foundation

$(TEST_METAL_KQUANTS): tests/test_metal_kquants.m src/kernels_metal.h $(QUANTS_OBJ) $(HDR)
	$(CC) -std=gnu11 -Wall -Wextra -Wno-unused-parameter -I src \
	    tests/test_metal_kquants.m $(QUANTS_OBJ) -o $@ -lm -lpthread \
	    -framework Metal -framework Foundation

$(TEST_METAL_TENSOR): tests/test_metal_tensor.m src/kernels_tensor_metal.h
	$(CC) -std=gnu11 -Wall -Wextra tests/test_metal_tensor.m -o $@ \
	    -framework Metal -framework Foundation

# compat.c joins the link because the partial-offload residency guard in
# metal.m calls plat_ram_available_bytes(): deciding whether a split pays
# needs to know how much RAM the CPU tail would have to stream through.
$(TEST_METAL_OWNERSHIP): tests/test_metal_ownership.m src/metal.m src/compat.c $(HDR)
	$(CC) -std=gnu11 -Wall -Wextra -Wno-unused-parameter -I src \
	    tests/test_metal_ownership.m src/compat.c -o $@ $(LDFLAGS)

# Runs inside `make test` on macOS: a shader that does not compile makes every
# run fall back to the CPU silently, which no correctness gate can see.
test-metal-shader-gate:
ifeq ($(BUILD_OS),Darwin)
	@$(MAKE) --no-print-directory $(TEST_METAL_SHADERS) $(TEST_METAL_KQUANTS) $(TEST_METAL_KVFP4) $(TEST_METAL_TENSOR) >/dev/null
	@./$(TEST_METAL_SHADERS)
	@./$(TEST_METAL_KQUANTS)
	@./$(TEST_METAL_KVFP4)
	@./$(TEST_METAL_TENSOR)
else
	@echo "metal shader gate skipped: macOS-only backend"
endif

test-metal-fallback: runner test.gguf
ifeq ($(BUILD_OS),Darwin)
	$(MAKE) --no-print-directory $(TEST_METAL_OWNERSHIP)
	./$(TEST_METAL_OWNERSHIP)
	@set -e; \
	if ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
		./$(RUNNER_EXE) -m test.gguf -p "hello" -n 8 --temp 0 --gpu off > metal-cpu.out 2>/dev/null; \
		env RUNNER_METAL_INJECT_FAILURE=once MallocScribble=1 MallocGuardEdges=1 \
		    env RUNNER_METAL_MM=0 RUNNER_METAL_ATTN_COOP=0 RUNNER_METAL_MV=0 RUNNER_METAL_MOE_MM=0 ./$(RUNNER_EXE) -m test.gguf -p "hello" -n 8 --temp 0 --gpu auto > metal-fallback.out 2> metal-fallback.err; \
		cmp -s metal-cpu.out metal-fallback.out; \
		grep -q "falling back to CPU" metal-fallback.err; \
		env RUNNER_METAL_INIT_INJECT_FAILURE=after-kv MallocScribble=1 MallocGuardEdges=1 \
		    env RUNNER_METAL_MM=0 RUNNER_METAL_ATTN_COOP=0 RUNNER_METAL_MV=0 RUNNER_METAL_MOE_MM=0 ./$(RUNNER_EXE) -m test.gguf -p "hello" -n 8 --temp 0 --gpu auto > metal-init-fallback.out 2> metal-init-fallback.err; \
		cmp -s metal-cpu.out metal-init-fallback.out; \
		grep -q "Metal initialization failed" metal-init-fallback.err; \
		env RUNNER_METAL_INIT_INJECT_FAILURE=device ./$(RUNNER_EXE) -m test.gguf -p "hello" -n 8 --temp 0 --gpu auto > metal-init-fallback.out 2> metal-init-fallback.err; \
		cmp -s metal-cpu.out metal-init-fallback.out; \
		grep -q "no Metal device is available — using CPU" metal-init-fallback.err; \
		echo "metal fallback ownership ok"; \
	else \
		echo "metal fallback runtime smoke skipped: no Metal device reported by --caps"; \
	fi
else
	@echo "metal fallback tests skipped: macOS-only backend"
endif

# Byte-identity for the Metal paths that only just got new kernels or widening:
# q2_K/q3_K/iq4_xs tiled+matvec coverage, and f16/bf16 matvec widening.
#
# They are tested TOGETHER because a checkpoint exercising only one does not
# exist in the wild: every real "Q2_K" GGUF is a mix, and llama.cpp's mix pairs
# q2_K with q3_K (measured: tinyllama-1.1b Q2_K is 45 q2_K + 110 q3_K tensors).
# Shipping q2_K alone would have run exactly nothing. f16 and bf16 are included
# here because their widened decode kernels carry the broadest blast radius.
#
# Short AND long prompts, because they take different code paths: a batch of
# more than one token uses the tiled GEMM (k_mm_*) and decode uses the matvec
# (k_mv_*), so a short-prompt-only check leaves half of each format's kernels
# ungated. A long prompt also spans several K-tiles and column-tiles rather
# than one.
#
# The checkpoint is not in the repo, so this skips loudly rather than passing
# vacuously -- and if the model IS present but falls back to CPU, that is a
# FAILURE, not a skip: a parity check with both sides on the CPU compares
# nothing at all, which is the exact defect class the 2026-08-09 gate audit
# found three times over.
# Never glob every downloaded model into `make test`. A split part makes the
# Metal arm fall back to CPU, a routed MoE may legitimately answer to a
# sensitivity floor rather than byte identity, and a 235B checkpoint turns a
# focused kernel smoke into an unbounded multi-hour gate. The small pinned
# filename is the automatic fixture; operators can still opt a specific
# standalone dense artifact in with KQUANT_MODELS=/path/model.gguf.
KQUANT_MODELS ?= $(wildcard models/tinyllama-q2k.gguf \
    models/SmolLM2-360M-Instruct-IQ3_S.gguf models/SmolLM2-360M-Instruct-IQ1_M.gguf)
test-metal-kquant: runner
ifeq ($(BUILD_OS),Darwin)
	@set -e; \
	if [ -z "$(KQUANT_MODELS)" ]; then \
	  echo "metal quant parity: SKIP (no q2_K/q3_K/iq4_xs/f16/bf16 checkpoint in models/)"; \
	  exit 0; fi; \
	if ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
	  long=$$($(PYTHON) -c "print(' '.join(['the quick brown fox jumps over the lazy dog']*40))"); \
	  for m in $(KQUANT_MODELS); do \
	    for which in short long; do \
	      if [ $$which = short ]; then prompt="The capital of France is"; \
	      else prompt="$$long"; fi; \
	      ./$(RUNNER_EXE) -m $$m -p "$$prompt" -n 12 --temp 0 --gpu off \
	        > metal-kquant-cpu.out 2>/dev/null; \
	      env RUNNER_METAL_ATTN_COOP=0 RUNNER_METAL_MV=0 ./$(RUNNER_EXE) -m $$m -p "$$prompt" -n 12 --temp 0 --gpu auto \
	        > metal-kquant-gpu.out 2> metal-kquant-gpu.err; \
	      if grep -q "without a Metal kernel" metal-kquant-gpu.err; then \
	        echo "FAIL: $$m fell back to CPU — this parity check would compare"; \
	        echo "  the CPU against itself and pass for the wrong reason"; \
	        exit 1; fi; \
	      grep -q "Metal backend" metal-kquant-gpu.err; \
	      cmp -s metal-kquant-cpu.out metal-kquant-gpu.out || { \
	        echo "FAIL: $$m Metal output differs from CPU ($$which prompt)"; \
	        exit 1; }; \
	    done; \
	    echo "  metal quant parity ok ($$m, byte-identical, short+long)"; \
	  done; \
	else \
	  echo "metal quant parity: SKIP (no Metal device reported by --caps)"; \
	fi
else
	@echo "metal quant parity: SKIP (macOS-only backend)"
endif

test-metal-prefill: runner test.gguf
ifeq ($(BUILD_OS),Darwin)
	@set -e; \
	if ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
		prompt="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"; \
		./$(RUNNER_EXE) -m test.gguf -p "$$prompt" -n 8 -b 8 --temp 0 --gpu off > metal-prefill-cpu.out 2>/dev/null; \
		env RUNNER_METAL_STATS=1 RUNNER_METAL_MM=0 ./$(RUNNER_EXE) -m test.gguf -p "$$prompt" -n 8 -b 8 --temp 0 --gpu auto > metal-prefill-native.out 2> metal-prefill-native.err; \
		cmp -s metal-prefill-cpu.out metal-prefill-native.out; \
		grep -q "metal: native prompt batch" metal-prefill-native.err; \
		grep -q "Metal backend" metal-prefill-native.err; \
		echo "metal prompt batch ok"; \
	else \
		echo "metal prompt batch smoke skipped: no Metal device reported by --caps"; \
	fi
else
	@echo "metal prompt batch smoke skipped: macOS-only backend"
endif

# A session in which EVERY forward is a single token. metal_ensure_batch()
# short-circuits at n <= batch_cap and gpu_init leaves batch_cap at 1, so such
# a session never enters it -- and any device buffer allocated only there stays
# nil for the whole run while the kernels that read it still dispatch. Every
# other Metal gate here prefills a multi-token batch and therefore cannot see
# that state. -b 1 is the smallest way to hold a real run in it; a single-token
# prompt or a prefix-cache hit that leaves one new token reaches it too.
#
# The generation is long enough to cross METAL_ATTN_MIN_CHUNK so the chunked
# decode attention -- the one path whose scratch lives only in ensure_batch --
# actually engages: test.gguf is ctx 256 with 4 heads, so the split starts at
# position 128 and the census must report attn_chunk > 0 or this gate is
# passing on a path it never took.
test-metal-decode-only: runner test.gguf
ifeq ($(BUILD_OS),Darwin)
	@set -e; \
	if ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
		./$(RUNNER_EXE) -m test.gguf -p "hello" -n 200 -b 1 --temp 0 --gpu off > metal-decode1-cpu.out 2>/dev/null; \
		env RUNNER_METAL_STATS=1 ./$(RUNNER_EXE) -m test.gguf -p "hello" -n 200 -b 1 --temp 0 --gpu auto > metal-decode1-gpu.out 2> metal-decode1-gpu.err; \
		grep -q "Metal backend" metal-decode1-gpu.err || { \
			echo "FAIL: Metal never engaged — this would compare the CPU with itself"; exit 1; }; \
		nch=$$(grep -oE "attn_chunk=[0-9]+" metal-decode1-gpu.err | grep -oE "[0-9]+" | tail -1); \
		[ -n "$$nch" ] && [ "$$nch" -gt 0 ] || { \
			echo "FAIL: chunked decode attention never dispatched (attn_chunk=$${nch:-none})"; \
			echo "      the gate would pass without exercising the path it exists for"; exit 1; }; \
		cmp -s metal-decode1-cpu.out metal-decode1-gpu.out || { \
			echo "FAIL: batch-1-only Metal output differs from CPU"; \
			echo "  cpu: "; head -c 300 metal-decode1-cpu.out; echo; \
			echo "  gpu: "; head -c 300 metal-decode1-gpu.out; echo; \
			exit 1; }; \
		echo "metal decode-only ok (every forward n=1, $$nch chunked dispatches, byte-identical)"; \
	else \
		echo "metal decode-only smoke skipped: no Metal device reported by --caps"; \
	fi
else
	@echo "metal decode-only smoke skipped: macOS-only backend"
endif

# A split GGUF is mapped one region PER PART, and gguf.h is explicit that
# A split GGUF has one mapping PER PART, and the weight wraps are keyed by
# host address, so each part's mapping gets its own tensor-boundary wraps and
# a multi-part set takes a FULL Metal offload like a single file. What cannot
# work is a partial layer split across parts (the prefix arithmetic is
# file-offset-based and parts are separate mappings), so --gpu-layers on a
# split set refuses to CPU rather than guessing.
#
# The gate: the split set offloads and its output is byte-identical BOTH to
# its own CPU run and to the single-file model it was split from — the same
# weights through two mapping layouts must be the same model. The forced
# --gpu-layers arm pins the loud refusal, and the single-file control keeps
# the multi-part path from quietly widening.
test-metal-split: runner
ifeq ($(BUILD_OS),Darwin)
	@set -e; \
	if ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
		mkdir -p test-gguf-split; \
		$(PYTHON) scripts/make-test-model.py test-gguf-split/whole.gguf; \
		$(PYTHON) scripts/gguf-split.py test-gguf-split/whole.gguf test-gguf-split/part 3; \
		part=test-gguf-split/part-00001-of-00003.gguf; \
		./$(RUNNER_EXE) -m $$part -p "hello" -n 12 --temp 0 --gpu off > metal-split-cpu.out 2>/dev/null; \
		./$(RUNNER_EXE) -m $$part -p "hello" -n 12 --temp 0 --gpu auto > metal-split-gpu.out 2> metal-split-gpu.err; \
		if grep -q "not trustworthy" metal-split-gpu.err; then \
			echo "FAIL: the backend addressed weights it has not wrapped and computed anyway"; \
			cat metal-split-gpu.err; exit 1; fi; \
		grep -q "Metal backend" metal-split-gpu.err || { \
			echo "FAIL: a split GGUF did not engage the Metal backend"; \
			cat metal-split-gpu.err; exit 1; }; \
		grep -q "weights copied" metal-split-gpu.err && { \
			echo "FAIL: the split set fell back to a copied buffer"; exit 1; }; \
		cmp -s metal-split-cpu.out metal-split-gpu.out || { \
			echo "FAIL: split-set Metal output differs from --gpu off"; exit 1; }; \
		./$(RUNNER_EXE) -m test-gguf-split/whole.gguf -p "hello" -n 12 --temp 0 --gpu auto \
			> metal-split-whole.out 2> metal-split-whole.err; \
		grep -q "Metal backend" metal-split-whole.err || { \
			echo "FAIL: the single-file control did not offload"; \
			cat metal-split-whole.err; exit 1; }; \
		cmp -s metal-split-gpu.out metal-split-whole.out || { \
			echo "FAIL: the split set and the file it was split from disagree"; exit 1; }; \
		./$(RUNNER_EXE) -m $$part -p "hello" -n 12 --temp 0 --gpu auto --gpu-layers 1 \
			> metal-split-partial.out 2> metal-split-partial.err; \
		grep -q "split GGUF" metal-split-partial.err || { \
			echo "FAIL: --gpu-layers on a split set did not refuse by name"; \
			cat metal-split-partial.err; exit 1; }; \
		cmp -s metal-split-cpu.out metal-split-partial.out || { \
			echo "FAIL: the partial-split refusal did not land on the CPU path"; exit 1; }; \
		echo "metal split GGUF ok (full offload byte-identical to CPU and to the unsplit file; --gpu-layers refuses)"; \
	else \
		echo "metal split GGUF smoke skipped: no Metal device reported by --caps"; \
	fi
else
	@echo "metal split GGUF smoke skipped: macOS-only backend"
endif

# A weight range no wrap holds must END the offload, not decorate it.
#
# metal_bind_weights() used to print "results from this model are not
# trustworthy", bind buffer 0 at the unresolvable offset and let the forward
# run — which is how a split GGUF produced numbers for as long as that path
# existed. The split case is refused at load now, so the remaining ways in are
# internal, and an internal bug is the LAST place to keep computing. This is
# the shape every other failure here already has: fail, fall back, be loud.
#
# RUNNER_METAL_INJECT_BIND_FAILURE is the only way to reach it deterministically
# — the same deliberate-hook approach docs/metal-fallback.md documents for the
# command-buffer failure it cannot induce on real hardware.
test-metal-bind-failure: runner test.gguf
ifeq ($(BUILD_OS),Darwin)
	@set -e; \
	if ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
		./$(RUNNER_EXE) -m test.gguf -p "hello" -n 12 --temp 0 --gpu off > metal-bind-cpu.out 2>/dev/null; \
		env RUNNER_METAL_INJECT_BIND_FAILURE=1 ./$(RUNNER_EXE) -m test.gguf -p "hello" -n 12 \
			--temp 0 --gpu auto > metal-bind-gpu.out 2> metal-bind-gpu.err; \
		grep -q "falling back to CPU" metal-bind-gpu.err || { \
			echo "FAIL: an unresolvable weight binding did not end the offload"; \
			cat metal-bind-gpu.err; exit 1; }; \
		if grep -q "not trustworthy" metal-bind-gpu.err; then \
			echo "FAIL: the backend announced untrustworthy results instead of failing"; \
			exit 1; fi; \
		cmp -s metal-bind-cpu.out metal-bind-gpu.out || { \
			echo "FAIL: the fallback result differs from --gpu off"; exit 1; }; \
		echo "metal bind failure ok (offload ends, run completes on the CPU)"; \
	else \
		echo "metal bind failure smoke skipped: no Metal device reported by --caps"; \
	fi
else
	@echo "metal bind failure smoke skipped: macOS-only backend"
endif

# The fp4 KV cache on Metal: the store kernel byte-identical to the CPU
# encoder (test_metal_kvfp4.m), the banner proving the fp4 cache was in
# use, and the tolerance gate's fp4 arms (GPU fp4 against CPU fp4 within the
# reassociation floor). Like test-metal-kv-q8 it wants a real small model.
# the store-kernel binary is Objective-C: a prerequisite only on Darwin, so
# test-metal (which make test runs everywhere) skips cleanly elsewhere
ifeq ($(BUILD_OS),Darwin)
TEST_METAL_KV_FP4_DEPS = $(TEST_METAL_KVFP4)
else
TEST_METAL_KV_FP4_DEPS =
endif
test-metal-kv-fp4: runner $(TEST_KV_TOL) $(TEST_METAL_KV_FP4_DEPS)
ifeq ($(BUILD_OS),Darwin)
	@set -e; \
	if ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
		./$(TEST_METAL_KVFP4); \
		model="$${ASAN_MODEL:-models/SmolLM2-135M-Instruct-Q8_0.gguf}"; \
		if [ ! -f "$$model" ]; then echo "metal fp4 KV smoke skipped: $$model not found"; exit 0; fi; \
		./$(RUNNER_EXE) -m "$$model" -p "hello" -n 1 --kv fp4 --gpu auto -v 2> metal-kv-fp4.err >/dev/null; \
		grep -q "(fp4)" metal-kv-fp4.err || { echo "FAIL: the fp4 cache was not in use on Metal"; cat metal-kv-fp4.err; exit 1; }; \
		./$(TEST_KV_TOL) "$$model"; \
		rm -f metal-kv-fp4.err; \
	else \
		echo "metal fp4 KV smoke skipped: no Metal device reported by --caps"; \
	fi
else
	@echo "metal fp4 KV smoke skipped: macOS-only backend"
endif

test-metal-kv-q8: runner $(TEST_KV_TOL)
ifeq ($(BUILD_OS),Darwin)
	@set -e; \
	if ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
		model="$${ASAN_MODEL:-models/SmolLM2-135M-Instruct-Q8_0.gguf}"; \
		if [ ! -f "$$model" ]; then echo "metal q8 KV smoke skipped: $$model not found"; exit 0; fi; \
		./$(RUNNER_EXE) -m "$$model" -p "hello" -n 1 --kv q8 --gpu auto -v 2> metal-kv-q8.err >/dev/null; \
		grep -q "q8_0" metal-kv-q8.err; \
		./$(TEST_KV_TOL) "$$model"; \
		echo "metal q8 KV ok"; \
	else \
		echo "metal q8 KV smoke skipped: no Metal device reported by --caps"; \
	fi
else
	@echo "metal q8 KV smoke skipped: macOS-only backend"
endif

test-metal-moe: runner
ifeq ($(BUILD_OS),Darwin)
	@set -e; \
	if ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
		$(PYTHON) scripts/make-test-moe.py test-moe-fixture; \
		./$(RUNNER_EXE) -m test-moe-fixture.dense.gguf -p "hello world" -n 12 --temp 0 --gpu off > metal-moe-dense.out 2>/dev/null; \
		env RUNNER_METAL_MM=0 RUNNER_METAL_ATTN_COOP=0 RUNNER_METAL_MV=0 RUNNER_METAL_MOE_MM=0 ./$(RUNNER_EXE) -m test-moe-fixture.moe1.gguf -p "hello world" -n 12 --temp 0 --gpu auto > metal-moe1.out 2> metal-moe1.err; \
		cmp -s metal-moe-dense.out metal-moe1.out; \
		grep -q "Metal backend" metal-moe1.err; \
		env RUNNER_METAL_MM=0 RUNNER_METAL_ATTN_COOP=0 RUNNER_METAL_MV=0 RUNNER_METAL_MOE_MM=0 ./$(RUNNER_EXE) -m test-moe-fixture.moe2.gguf -p "hello world" -n 12 --temp 0 --gpu auto > metal-moe2.out 2> metal-moe2.err; \
		cmp -s metal-moe-dense.out metal-moe2.out; \
		grep -q "Metal backend" metal-moe2.err; \
		echo "metal MoE ok"; \
	else \
		echo "metal MoE smoke skipped: no Metal device reported by --caps"; \
	fi
else
	@echo "metal MoE smoke skipped: macOS-only backend"
endif

test-metal-gptoss-moe: runner
ifeq ($(BUILD_OS),Darwin)
	@set -e; \
	if ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
		$(PYTHON) scripts/make-test-moe.py test-moe-fixture; \
		./$(RUNNER_EXE) -m test-moe-fixture.gptoss-mxfp4.gguf -p "hello world" -n 8 --temp 0 --gpu off > metal-gptoss-moe-cpu.out 2>/dev/null; \
		env RUNNER_METAL_MM=0 RUNNER_METAL_ATTN_COOP=0 RUNNER_METAL_MV=0 RUNNER_METAL_MOE_MM=0 ./$(RUNNER_EXE) -m test-moe-fixture.gptoss-mxfp4.gguf -p "hello world" -n 8 --temp 0 --gpu auto > metal-gptoss-moe-gpu.out 2> metal-gptoss-moe-gpu.err; \
		cmp -s metal-gptoss-moe-cpu.out metal-gptoss-moe-gpu.out; \
		grep -q "Metal backend" metal-gptoss-moe-gpu.err; \
		echo "metal gpt-oss MoE ok"; \
	else \
		echo "metal gpt-oss MoE smoke skipped: no Metal device reported by --caps"; \
	fi
else
	@echo "metal gpt-oss MoE smoke skipped: macOS-only backend"
endif

# CUDA codebook i-quants: the device decoders against the CPU ones at logit
# precision on the llama-quantize fixtures of all seven types (the GPU leg of
# tests/test_iquants.py), REQUIRED rather than skippable: a box that lacks a
# device, the fixtures, or the gate binary FAILS this target instead of
# reading green. The fixtures are quantized with the llama.cpp tools in
# RUNNER_LLAMA_CPP_BIN, or read pre-built from RUNNER_IQ_FIXTURES on a box
# without them (the Windows CUDA box); the llama.cpp comparison leg runs
# where llama-server is present and is not required here. Part of the
# release evidence on both CUDA classes since 0.5.3.
test-cuda-iquants: runner $(TEST_GPU_ID)
	RUNNER_REQUIRE_IQ_GATES=gpu $(PYTHON) -m pytest -q tests/test_iquants.py

# CUDA Q3_K decode (k_gemv_q3_K): a fixture quantized to Q3_K, the device
# against the CPU at logit precision. Q3_K's 110-byte blocks put three of
# every four at an address that is not 8-byte aligned; the GEMV read them
# with 8-byte loads and faulted, every Q3_K file then decoded on the CPU
# (Qwen2.5-32B Q3_K_S at 0.85 tok/s on the datacenter MIG slice), and no
# gate saw it because the identity run finished on the CPU and compared it
# with itself. test-gpu-identity now fails a run that loses its backend.
test-cuda-q3k: runner $(TEST_GPU_ID)
	@set -e; \
	$(PYTHON) scripts/make-test-model.py --wide test-q3k-src.gguf > /dev/null; \
	./$(RUNNER_EXE) -m test-q3k-src.gguf --quantize test-q3k.gguf --quant q3_k > /dev/null 2>&1; \
	./$(TEST_GPU_ID) test-q3k.gguf

# llama-quantize from RUNNER_LLAMA_CPP_BIN for the CUDA gate steps that
# quantize fixtures with it, resolved by name: `lq=$$($(LLAMA_QUANTIZE))` sets
# lq to $RUNNER_LLAMA_CPP_BIN/llama-quantize or .../llama-quantize.exe,
# whichever is an executable file, or fails with the directory and both names
# on stderr. Never a glob: a llama.cpp Windows release directory holds
# llama-quantize-impl.dll beside llama-quantize.exe, "-" sorts before "."
# under C collation, and the `llama-quantize*` glob these steps used picked
# the DLL (the 1.6.0 CUDA gate on the Windows CUDA host; under a UTF-8 locale
# the exe sorted first and the same step had passed). The .exe name is tried
# first: MSYS2's stat resolves a bare name to its .exe, so the other order
# printed llama-quantize for llama-quantize.exe (the windows-suite job, every
# push from d8544b6f to f960e470). `make llama-quantize-path` prints what the
# steps run (tests/test_ci_contracts.py).
LLAMA_QUANTIZE = d="$$RUNNER_LLAMA_CPP_BIN"; \
	[ -n "$$d" ] || { echo "llama-quantize: RUNNER_LLAMA_CPP_BIN is not set" >&2; exit 1; }; \
	for n in llama-quantize.exe llama-quantize; do \
	  if [ -f "$$d/$$n" ] && [ -x "$$d/$$n" ]; then echo "$$d/$$n"; exit 0; fi; \
	done; \
	echo "llama-quantize: neither llama-quantize nor llama-quantize.exe is an executable file in RUNNER_LLAMA_CPP_BIN ($$d)" >&2; exit 1
.PHONY: llama-quantize-path
llama-quantize-path:
	@$(LLAMA_QUANTIZE)

# The tensor-core prefill of the K-quants and legacy quants that had none
# until 2026-10-21 (kernels_kq_stage.cuh, decoders held to dequant_row by
# test-kq-stage), through test-tc-tol's teacher-forced tolerance gate: a
# 256-wide fixture at Q3_K (runner --quantize) and, where RUNNER_LLAMA_CPP_BIN
# has llama-quantize, at Q5_K, Q2_K, Q4_1, Q5_0 and Q5_1, each on the TC2
# kernels TC2's auto mode picks (compute_80 at 8.0 and up), on the
# compute_75 TC2 kernels and on the TC route's kernels. test-tc-tol forces
# the tensor-core route on, so the legs run whether or not a type is
# promoted.
test-cuda-tc-kq: runner $(TEST_TC_TOL)
	@set -e; \
	$(PYTHON) scripts/make-test-model.py --wide test-tckq-wide.gguf > /dev/null; \
	./$(RUNNER_EXE) -m test-tckq-wide.gguf --quantize test-tckq-q3_k.gguf --quant q3_k > /dev/null 2>&1; \
	files="test-tckq-q3_k.gguf"; \
	if lq=$$($(LLAMA_QUANTIZE)); then \
	  for q in Q5_K_S Q2_K Q4_1 Q5_0 Q5_1; do \
	    "$$lq" test-tckq-wide.gguf test-tckq-$$q.gguf $$q > /dev/null 2>&1 && files="$$files test-tckq-$$q.gguf" || \
	      echo "test-cuda-tc-kq: $$lq $$q failed, $$q not exercised"; \
	  done; \
	else echo "test-cuda-tc-kq: no llama-quantize in RUNNER_LLAMA_CPP_BIN, only Q3_K exercised"; fi; \
	for f in $$files; do \
	  for leg in default "RUNNER_CUDA_TC2=1 RUNNER_CUDA_SM80=0" RUNNER_CUDA_TC2=0; do \
	    e=$$leg; [ "$$leg" = default ] && e=; \
	    env $$e ./$(TEST_TC_TOL) $$f > test-cuda-tc-kq.out 2>&1 || \
	      { cat test-cuda-tc-kq.out; echo "test-cuda-tc-kq: $$f [$$leg] FAILED"; exit 1; }; \
	    if ! grep -q 'tc-tol: ok' test-cuda-tc-kq.out || grep -q 'skipped' test-cuda-tc-kq.out; then \
	      cat test-cuda-tc-kq.out; echo "test-cuda-tc-kq: $$f [$$leg] did not run"; exit 1; fi; \
	    echo "$$f $$leg: $$(grep 'tc-tol: ok' test-cuda-tc-kq.out)"; \
	  done; \
	done

# MoE experts in the 32-element legacy types on the fused device routing
# (k_moe_mv_q5_0 / q4_1 / q5_1, 2026-10-21; until then a model with such
# experts, llama-quantize's Q4_K_M of gemma-4-26B-A4B among them, ran its MoE
# layers on the eager path): the moe4 fixture quantized to each type with
# llama-quantize (RUNNER_LLAMA_CPP_BIN), held to the eager path by
# test-moe-tol's teacher-forced gate. test-moe-tol skips when the two paths
# are bit-identical, which here means the fused path never engaged, so a skip
# fails the step. Without llama-quantize there is no fixture, so the step
# fails rather than narrowing its coverage as the other llama-quantize steps
# do: a CUDA gate host needs it.
test-cuda-moe-legacy: runner $(TEST_MOE_TOL)
	@set -e; \
	lq=$$($(LLAMA_QUANTIZE)) || { echo "test-cuda-moe-legacy: needs llama-quantize in RUNNER_LLAMA_CPP_BIN"; exit 1; }; \
	$(PYTHON) scripts/make-test-moe.py test-moe-legacy > /dev/null; \
	for q in Q5_0 Q4_1 Q5_1; do \
	  f=test-moe-legacy-$$q.gguf; \
	  "$$lq" test-moe-legacy.moe4.gguf $$f $$q > test-cuda-moe-legacy.out 2>&1 || \
	    { tail -n 20 test-cuda-moe-legacy.out; echo "test-cuda-moe-legacy: $$lq $$q failed"; exit 1; }; \
	  ./$(TEST_MOE_TOL) $$f > test-cuda-moe-legacy.out 2>&1 || \
	    { cat test-cuda-moe-legacy.out; echo "test-cuda-moe-legacy: $$q FAILED"; exit 1; }; \
	  if ! grep -q 'moe-tol: ok' test-cuda-moe-legacy.out || grep -qi 'skip' test-cuda-moe-legacy.out; then \
	    cat test-cuda-moe-legacy.out; echo "test-cuda-moe-legacy: $$q did not run the fused path"; exit 1; fi; \
	  echo "$$f: $$(grep 'moe-tol: ok' test-cuda-moe-legacy.out)"; \
	done

# The by-expert MoE prefill (moe_twin) against the fused per-token path it
# must reproduce byte for byte, on a 32-row prefill: the routed top-2-of-4
# fixture with its experts requantized to Q8_0, the gpt-oss fixture (MXFP4
# experts, per-expert gate, up and down biases) and gemma-4's dual-branch
# fixture at Q8_0 (dense shared MLP plus routed experts), and both again at
# Q4_0, whose fused expert kernel is the lane-per-block k_mv_q4_0 body rather
# than the coalesced GEMV the other types share (gemma-4-26B-A4B's Q4_0
# experts once came out of the by-expert path with other bytes). The Q8_0
# pair runs once more under RUNNER_MOE_GROUPED=1: the grouped prefill takes a
# tile ahead of both routes, so the pair holds it off for its own runs (a
# gate with every opt-in on once found the pair with nothing to compare).
test-cuda-moe-twin: runner $(TEST_GPU_ID)
	@set -e; \
	$(PYTHON) scripts/make-test-moe.py test-moe-twin > /dev/null; \
	./$(RUNNER_EXE) -m test-moe-twin.moe4.gguf --quantize test-moe-twin-q8.gguf --quant q8_0 > /dev/null 2>&1; \
	GPU_ID_MOE_TWIN=1 ./$(TEST_GPU_ID) test-moe-twin-q8.gguf; \
	RUNNER_MOE_GROUPED=1 GPU_ID_MOE_TWIN=1 ./$(TEST_GPU_ID) test-moe-twin-q8.gguf; \
	GPU_ID_MOE_TWIN=1 ./$(TEST_GPU_ID) test-moe-twin.gptoss-mxfp4.gguf; \
	./$(RUNNER_EXE) -m test-moe-twin.gemma4-moe.gguf --quantize test-moe-twin-g4.gguf --quant q8_0 > /dev/null 2>&1; \
	GPU_ID_MOE_TWIN=1 ./$(TEST_GPU_ID) test-moe-twin-g4.gguf; \
	./$(RUNNER_EXE) -m test-moe-twin.moe4.gguf --quantize test-moe-twin-q40.gguf --quant q4_0 > /dev/null 2>&1; \
	GPU_ID_MOE_TWIN=1 ./$(TEST_GPU_ID) test-moe-twin-q40.gguf; \
	./$(RUNNER_EXE) -m test-moe-twin.gemma4-moe.gguf --quantize test-moe-twin-g4q40.gguf --quant q4_0 > /dev/null 2>&1; \
	GPU_ID_MOE_TWIN=1 ./$(TEST_GPU_ID) test-moe-twin-g4q40.gguf

# The decode microbatch on CUDA, byte for byte against sequential decode
# (docs/cuda-microbatch-identity-2026-08-18.md) on the F32, Q8_0 and BF16
# fixtures, a 256-wide model at Q4_K, Q6_K, Q4_0 and Q3_K, the five codebook
# i-quants (make-test-model.py --quant) and, where RUNNER_LLAMA_CPP_BIN has
# llama-quantize, Q5_K and IQ4_XS, with the default routes and with the
# grouped decode attention off (the two decode attentions the microbatch
# twins). `make test` runs the F32 case where it builds, which on a Mac is
# Metal, so until 2026-10-21 no gate held the CUDA microbatch: making the
# grouped decode attention the default left the microbatch on the old kernel,
# and no gate saw it; IQ4_XS's twins were the lane kernels' twins while its
# single step ran the coalesced GEMV (since #243), and no gate had an IQ4_XS
# file; a codebook i-quant file never batched at all. A host without a CUDA
# device fails here, as a gate that did not run.
test-cuda-batch-identity: runner $(TEST_BATCH_ID) test.gguf test-q8.gguf test-bf16.gguf
	@set -e; \
	$(PYTHON) scripts/make-test-model.py --wide test-batch-wide.gguf > /dev/null; \
	files="test.gguf test-q8.gguf test-bf16.gguf"; \
	for q in q4_k q6_k q4_0 q3_k; do \
	  ./$(RUNNER_EXE) -m test-batch-wide.gguf --quantize test-batch-wide-$$q.gguf --quant $$q > /dev/null 2>&1; \
	  files="$$files test-batch-wide-$$q.gguf"; \
	done; \
	for q in iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s; do \
	  $(PYTHON) scripts/make-test-model.py --quant $$q test-batch-$$q.gguf > /dev/null; \
	  files="$$files test-batch-$$q.gguf"; \
	done; \
	if lq=$$($(LLAMA_QUANTIZE)); then \
	  for q in Q5_K_S IQ4_XS; do \
	    "$$lq" test-batch-wide.gguf test-batch-wide-$$q.gguf $$q > /dev/null 2>&1 && files="$$files test-batch-wide-$$q.gguf" || \
	      echo "test-cuda-batch-identity: $$lq $$q failed, $$q not exercised"; \
	  done; \
	else echo "test-cuda-batch-identity: no llama-quantize in RUNNER_LLAMA_CPP_BIN, the Q5_K and IQ4_XS twins not exercised"; fi; \
	for f in $$files; do \
	  for dec in 1 0; do \
	    RUNNER_CUDA_ATTN_DEC=$$dec ./$(TEST_BATCH_ID) $$f > test-cuda-batch-identity.out 2>&1 || \
	      { cat test-cuda-batch-identity.out; echo "test-cuda-batch-identity: $$f attn-dec=$$dec FAILED"; exit 1; }; \
	    if grep -q 'skipped' test-cuda-batch-identity.out; then \
	      echo "test-cuda-batch-identity: skipped (no CUDA device)"; exit 1; fi; \
	    echo "$$f attn-dec=$$dec: $$(tail -1 test-cuda-batch-identity.out)"; \
	  done; \
	done

# A speculative verify round's rows on CUDA, byte for byte against solo
# decode steps (tests/test_spec_identity.c, rounds 2 to 16 rows with the
# walk's rollback): the dense fixtures (F32, Q8_0, BF16), a 256-wide model at
# Q4_K, Q6_K and Q3_K, the codebook i-quants IQ2_XXS and IQ3_S (and IQ4_XS
# where RUNNER_LLAMA_CPP_BIN has llama-quantize), the MoE ones (Q8_0, Q4_0, gpt-oss MXFP4, gemma-4's dual
# branch), the Gated DeltaNet tracer and a dense Mamba-2 hybrid, with the
# default routes, then the grouped MoE prefill and the grouped decode
# attention forced each way. Until 2026-10-21 a CUDA verify row ran the
# prompt kernels, so a drafted run equalled the plain one only up to
# near-ties (4 of 12 runs differed on the MoE pairs).
SPEC_ID_RUN = ./$(TEST_SPEC_ID) $$f > test-cuda-spec-identity.out 2>&1 || { cat test-cuda-spec-identity.out; exit 1; }; \
	  if grep -q 'cpu,\|skipped' test-cuda-spec-identity.out; then \
	    cat test-cuda-spec-identity.out; echo "test-cuda-spec-identity: skipped (no CUDA device for $$f)"; exit 1; fi; \
	  echo "$$f $$env: $$(tail -1 test-cuda-spec-identity.out)"
test-cuda-spec-identity: runner $(TEST_SPEC_ID) test.gguf test-q8.gguf test-bf16.gguf test-ornith-draft.gguf
	@set -e; \
	$(PYTHON) scripts/make-test-moe.py test-spec-moe > /dev/null; \
	./$(RUNNER_EXE) -m test-spec-moe.moe4.gguf --quantize test-spec-moe-q8.gguf --quant q8_0 > /dev/null 2>&1; \
	./$(RUNNER_EXE) -m test-spec-moe.moe4.gguf --quantize test-spec-moe-q40.gguf --quant q4_0 > /dev/null 2>&1; \
	./$(RUNNER_EXE) -m test-spec-moe.gemma4-moe.gguf --quantize test-spec-moe-g4.gguf --quant q8_0 > /dev/null 2>&1; \
	$(PYTHON) scripts/make-test-model.py --wide test-spec-wide.gguf > /dev/null; \
	./$(RUNNER_EXE) -m test-spec-wide.gguf --quantize test-spec-wide-q4k.gguf --quant q4_k > /dev/null 2>&1; \
	./$(RUNNER_EXE) -m test-spec-wide.gguf --quantize test-spec-wide-q6k.gguf --quant q6_k > /dev/null 2>&1; \
	./$(RUNNER_EXE) -m test-spec-wide.gguf --quantize test-spec-wide-q3k.gguf --quant q3_k > /dev/null 2>&1; \
	$(PYTHON) scripts/make-test-model.py --quant iq2_xxs test-spec-iq2_xxs.gguf > /dev/null; \
	$(PYTHON) scripts/make-test-model.py --quant iq3_s test-spec-iq3_s.gguf > /dev/null; \
	$(PYTHON) scripts/make-test-hybrid.py test-spec-hybrid --dense --rope > /dev/null; \
	files="test.gguf test-q8.gguf test-bf16.gguf test-spec-wide-q4k.gguf test-spec-wide-q6k.gguf \
	       test-spec-wide-q3k.gguf test-spec-iq2_xxs.gguf test-spec-iq3_s.gguf \
	       test-spec-moe-q8.gguf test-spec-moe-q40.gguf test-spec-moe.gptoss-mxfp4.gguf test-spec-moe-g4.gguf \
	       test-ornith-draft.gguf test-spec-hybrid.gguf"; \
	if lq=$$($(LLAMA_QUANTIZE)); then \
	  "$$lq" test-spec-wide.gguf test-spec-wide-IQ4_XS.gguf IQ4_XS > /dev/null 2>&1 && files="$$files test-spec-wide-IQ4_XS.gguf" || \
	    echo "test-cuda-spec-identity: $$lq IQ4_XS failed, IQ4_XS not exercised"; \
	else echo "test-cuda-spec-identity: no llama-quantize in RUNNER_LLAMA_CPP_BIN, the IQ4_XS twins not exercised"; fi; \
	env=default; for f in $$files; do $(SPEC_ID_RUN); done; \
	env=RUNNER_MOE_GROUPED=1; f=test-spec-moe-q8.gguf; export RUNNER_MOE_GROUPED=1; $(SPEC_ID_RUN); unset RUNNER_MOE_GROUPED; \
	export RUNNER_CUDA_ATTN_DEC=0; env=RUNNER_CUDA_ATTN_DEC=0; \
	for f in test-q8.gguf test-spec-moe.gptoss-mxfp4.gguf; do $(SPEC_ID_RUN); done

# Per-layer FFN widths on CUDA (the gemma-4 E2B export shape: widths that
# alternate between layers, an ARRAY-typed feed_forward_length): the device
# runs each layer at its own width against the CPU at logit precision, in
# f32 and requantized to q8_0 (the tensor-core prefill and the q8 decode
# kernels). Until 2026-10-20 CUDA refused these files and they ran on the CPU.
test-cuda-varff: runner $(TEST_GPU_ID)
	@set -e; \
	$(PYTHON) scripts/make-test-model.py --eseries 3,16 --ffn-widths 96,96,192,96,96,192 test-varff.gguf > /dev/null; \
	./$(RUNNER_EXE) -m test-varff.gguf --quantize test-varff-q8.gguf --quant q8_0 > /dev/null 2>&1; \
	./$(TEST_GPU_ID) test-varff.gguf; \
	./$(TEST_GPU_ID) test-varff-q8.gguf

# TC2, on by default on compute capability 8.0+ (its compute_80 kernels),
# against the tensor-core route on the device the gate runs on: chunked
# --score byte for byte on a Q8_0 and a Q4_K fixture, with the launch counts
# (scripts/cuda-tc2-bytes.py). The CPU-against-GPU harnesses pin the
# reference route, so this is the step that checks the default kernels on a
# card nobody has measured. Skips itself without a CUDA device.
test-cuda-tc2-bytes: runner
	@set -e; \
	$(PYTHON) scripts/make-test-model.py --wide --quant q8_0 test-tc2b-q8.gguf > /dev/null; \
	$(PYTHON) scripts/make-test-model.py --wide test-tc2b-src.gguf > /dev/null; \
	./$(RUNNER_EXE) -m test-tc2b-src.gguf --quantize test-tc2b-q4k.gguf --quant q4_k > /dev/null 2>&1; \
	$(PYTHON) scripts/cuda-tc2-bytes.py ./$(RUNNER_EXE) test-tc2b-q8.gguf test-tc2b-q4k.gguf

# The same gate on a Mac: the Metal codebook i-quant kernels (2026-09-17)
# against the CPU decoders at logit precision on the fixture set, which
# tests/test_iquants.py builds with llama-quantize/llama-imatrix from
# RUNNER_LLAMA_CPP_BIN (Homebrew's llama.cpp: RUNNER_LLAMA_CPP_BIN=/opt/homebrew/bin).
test-metal-iquants: runner $(TEST_GPU_ID)
	RUNNER_REQUIRE_IQ_GATES=gpu $(PYTHON) -m pytest -q tests/test_iquants.py

# The fp16 range of the tensor-core operands: a fixture whose FFN activation
# is 1e7 at the ffn_down input (finite in fp32, far past fp16's 65504)
# through the forced tensor-core gate. Every CUDA TC kernel stages each
# token column scaled to fp16's range since 2026-09-14; before that the
# staged operand was Inf and every logit NaN, which the gate now fails
# outright. Runs on any CUDA box (the gate skips itself, and this target
# passes, where there is no device). Metal's tiled path stages activations
# the same way and was the last unscaled one until 2026-09-17 (k_colabsmax
# + the scaled MM_BODY in kernels.metal), so the gate runs on Darwin too.
test-tc-overflow: runner $(TEST_TC_TOL)
	@set -e; \
	$(PYTHON) scripts/make-test-model.py --wide --quant q8_0 --act-fp16-overflow test-fp16ovf.gguf > /dev/null; \
	./$(TEST_TC_TOL) test-fp16ovf.gguf > tc-overflow.out 2>&1 || { cat tc-overflow.out; exit 1; }; \
	grep -q "tc-tol: ok" tc-overflow.out || { cat tc-overflow.out; exit 1; }; \
	grep -q "ok (skipped)" tc-overflow.out && echo "tc overflow gate: skipped (no CUDA device)" || \
	  { grep -q "TC dispatches: Q8_0=" tc-overflow.out || { cat tc-overflow.out; exit 1; }; echo "tc overflow gate ok"; }

# CUDA qwen4exp (Qwen3.8-Flash-Next): the device path against the CPU path at
# logit precision on the family fixture, in the three placements the slice
# runs it in. Each arm pins a defect the greedy-64 identity check of
# 2026-10-06 did not see (2026-10-07): the 512-expert fixture pins the eager
# router's expert bound (it selected among the first 256 of 512, on the
# eager path a fused-ineligible expert type forces: the "eager" arm runs it
# through RUNNER_MOE_EAGER=1, the #243 binary flips 3 of 56 there); the forced
# partial split (--gpu-layers 2) pins the boundary hand-off of the wide
# hyper-connection residual (the CPU layers continued from a stale stream);
# device experts and host experts both run; and a 2-slot server on the
# fixture split in two parts must share one device copy (a split GGUF had
# no shared-weights identity, so slot 2 built privately, found no VRAM and
# fell to the CPU: two slots, two answers); and the NextN/MTP head on the
# device must draft the CPU head's logits step for step (same argmax,
# top logit and top-2 gap within reduction-order residue) and leave the
# text identical (RUNNER_MTP_DUMP on the fixture's MTP block, an
# 86-token prompt so the head's queue drains in two tiles); and the BF16
# split-K decode GEMV, on a fixture widened to reach it (512-wide residual,
# BF16 hc mixers: a 2,048-wide mixer input), must launch, stay inside the
# CPU bound, and give the same bytes on a second device run. Skips itself
# without a CUDA device. 2e-5 is reduction-order residue on the fixture
# (measured 4.8e-7).
test-cuda-qwen4exp: runner $(TEST_GPU_ID) $(TEST_CUDA_TEARDOWN) test.gguf
	@set -e; \
	if ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'cuda' else 1)"; then \
		$(PYTHON) scripts/make-test-qwen4exp.py test-qwen4exp.gguf > /dev/null; \
		QWEN4EXP_TEST_EXPERTS=512 $(PYTHON) scripts/make-test-qwen4exp.py test-qwen4exp-512.gguf > /dev/null; \
		for f in test-qwen4exp.gguf test-qwen4exp-512.gguf; do \
		  ./$(RUNNER_EXE) -b 1 -m $$f --score -p "hello world one two three four five six" --gpu off --no-tray -c 128 2>/dev/null > cuda-q4e-cpu.score; \
		  for arm in "" "--cpu-moe" "--gpu-layers 2" "--gpu-layers 2 --cpu-moe" "eager"; do \
		    E=""; if [ "$$arm" = eager ]; then E="RUNNER_MOE_EAGER=1"; arm=""; fi; \
		    env $$E ./$(RUNNER_EXE) -b 1 -m $$f --score -p "hello world one two three four five six" $$arm --no-tray -c 128 2>cuda-q4e-gpu.err > cuda-q4e-gpu.score; \
		    [ -n "$$E" ] && arm="eager experts"; \
		    grep -q "CUDA backend" cuda-q4e-gpu.err || { echo "FAIL: $$f [$$arm] did not run on the device"; cat cuda-q4e-gpu.err; exit 1; }; \
		    $(PYTHON) -c "import json,sys; a=json.load(open('cuda-q4e-cpu.score')); b=json.load(open('cuda-q4e-gpu.score')); d=max(abs(x-y) for x,y in zip(a['logprobs'],b['logprobs'])); am=sum(1 for x,y in zip(a['argmax'],b['argmax']) if x!=y); assert len(a['logprobs'])==len(b['logprobs'])==56 and d < 2e-5 and am == 0, (len(a['logprobs']), len(b['logprobs']), d, am); print('%s [%s]: cpu/cuda logprob max diff %.2e, argmax flips %d' % (sys.argv[1], sys.argv[2], d, am))" $$f "$$arm"; \
		  done; \
		done; \
		rm -f cuda-q4e-cpu.score cuda-q4e-gpu.score cuda-q4e-gpu.err test-qwen4exp-512.gguf; \
		./$(TEST_CUDA_TEARDOWN) test.gguf test-qwen4exp.gguf 3; \
		QWEN4EXP_TEST_MTP=1 $(PYTHON) scripts/make-test-qwen4exp.py test-q4e-mtp.gguf > /dev/null; \
		P="hello world one two three four five six seven eight nine ten"; \
		rm -f cuda-q4e-mtp-cpu.dump cuda-q4e-mtp-gpu.dump; \
		RUNNER_MTP_DUMP=cuda-q4e-mtp-cpu.dump ./$(RUNNER_EXE) -m test-q4e-mtp.gguf -p "$$P" -n 24 --temp 0 --gpu off --mtp --no-tray -c 128 > cuda-q4e-mtp-cpu.out 2>/dev/null; \
		for arm in "" "--cpu-moe"; do \
		  rm -f cuda-q4e-mtp-gpu.dump; \
		  RUNNER_MTP_DUMP=cuda-q4e-mtp-gpu.dump ./$(RUNNER_EXE) -m test-q4e-mtp.gguf -p "$$P" -n 24 --temp 0 --mtp $$arm --no-tray -c 128 > cuda-q4e-mtp-gpu.out 2>cuda-q4e-mtp-gpu.err; \
		  grep -q "CUDA backend" cuda-q4e-mtp-gpu.err || { echo "FAIL: --mtp [$$arm] did not run on the device"; cat cuda-q4e-mtp-gpu.err; exit 1; }; \
		  $(PYTHON) -c "import sys; a=open('cuda-q4e-mtp-cpu.out','rb').read(); b=open('cuda-q4e-mtp-gpu.out','rb').read(); assert a and a == b, 'generated text differs'; da=[l.split() for l in open('cuda-q4e-mtp-cpu.dump') if l.strip()]; db=[l.split() for l in open('cuda-q4e-mtp-gpu.dump') if l.strip()]; assert len(da) == len(db) > 10, (len(da), len(db)); bad=[(x,y) for x,y in zip(da,db) if x[4] != y[4] or abs(float(x[5])-float(y[5])) > 1e-5 or abs(float(x[7])-float(y[7])) > 1e-5]; assert not bad, 'head logits differ: %s' % bad[:2]; print('mtp head [%s]: %d draft steps, device head == CPU head (argmax, logit and gap within 1e-5), text identical' % (sys.argv[1], len(da)))" "$$arm"; \
		done; \
		rm -f test-q4e-mtp.gguf cuda-q4e-mtp-*; \
		QWEN4EXP_TEST_EMBD=512 QWEN4EXP_TEST_HC_BF16=1 $(PYTHON) scripts/make-test-qwen4exp.py test-q4e-bf16.gguf > /dev/null; \
		GPU_ID_REQUIRE_SPLITK=1 ./$(TEST_GPU_ID) test-q4e-bf16.gguf; \
		rm -f test-q4e-bf16.gguf; \
		$(PYTHON) scripts/gguf-split.py test-qwen4exp.gguf test-q4e-split 2 > /dev/null; \
		./$(RUNNER_EXE) --serve --port 18840 -m test-q4e-split-00001-of-00002.gguf --parallel 2 --no-tray -c 256 > cuda-q4e-serve.log 2>&1 & pid=$$!; \
		for i in $$(seq 1 60); do curl -sf http://127.0.0.1:18840/health > /dev/null 2>&1 && break; sleep 1; done; \
		kill $$pid 2>/dev/null; wait $$pid 2>/dev/null || true; \
		grep -q "reusing resident weights" cuda-q4e-serve.log || { echo "FAIL: the second slot of a split GGUF did not share the first slot's device weights"; cat cuda-q4e-serve.log; exit 1; }; \
		echo "split GGUF, 2 slots: device weights shared"; \
		rm -f test-q4e-split-0000?-of-00002.gguf cuda-q4e-serve.log; \
	else echo "cuda qwen4exp gate: skipped (no CUDA device)"; fi

# CUDA NVFP4 (ModelOpt two-level export): the device kernels and the companion
# scale in their tails against the CPU seam. Token identity on a generated
# fixture and on any real NVFP4 file named in NVFP4_MODEL, logprob agreement
# to reduction-order residue, and proof the device actually ran the layers.
# Skips itself on a box whose --caps report no CUDA device.
test-cuda-nvfp4: runner
	@set -e; \
	if ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'cuda' else 1)"; then \
		$(PYTHON) scripts/make-test-model.py test-nvfp4.gguf --quant nvfp4 > /dev/null; \
		for f in test-nvfp4.gguf $(NVFP4_MODEL); do \
		  ./$(RUNNER_EXE) -m $$f -p "The capital of France is" -n 16 --temp 0 --gpu off --no-tray > cuda-nvfp4-cpu.out 2>/dev/null; \
		  ./$(RUNNER_EXE) -m $$f -p "The capital of France is" -n 16 --temp 0 --gpu auto --no-tray --transcript cuda-nvfp4-gpu.json > cuda-nvfp4-gpu.out 2>cuda-nvfp4-gpu.err; \
		  $(PYTHON) -c "import sys; a=open('cuda-nvfp4-cpu.out','rb').read(); b=open('cuda-nvfp4-gpu.out','rb').read(); sys.exit(0 if a == b and a else 1)" || { echo "FAIL: $$f CPU and CUDA tokens differ"; cat cuda-nvfp4-cpu.out; echo; cat cuda-nvfp4-gpu.out; echo; exit 1; }; \
		  $(PYTHON) -c "import json,sys; p=json.load(open('cuda-nvfp4-gpu.json'))['profile']; assert p['gpu_layers'] > 0, 'CUDA declined ' + sys.argv[1]" $$f; \
		  ./$(RUNNER_EXE) -m $$f --score -p "The capital of France is Paris, and the runner trains the model on hardware you own." --gpu off --no-tray 2>/dev/null > cuda-nvfp4-cpu.score; \
		  ./$(RUNNER_EXE) -m $$f --score -p "The capital of France is Paris, and the runner trains the model on hardware you own." --gpu auto --no-tray 2>/dev/null > cuda-nvfp4-gpu.score; \
		  $(PYTHON) -c "import json,sys; a=json.load(open('cuda-nvfp4-cpu.score'))['logprobs']; b=json.load(open('cuda-nvfp4-gpu.score'))['logprobs']; d=max(abs(x-y) for x,y in zip(a,b)); assert len(a)==len(b)>3 and d < 2e-2, (len(a), len(b), d); print(sys.argv[1], 'cpu/cuda logprob max diff %.2e over %d positions' % (d, len(a)))" $$f; \
		done; \
		echo "cuda nvfp4 ok"; \
	else \
		echo "cuda nvfp4 smoke skipped: no CUDA device reported by --caps"; \
	fi

test-metal-gemma4-moe: runner
ifeq ($(BUILD_OS),Darwin)
	@set -e; \
	if ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
		$(PYTHON) scripts/make-test-moe.py test-moe-fixture; \
		./$(RUNNER_EXE) -m test-moe-fixture.gemma4-moe.gguf -p "hello world" -n 8 --temp 0 --gpu off > metal-gemma4-moe-cpu.out 2>/dev/null; \
		env RUNNER_METAL_MM=0 RUNNER_METAL_ATTN_COOP=0 RUNNER_METAL_MV=0 RUNNER_METAL_MOE_MM=0 ./$(RUNNER_EXE) -m test-moe-fixture.gemma4-moe.gguf -p "hello world" -n 8 --temp 0 --gpu auto > metal-gemma4-moe-gpu.out 2> metal-gemma4-moe-gpu.err; \
		cmp -s metal-gemma4-moe-cpu.out metal-gemma4-moe-gpu.out; \
		grep -q "Metal backend" metal-gemma4-moe-gpu.err; \
		echo "metal gemma4 MoE ok"; \
	else \
		echo "metal gemma4 MoE smoke skipped: no Metal device reported by --caps"; \
	fi
else
	@echo "metal gemma4 MoE smoke skipped: macOS-only backend"
endif

test-metal-gemma4-hetero: runner
ifeq ($(BUILD_OS),Darwin)
	@set -e; \
	if ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
		$(PYTHON) scripts/make-test-model.py --gemma4-hetero test-g4h.gguf; \
		./$(RUNNER_EXE) -m test-g4h.gguf -p "hello world" -n 8 --temp 0 --gpu off > metal-g4h-dense-cpu.out 2>/dev/null; \
		env RUNNER_METAL_MM=0 RUNNER_METAL_ATTN_COOP=0 RUNNER_METAL_MV=0 RUNNER_METAL_MOE_MM=0 ./$(RUNNER_EXE) -m test-g4h.gguf -p "hello world" -n 8 --temp 0 --gpu auto > metal-g4h-dense-gpu.out 2> metal-g4h-dense-gpu.err; \
		cmp -s metal-g4h-dense-cpu.out metal-g4h-dense-gpu.out; \
		grep -q "Metal backend" metal-g4h-dense-gpu.err; \
		$(PYTHON) scripts/make-test-moe.py test-moe-fixture; \
		./$(RUNNER_EXE) -m test-moe-fixture.gemma4-moe-hetero.gguf -p "hello world" -n 8 --temp 0 --gpu off > metal-g4h-moe-cpu.out 2>/dev/null; \
		env RUNNER_METAL_MM=0 RUNNER_METAL_ATTN_COOP=0 RUNNER_METAL_MV=0 RUNNER_METAL_MOE_MM=0 ./$(RUNNER_EXE) -m test-moe-fixture.gemma4-moe-hetero.gguf -p "hello world" -n 8 --temp 0 --gpu auto > metal-g4h-moe-gpu.out 2> metal-g4h-moe-gpu.err; \
		cmp -s metal-g4h-moe-cpu.out metal-g4h-moe-gpu.out; \
		grep -q "Metal backend" metal-g4h-moe-gpu.err; \
		echo "metal gemma4 heterogeneous ok"; \
	else \
		echo "metal gemma4 hetero smoke skipped: no Metal device reported by --caps"; \
	fi
else
	@echo "metal gemma4 hetero smoke skipped: macOS-only backend"
endif

# Big-model Metal identity. The fixture gates above prove the gemma4/gpt-oss
# MoE wiring at sub-1 MiB geometry; they cannot prove the kernels at real
# expert-bundle geometry. On 2026-08-31 a real artifact diverged on a host
# where every fixture gate passed (the 2026-08-31 record is archived in Shade
# under research/runner-engine/archive/runner-docs-2026-10-13b/). Opt-in by path, so a
# checkout without weights stays green:
#   make test-metal-bigmodel BIGMODEL=models/gpt-oss-20b-MXFP4.gguf
# The prompt matters. A raw completion fed to an instruction-tuned model can
# walk into a degenerate loop where adjacent logits tie, and token identity
# there measures chaos rather than correctness: gemma-4-26B-A4B on
# "The capital of France is" diverges at token 14 on a near-tie, and the CPU
# arm disagrees with ITSELF on the same prompt under --kv q8. Default to a
# realistic instruction, override for a specific investigation.
BIGMODEL ?=
BIGPROMPT ?= Explain photosynthesis in two sentences.
test-metal-bigmodel: runner
ifeq ($(BUILD_OS),Darwin)
	@set -e; \
	if [ -z "$(BIGMODEL)" ]; then \
		echo "metal big-model identity skipped: set BIGMODEL=<path.gguf>"; \
	elif [ ! -f "$(BIGMODEL)" ]; then \
		echo "metal big-model identity skipped: $(BIGMODEL) not found"; \
	elif ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
		./$(RUNNER_EXE) -m "$(BIGMODEL)" -p "$(BIGPROMPT)" -n 32 --temp 0 --gpu off > metal-bigmodel-cpu.out 2>/dev/null; \
		env RUNNER_METAL_MM=0 RUNNER_METAL_ATTN_COOP=0 RUNNER_METAL_MV=0 RUNNER_METAL_MOE_MM=0 \
		  ./$(RUNNER_EXE) -m "$(BIGMODEL)" -p "$(BIGPROMPT)" -n 32 --temp 0 --gpu auto > metal-bigmodel-gpu.out 2>/dev/null; \
		if cmp -s metal-bigmodel-cpu.out metal-bigmodel-gpu.out; then \
			echo "metal big-model identity ok: $(BIGMODEL)"; \
		else \
			echo "FAIL: CPU/Metal divergence on $(BIGMODEL)"; \
			echo "  cpu: $$(head -c 120 metal-bigmodel-cpu.out)"; \
			echo "  gpu: $$(head -c 120 metal-bigmodel-gpu.out)"; \
			exit 1; \
		fi; \
	else \
		echo "metal big-model identity skipped: no Metal device reported by --caps"; \
	fi
else
	@echo "metal big-model identity skipped: macOS-only backend"
endif

# Expert-major MoE kernels (RUNNER_METAL_MOE_EM): the twin family shares its
# dot bodies with slot-major, so the contract is BYTE IDENTITY across the
# whole run (prefill engages expert-major, the decode tail stays slot-major —
# both shapes are exercised by one comparison). The engagement grep is what
# keeps the gate non-vacuous: without it, a build whose expert-major
# pipelines fail to compile would fall back to slot-major and pass on
# self-agreement.
test-metal-moe-em: runner test-moe-fixture.moe1.gguf test-moe-fixture.moe4.gguf test-moe-fixture.gptoss-mxfp4.gguf test-moe-fixture.gemma4-moe.gguf
ifeq ($(BUILD_OS),Darwin)
	@set -e; \
	if ! ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
	  echo "metal moe expert-major: SKIP (no Metal device)"; exit 0; fi; \
	prompt="The quick brown fox jumps over the lazy dog again and again"; \
	./$(RUNNER_EXE) -m test-moe-fixture.moe4.gguf \
	  --quantize test-moe-fixture.moe4-q8.gguf --quant q8_0 >/dev/null 2>&1; \
	for f in test-moe-fixture.moe1.gguf test-moe-fixture.moe4.gguf \
	         test-moe-fixture.moe4-q8.gguf \
	         test-moe-fixture.gptoss-mxfp4.gguf test-moe-fixture.gemma4-moe.gguf; do \
	  RUNNER_METAL_MOE_MM=0 ./$(RUNNER_EXE) -m $$f -p "$$prompt" -n 16 -b 32 \
	    --temp 0 --no-tray > moe-em-off.out 2>/dev/null; \
	  RUNNER_METAL_MOE_MM=0 RUNNER_METAL_MOE_EM=1 ./$(RUNNER_EXE) -m $$f \
	    -p "$$prompt" -n 16 -b 32 --temp 0 --no-tray \
	    > moe-em-on.out 2> moe-em-on.err; \
	  grep -q "expert-major prefill kernels on" moe-em-on.err || { \
	    echo "FAIL: $$f never dispatched the expert-major kernels — vacuous"; \
	    exit 1; }; \
	  cmp -s moe-em-off.out moe-em-on.out || { \
	    echo "FAIL: $$f expert-major output differs from slot-major"; exit 1; }; \
	  echo "  metal moe expert-major ok ($$f, byte-identical)"; \
	done; \
	rm -f moe-em-off.out moe-em-on.out moe-em-on.err
else
	@echo "metal moe expert-major: SKIP (macOS-only backend)"
endif

# Grouped-MMA MoE prefill (RUNNER_METAL_MOE_MM): unlike the expert-major
# matvec twins above, this stages operands in half and reassociates the
# k-sum, so its contract is the tolerance bound, not byte identity — the
# same deal dense RUNNER_METAL_MM already has. The gate runs the full-model
# identity bound with the feature engaged (engagement grep keeps it
# non-vacuous) plus a greedy smoke that must complete coherently.
# The test-gpu-identity leg below pins RUNNER_METAL_MOE_MM=0 inside the tool
# (the grouped path is not byte-identical by design), so it measures the
# matvec path on the MoE fixture; the grouped path answers to the byte
# comparison at fixture scale and to TEST_MOE_MM_AB, including the
# half-staged twins on the fp16-overflow fixture (2026-09-17).
test-metal-moe-mm: runner $(TEST_GPU_ID) $(TEST_MOE_MM_AB) test-moe-fixture.gptoss-mxfp4.gguf
ifeq ($(BUILD_OS),Darwin)
	@set -e; \
	if ! ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
	  echo "metal moe grouped-mma: SKIP (no Metal device)"; exit 0; fi; \
	RUNNER_METAL_MOE_MM=1 ./$(RUNNER_EXE) -m test-moe-fixture.gptoss-mxfp4.gguf \
	  -p "The quick brown fox jumps over the lazy dog again" -n 16 -b 32 \
	  --temp 0 --no-tray > moe-mm.out 2> moe-mm.err; \
	grep -q "grouped-MMA prefill kernels on" moe-mm.err || { \
	  echo "FAIL: grouped-MMA never engaged — vacuous"; exit 1; }; \
	RUNNER_METAL_MOE_MM=0 ./$(RUNNER_EXE) -m test-moe-fixture.gptoss-mxfp4.gguf \
	  -p "The quick brown fox jumps over the lazy dog again" -n 16 -b 32 \
	  --temp 0 --no-tray > moe-mm-ref.out 2>/dev/null; \
	cmp -s moe-mm.out moe-mm-ref.out || { \
	  echo "FAIL: grouped-MMA diverges from the matvec path at fixture scale"; \
	  echo "  (empirically byte-identical there: values are small enough that"; \
	  echo "  the half staging rounds losslessly. A legitimate future change"; \
	  echo "  that alters this prints THIS message — recalibrate consciously,"; \
	  echo "  do not delete the leg: it is what catches a wrong scale or a"; \
	  echo "  misrouted column, which the fixture-scale identity bound cannot.)"; \
	  exit 1; }; \
	RUNNER_METAL_MOE_MM=1 ./$(TEST_GPU_ID) test-moe-fixture.gptoss-mxfp4.gguf || { \
	  echo "FAIL: the MoE fixture breaches the identity bound on the matvec path"; exit 1; }; \
	./$(TEST_MOE_MM_AB) test-moe-fixture.gptoss-mxfp4.gguf || { \
	  echo "FAIL: grouped-MMA fails the house fidelity bar at fixture scale"; exit 1; }; \
	$(PYTHON) scripts/make-test-moe.py test-moe-ovf --act-fp16-overflow > /dev/null; \
	./$(RUNNER_EXE) -m test-moe-ovf.moe1.gguf --quantize test-moe-ovf.moe1-q8 --quant q8_0 > /dev/null 2>&1; \
	mv -f test-moe-ovf.moe1-q8 test-moe-ovf.moe1-q8.gguf 2>/dev/null || true; \
	RUNNER_METAL_MOE_MM=half ./$(TEST_MOE_MM_AB) test-moe-ovf.moe1-q8.gguf half > moe-mmh-ovf.out 2>&1 || { \
	  cat moe-mmh-ovf.out; echo "FAIL: the half-staged expert GEMM fails the house bar on the fp16-overflow fixture"; \
	  echo "  (unscaled staging read <unk> for every token here, 0/24 top-1)"; exit 1; }; \
	grep -q "half-staged" moe-mmh-ovf.out || { \
	  echo "FAIL: the half-staged twins never engaged on the overflow fixture, vacuous"; exit 1; }; \
	echo "  metal moe grouped-mma ok (engaged, fixture-identical, house bar held; half twins finite past fp16 range)"; \
	rm -f moe-mm.out moe-mm-ref.out moe-mm.err moe-mmh-ovf.out test-moe-ovf.*.gguf
else
	@echo "metal moe grouped-mma: SKIP (macOS-only backend)"
endif

# Decode fusion (RUNNER_METAL_FUSE): rope+store, add+norm and MoE
# gate+up+act each fold into one dispatch on the decode path, and every one
# of them claims BYTE IDENTITY against the unfused kernels — so that is the
# gate, across the architecture roster (dense, swa, E-series, gpt-oss MoE
# with biases and swiglu_oai, q8-requantized experts), with the engagement
# grep keeping a fallen-back build from passing on self-agreement.
# test-swa.gguf and test-es.gguf are otherwise generated inside the metal-swa
# and metal-eseries recipes, which leaves standalone `make test-metal-fuse`
# with a prerequisite and no rule. Neither of those recipes runs in `make
# test`, so on a fresh tree test-es.gguf never existed at all and `make test`
# stopped here with "No rule to make target test-es.gguf".
test-swa.gguf: scripts/make-test-model.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-model.py --arch qwen3 --swa 8,2 test-swa.gguf

# The E-series fixture the fuse roster (and the sanitizer fixture list in
# tests/test_hostile_geometry.py) names by that exact path: shared_kv=3,
# ple_dim=16, the richest of the three configs test-metal-eseries sweeps and
# the one that recipe used to leave behind as its last iteration. The eseries
# loop now writes its own scratch file, so this rule is the only producer of
# test-es.gguf and its config cannot silently change under the fuse gate.
test-es.gguf: scripts/make-test-model.py $(GGUF_FIXTURE_LIB)
	$(PYTHON) scripts/make-test-model.py --eseries 3,16 test-es.gguf

# The fixture prerequisites are macOS-only: on Linux the recipe body is a
# skip line, and naming Darwin fixtures as prerequisites there stops the
# whole `make test` with "No rule to make target" instead of skipping.
ifeq ($(BUILD_OS),Darwin)
TEST_METAL_FUSE_DEPS = test-swa.gguf test-es.gguf test-moe-fixture.gptoss-mxfp4.gguf test-moe-fixture.gemma4-moe.gguf test-qk.gguf test-qkw.gguf
else
TEST_METAL_FUSE_DEPS =
endif
test-metal-fuse: runner $(TEST_METAL_FUSE_DEPS)
ifeq ($(BUILD_OS),Darwin)
	@set -e; \
	if ! ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
	  echo "metal decode fusion: SKIP (no Metal device)"; exit 0; fi; \
	prompt="The quick brown fox jumps over the lazy dog"; \
	./$(RUNNER_EXE) -m test-moe-fixture.moe4.gguf \
	  --quantize test-moe-fixture.moe4-q8.gguf --quant q8_0 >/dev/null 2>&1; \
	for f in test.gguf test-swa.gguf test-es.gguf \
	         test-moe-fixture.gptoss-mxfp4.gguf \
	         test-moe-fixture.moe4-q8.gguf \
	         test-moe-fixture.gemma4-moe.gguf; do \
	  RUNNER_METAL_FUSE=0 ./$(RUNNER_EXE) -m $$f -p "$$prompt" -n 24 \
	    --temp 0 --no-tray > fuse-off.out 2>/dev/null; \
	  ./$(RUNNER_EXE) -m $$f -p "$$prompt" -n 24 --temp 0 --no-tray \
	    > fuse-on.out 2> fuse-on.err; \
	  grep -q "decode fusion on" fuse-on.err || { \
	    echo "FAIL: $$f never dispatched a fused kernel — vacuous"; exit 1; }; \
	  cmp -s fuse-off.out fuse-on.out || { \
	    echo "FAIL: $$f fused decode differs from unfused"; exit 1; }; \
	  echo "  metal decode fusion ok ($$f, byte-identical)"; \
	done; \
	RUNNER_METAL_FUSE=0 ./$(RUNNER_EXE) -m test-moe-fixture.moe4-q8.gguf \
	  --score -p "$$prompt $$prompt $$prompt" --no-tray > fuse-sc-off.json 2>/dev/null; \
	./$(RUNNER_EXE) -m test-moe-fixture.moe4-q8.gguf \
	  --score -p "$$prompt $$prompt $$prompt" --no-tray > fuse-sc-on.json 2>/dev/null; \
	cmp -s fuse-sc-off.json fuse-sc-on.json || { \
	  echo "FAIL: fused teacher-forced logprobs differ from unfused"; exit 1; }; \
	echo "  metal decode fusion ok (score logprobs byte-identical)"; \
	./$(RUNNER_EXE) -m test-qk.gguf \
	  --quantize test-qk-q8.gguf --quant q8_0 >/dev/null 2>&1; \
	./$(RUNNER_EXE) -m test-qkw.gguf \
	  --quantize test-qkw-q4k.gguf --quant q4_k >/dev/null 2>&1; \
	./$(RUNNER_EXE) -m test-qkw.gguf \
	  --quantize test-qkw-q6k.gguf --quant q6_k >/dev/null 2>&1; \
	./$(RUNNER_EXE) -m test-moe-fixture.moe4.gguf \
	  --quantize test-moe-fixture.moe4-q4.gguf --quant q4_0 >/dev/null 2>&1; \
	for spec in "test-qk-q8.gguf:(q8_0)" "test-qkw-q4k.gguf:(q4_k)" \
	            "test-qkw-q6k.gguf:(q6_k)" \
	            "test-moe-fixture.moe4-q4.gguf:(q4_0)"; do \
	  f=$${spec%%:*}; want=$${spec##*:}; \
	  RUNNER_METAL_MV=0 RUNNER_METAL_FUSE=0 ./$(RUNNER_EXE) -m $$f -p "$$prompt" -n 24 \
	    --temp 0 --no-tray > fuse-off.out 2>/dev/null; \
	  RUNNER_METAL_MV=0 RUNNER_METAL_FRONT=all ./$(RUNNER_EXE) -m $$f -p "$$prompt" -n 24 \
	    --temp 0 --no-tray > fuse-on.out 2> fuse-on.err; \
	  grep -qF "attention front on $$want" fuse-on.err || { \
	    echo "FAIL: $$f never engaged the $$want front kernel — vacuous"; \
	    exit 1; }; \
	  cmp -s fuse-off.out fuse-on.out || { \
	    echo "FAIL: $$f front-fused decode differs from unfused"; exit 1; }; \
	  echo "  attention front ok ($$f $$want, byte-identical)"; \
	done; \
	./$(RUNNER_EXE) -m test-moe-fixture.moe4-q4.gguf -p "$$prompt" -n 4 \
	  --temp 0 --no-tray 2> fuse-on.err >/dev/null; \
	grep -qF "attention front on (q4_0)" fuse-on.err || { \
	  echo "FAIL: MoE fixture no longer front-admitted by DEFAULT"; exit 1; }; \
	RUNNER_METAL_MV=0 RUNNER_METAL_FUSE=0 ./$(RUNNER_EXE) -m test-qkw-q4k.gguf \
	  --score -p "$$prompt $$prompt $$prompt" --no-tray > fuse-sc-off.json 2>/dev/null; \
	RUNNER_METAL_MV=0 RUNNER_METAL_FRONT=all ./$(RUNNER_EXE) -m test-qkw-q4k.gguf \
	  --score -p "$$prompt $$prompt $$prompt" --no-tray > fuse-sc-on.json 2>/dev/null; \
	cmp -s fuse-sc-off.json fuse-sc-on.json || { \
	  echo "FAIL: front (q4_k+qk-norm) teacher-forced logprobs differ"; exit 1; }; \
	echo "  attention front ok (qk-norm score logprobs byte-identical)"; \
	for spec in "test-moe-fixture.moe4-q4.gguf:" "test-qkw-q6k.gguf:all" \
	            "test-moe-fixture.gptoss-mxfp4.gguf:"; do \
	  f=$${spec%%:*}; fr=$${spec##*:}; \
	  RUNNER_METAL_FUSE=0 ./$(RUNNER_EXE) -m $$f \
	    --score -p "$$prompt $$prompt $$prompt" --no-tray \
	    > fuse-sc-off.json 2>/dev/null; \
	  RUNNER_METAL_FRONT=$$fr ./$(RUNNER_EXE) -m $$f \
	    --score -p "$$prompt $$prompt $$prompt" --no-tray \
	    > fuse-sc-on.json 2>/dev/null; \
	  cmp -s fuse-sc-off.json fuse-sc-on.json || { \
	    echo "FAIL: $$f fused teacher-forced logprobs differ (per-type"\
	         "contraction — see the F6/front fma notes in kernels.metal)"; \
	    exit 1; }; \
	  echo "  fusion score ok ($$f, logprobs byte-identical)"; \
	done; \
	rm -f fuse-off.out fuse-on.out fuse-on.err fuse-sc-off.json fuse-sc-on.json
else
	@echo "metal decode fusion: SKIP (macOS-only backend)"
endif

# Multi-buffer wrap at real-checkpoint size. test-metal-multibuf forces the
# split on sub-GB fixtures; this arm forces multi-GB buffers on a real model,
# because the 128 GB Apple Silicon measurement (2026-09-01) showed no on-disk artifact
# crosses the real ceiling naturally: maxBufferLength there is 80.6 GiB and
# the largest single file is 79.8 GiB. A 16 GiB forced cap on a 17 GB+ model
# is the closest honest exercise: several real multi-GB wraps, cuts on real
# tensor boundaries, byte-identical output demanded. Opt-in by path:
#   make test-metal-bigmodel-multibuf BIGMODEL=models/Llama-3.3-70B-Instruct-Q4_0.gguf
test-metal-bigmodel-multibuf: runner
ifeq ($(BUILD_OS),Darwin)
	@set -e; \
	if [ -z "$(BIGMODEL)" ]; then \
		echo "metal big-model multibuf skipped: set BIGMODEL=<path.gguf>"; \
	elif [ ! -f "$(BIGMODEL)" ]; then \
		echo "metal big-model multibuf skipped: $(BIGMODEL) not found"; \
	elif ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
		./$(RUNNER_EXE) -m "$(BIGMODEL)" -p "$(BIGPROMPT)" -n 32 --temp 0 \
		  --gpu auto > metal-bigmb-one.out 2>/dev/null; \
		RUNNER_METAL_MAX_BUF=17179869184 ./$(RUNNER_EXE) -m "$(BIGMODEL)" \
		  -p "$(BIGPROMPT)" -n 32 --temp 0 --gpu auto \
		  > metal-bigmb-many.out 2> metal-bigmb-many.err; \
		grep -q "Metal backend" metal-bigmb-many.err || { \
		  echo "FAIL: $(BIGMODEL) did not engage Metal under the 16 GiB cap"; exit 1; }; \
		grep -q "weights copied" metal-bigmb-many.err && { \
		  echo "FAIL: $(BIGMODEL) fell back to a copied buffer"; exit 1; }; \
		n=$$(grep -oE "wrapped in [0-9]+" metal-bigmb-many.err | grep -oE "[0-9]+" | head -1); \
		[ -n "$$n" ] && [ "$$n" -ge 2 ] || { \
		  echo "FAIL: $(BIGMODEL) did not split (wrapped in $${n:-1})"; exit 1; }; \
		cmp -s metal-bigmb-one.out metal-bigmb-many.out || { \
		  echo "FAIL: $(BIGMODEL) output differs across a $$n-buffer split"; exit 1; }; \
		echo "metal big-model multibuf ok ($(BIGMODEL), $$n buffers, byte-identical)"; \
		rm -f metal-bigmb-one.out metal-bigmb-many.out metal-bigmb-many.err; \
	else \
		echo "metal big-model multibuf skipped: no Metal device reported by --caps"; \
	fi
else
	@echo "metal big-model multibuf skipped: macOS-only backend"
endif

test-metal-gelu-overflow: runner
ifeq ($(BUILD_OS),Darwin)
	@set -e; \
	if ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
		$(PYTHON) scripts/make-test-model.py --arch gemma3 --act-overflow test-actovf.gguf; \
		./$(RUNNER_EXE) -m test-actovf.gguf -p "hello world" -n 8 --temp 0 --gpu off > metal-actovf-cpu.out 2>/dev/null; \
		env RUNNER_METAL_MM=0 RUNNER_METAL_ATTN_COOP=0 RUNNER_METAL_MV=0 RUNNER_METAL_MOE_MM=0 ./$(RUNNER_EXE) -m test-actovf.gguf -p "hello world" -n 8 --temp 0 --gpu auto > metal-actovf-gpu.out 2> metal-actovf-gpu.err; \
		cmp -s metal-actovf-cpu.out metal-actovf-gpu.out; \
		grep -q "Metal backend" metal-actovf-gpu.err; \
		echo "metal GELU overflow ok"; \
	else \
		echo "metal GELU overflow smoke skipped: no Metal device reported by --caps"; \
	fi
else
	@echo "metal GELU overflow smoke skipped: macOS-only backend"
endif

test-metal-eseries: runner
ifeq ($(BUILD_OS),Darwin)
	@set -e; \
	if ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
		for cfg in 0,16 3,0 3,16; do \
		  $(PYTHON) scripts/make-test-model.py --eseries $$cfg test-es-cfg.gguf; \
		  ./$(RUNNER_EXE) -m test-es-cfg.gguf -p "hello world" -n 8 --temp 0 --gpu off > metal-es-cpu.out 2>/dev/null; \
		  env RUNNER_METAL_MM=0 RUNNER_METAL_ATTN_COOP=0 RUNNER_METAL_MV=0 RUNNER_METAL_MOE_MM=0 ./$(RUNNER_EXE) -m test-es-cfg.gguf -p "hello world" -n 8 --temp 0 --gpu auto > metal-es-gpu.out 2> metal-es-gpu.err; \
		  cmp -s metal-es-cpu.out metal-es-gpu.out || { echo "eseries $$cfg differs"; exit 1; }; \
		  grep -q "Metal backend" metal-es-gpu.err || { echo "eseries $$cfg: Metal never engaged"; exit 1; }; \
		  case $$cfg in *,0) ;; *) grep -q "per-layer embedding table stays on the host" metal-es-gpu.err || { echo "eseries $$cfg: the per-layer embedding table was wrapped"; exit 1; };; esac; \
		done; \
		echo "metal E-series ok"; \
	else \
		echo "metal E-series smoke skipped: no Metal device reported by --caps"; \
	fi
else
	@echo "metal E-series smoke skipped: macOS-only backend"
endif

# Multi-buffer weight wrap, forced. Every model this is developed on fits a
# single MTLBuffer, so the split path would otherwise ship untested:
# RUNNER_METAL_MAX_BUF shrinks the per-buffer ceiling until a fixture has to
# span several wraps. What is being gated is that the split changes NOTHING --
# the wrap is at tensor boundaries, so output must stay byte-identical.
test-metal-multibuf: runner $(TEST_GPU_ID) test.gguf
ifeq ($(BUILD_OS),Darwin)
	@set -e; \
	if ! ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
	  echo "metal multi-buffer: SKIP (no Metal device)"; exit 0; fi; \
	prompt="The city of Lisbon sits on seven hills above the Tagus estuary"; \
	for pair in test.gguf:262144 \
	            $(if $(wildcard test-moe-fixture.moe1.gguf),test-moe-fixture.moe1.gguf:65536) \
	            $(if $(wildcard models/SmolLM2-135M-Instruct-Q8_0.gguf),models/SmolLM2-135M-Instruct-Q8_0.gguf:33554432); do \
	  m=$${pair%%:*}; cap=$${pair##*:}; \
	  ./$(RUNNER_EXE) -m $$m -p "$$prompt" -n 12 -b 8 --temp 0 --gpu auto \
	    > mb-one.out 2>/dev/null; \
	  RUNNER_METAL_MAX_BUF=$$cap ./$(RUNNER_EXE) -m $$m -p "$$prompt" -n 12 -b 8 \
	    --temp 0 --gpu auto > mb-many.out 2> mb-many.err; \
	  grep -q "Metal backend" mb-many.err || { \
	    echo "FAIL: $$m did not engage Metal under a forced buffer cap"; exit 1; }; \
	  grep -q "weights copied" mb-many.err && { \
	    echo "FAIL: $$m fell back to a copied buffer — the split path was not exercised"; exit 1; }; \
	  n=$$(grep -oE "wrapped in [0-9]+" mb-many.err | grep -oE "[0-9]+" | head -1); \
	  [ -n "$$n" ] && [ "$$n" -ge 2 ] || { \
	    echo "FAIL: $$m did not split (wrapped in $${n:-1}) — the gate would pass vacuously"; exit 1; }; \
	  cmp -s mb-one.out mb-many.out || { \
	    echo "FAIL: $$m output differs across a $$n-buffer split"; exit 1; }; \
	  echo "  metal multi-buffer ok ($$m, $$n buffers, byte-identical)"; \
	done; \
	RUNNER_METAL_MAX_BUF=262144 ./$(TEST_GPU_ID) test.gguf; \
	rm -f mb-one.out mb-many.out mb-many.err
else
	@echo "metal multi-buffer: SKIP (macOS-only backend)"
endif

test-metal-swa: runner
ifeq ($(BUILD_OS),Darwin)
	@set -e; \
	if ./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if (d.get('gpu') or {}).get('backend') == 'metal' else 1)"; then \
		$(PYTHON) scripts/make-test-model.py --arch qwen3 --swa 8,2 test-swa.gguf; \
		./$(RUNNER_EXE) -m test-swa.gguf -p "abcdefghijklmnopqrstuvwxyz0123456789" -n 12 --temp 0 --gpu off > metal-swa-cpu.out 2>/dev/null; \
		env RUNNER_METAL_MM=0 RUNNER_METAL_ATTN_COOP=0 RUNNER_METAL_MV=0 RUNNER_METAL_MOE_MM=0 ./$(RUNNER_EXE) -m test-swa.gguf -p "abcdefghijklmnopqrstuvwxyz0123456789" -n 12 --temp 0 --gpu auto > metal-swa-gpu.out 2> metal-swa-gpu.err; \
		cmp -s metal-swa-cpu.out metal-swa-gpu.out; \
		grep -q "Metal backend" metal-swa-gpu.err; \
		echo "metal SWA ok"; \
	else \
		echo "metal SWA smoke skipped: no Metal device reported by --caps"; \
	fi
else
	@echo "metal SWA smoke skipped: macOS-only backend"
endif

# The Metal gates of `make test`, as one named target: every backend check
# that needs an Apple GPU, in the order `make test` runs them. On any other
# platform each self-skips with a printed reason. CI runs this target on the
# hosted macOS runner for every pull request, where the whole suite would
# take half an hour; `make test` runs it on the merge and on a lab Mac.
# The two KV-cache smokes run their model half only where
# models/SmolLM2-135M-Instruct-Q8_0.gguf (or ASAN_MODEL) exists.
#
# Every test-metal-* target is called here and every test-cuda-* target is a
# step of scripts/cuda-gate.sh, except the ones DEVICE_OPT_IN names, each for
# the input it needs that no lane has (scripts/check-ci-roster.py holds this:
# ten Metal smokes went two months with no caller, and three CUDA gates):
#   test-metal-iquants            llama-quantize (RUNNER_LLAMA_CPP_BIN) and its fixtures
#   test-metal-bigmodel           BIGMODEL=, a real multi-GB GGUF
#   test-metal-bigmodel-multibuf  BIGMODEL=, the same
DEVICE_OPT_IN = test-metal-iquants test-metal-bigmodel test-metal-bigmodel-multibuf
.PHONY: test-metal
test-metal:
	@[ -f test-moe-fixture.moe1.gguf ] || $(PYTHON) scripts/make-test-moe.py test-moe-fixture
	$(MAKE) --no-print-directory test-metal-shader-gate
	$(MAKE) --no-print-directory test-metal-fallback PYTHON="$(PYTHON)"
	$(MAKE) --no-print-directory test-metal-kquant
	$(MAKE) --no-print-directory test-metal-decode-only
	$(MAKE) --no-print-directory test-metal-split
	$(MAKE) --no-print-directory test-metal-bind-failure
	$(MAKE) --no-print-directory test-metal-multibuf
	$(MAKE) --no-print-directory test-metal-moe-em
	$(MAKE) --no-print-directory test-metal-moe-mm
	$(MAKE) --no-print-directory test-metal-fuse
	$(MAKE) --no-print-directory test-metal-prefill
	$(MAKE) --no-print-directory test-metal-moe
	$(MAKE) --no-print-directory test-metal-gptoss-moe
	$(MAKE) --no-print-directory test-metal-gemma4-moe
	$(MAKE) --no-print-directory test-metal-gemma4-hetero
	$(MAKE) --no-print-directory test-metal-eseries
	$(MAKE) --no-print-directory test-metal-swa
	$(MAKE) --no-print-directory test-metal-gelu-overflow
	$(MAKE) --no-print-directory test-metal-kv-fp4
	$(MAKE) --no-print-directory test-metal-kv-q8

# `make test` is three targets run one after another: the C gates, the batch
# gates (the microbatch byte-identity family, long and independent), and the
# python target (the client tests and the root pytest). A local run keeps the
# same order and load as the one recipe this used to be; CI runs the three as
# parallel jobs on macOS and Windows (R6.13 step 2), where the root pytest
# also runs under pytest-xdist when PYTEST_WORKERS is set.
test: test-gates
	$(MAKE) --no-print-directory test-python PYTHON="$(PYTHON)"

test-gates:
	$(MAKE) --no-print-directory test-gates-core PYTHON="$(PYTHON)"
	$(MAKE) --no-print-directory test-gates-batch PYTHON="$(PYTHON)"

test-gates-core: test-python-deps $(TEST_JSON_SCHEMA) $(TEST_SVAL_WALK) $(TEST_JSON_OOM) $(TEST_SCHEMA_OOM) $(TEST_SAMPLER) $(TEST_LORA_GRAD) $(TEST_DPO_GRAD) $(TEST_KD_GRAD) $(TEST_MVT) $(TEST_TRAIN_GPU) $(TEST_MVCANON) \
      $(TEST_TOKENIZER) $(TEST_TOK_MERGE) $(TEST_TOKENIZER_OOM) $(TEST_TOK_CLASSES) $(TEST_TEMPLATE) $(TEST_PROMPT_MARKS) \
      $(TEST_TEMPLATE_OOM) $(TEST_TEMPLATE_FAMILIES) \
      $(TEST_TOOLS) $(TEST_SHARED) $(TEST_MLOCK) test-es-mlock.gguf $(TEST_FILE_ID) $(TEST_SPEC_ID) $(TEST_BIND) $(TEST_HOST_HEADER) \
      $(TEST_PREFIX) $(TEST_GRAMMAR_FF) $(TEST_CHOICE_MASS) $(TEST_LOOKUP_DRAFT) $(TEST_VRAMREG) $(TEST_UPLOAD_REPORT) $(TEST_KV_TOL) $(TEST_KV_FP4) $(TEST_CUDA_KVFP4) $(TEST_TC_TOL) $(TEST_I8_TOL) $(TEST_MV_TOL) $(TEST_ATTN_TOL) $(TEST_GPU_ID) $(TEST_MOE_TOL) $(TEST_MOE_ROUTER) $(TEST_PAGING_WARN) $(TEST_AUTOFIT) $(TEST_RESP_SM_DEP) \
      $(TEST_QUANTS_SIMD) $(TEST_IQ_DECODE) $(TEST_KQ_STAGE) $(TEST_INSTANCES) $(TEST_INSTANCES_OOM) $(TEST_METAL_ADMISSION) $(TEST_TRAY_CORE) $(TEST_TRAY_WIN_DEP) $(TEST_UTF8_CONSOLE_DEP) $(RUNNER_NO_MANIFEST) \
      $(TEST_QUANTIZE) $(TEST_QUANTIZE_OOM) test-lora.full.gguf \
      $(TEST_VRAM_ROLLBACK) $(TEST_GGUF_GETTERS) $(TEST_HFHUB) $(TEST_GGUF_SPLIT) $(TEST_PARSE) $(TEST_ENVELOPE) $(TEST_SHA256) $(TEST_ED25519) $(TEST_MLDSA) $(TEST_PMATH) $(TEST_FPCLASS) $(TEST_ACT) $(TEST_ECDSA) $(TEST_WATERMARK) $(TEST_CANON_KERNELS) \
      $(TEST_THREAD_DEFAULT) \
      $(TEST_MODEL_LOAD_FAILURE) $(TEST_MODEL_LOAD_OOM) $(TEST_MOE_HOST_BRIDGE) $(TEST_LORA_CHAINS) $(TEST_MVT_CHAINS) $(TEST_DN_RECURRENCE) $(TEST_ATTN_BW) $(TEST_RESTART) $(TEST_PFX_PERSIST) \
      $(TEST_SCHED_TURN) $(TEST_RESIDENCY) $(TEST_CAPACITY) $(TEST_BUDGET) $(TEST_ATTRIB_DEP) \
      $(TEST_STOP_CONSTRAINT) $(TEST_MSG_OOM_DEP) $(TEST_RECURRENT) $(TEST_PENALTY_WINDOW) $(TEST_ROPE_YARN) $(TEST_ATTN_SCALE) $(TEST_DOT_ROUTE) $(TEST_REQUEST_STOP) \
      runner test.gguf test-q8.gguf test-bf16.gguf test-q4k.gguf test-ornith.gguf test-clef.gguf test-ornith-draft.gguf test-yarn.gguf \
      test-qk.gguf test-gemma3.gguf test-gemma4.gguf test-es.gguf test-muse.gguf test-granite.gguf \
      test-gemma3-62.gguf test-gemma3-26.gguf test-qwen4exp.gguf test-lfm2.gguf test-lfm2-moe.gguf
	./$(TEST_RECURRENT)
	RECURRENT_FIXTURE=test-lfm2.gguf ./$(TEST_RECURRENT)
	RECURRENT_FIXTURE=test-lfm2-moe.gguf ./$(TEST_RECURRENT)
	@# qwen4exp: chunked prefill must equal the single-shot prefill (the PLE
	@# conv history carries across batches; llama.cpp's does not), and a row
	@# scored out of a batch (the per-row hyper-connection head) must equal
	@# the same row scored alone. Found: the conv kernel scratch overran its
	@# region at every batch size, and the per-row head read the stream mean.
	./runner -m test-qwen4exp.gguf -p "hello world one two three four" -n 4 --temp 0 --gpu off -c 128 > test-qwen4exp.a.out 2>/dev/null
	./runner -b 3 -m test-qwen4exp.gguf -p "hello world one two three four" -n 4 --temp 0 --gpu off -c 128 > test-qwen4exp.b.out 2>/dev/null
	@# bytes, via python: the Windows CI shell (MSYS2) has no cmp
	$(PYTHON) -c "import sys; a=open('test-qwen4exp.a.out','rb').read(); b=open('test-qwen4exp.b.out','rb').read(); print('qwen4exp chunked vs single-shot prefill: %s' % ('identical' if a==b else 'DIFFER')); sys.exit(0 if a==b else 1)"
	./runner -b 1 -m test-qwen4exp.gguf --score -p "hello world one two three four five six" --gpu off -c 128 > test-qwen4exp.a.out 2>/dev/null
	RUNNER_SCORE_CHUNKED=1 ./runner -b 3 -m test-qwen4exp.gguf --score -p "hello world one two three four five six" --gpu off -c 128 > test-qwen4exp.b.out 2>/dev/null
	@# bytes: below MV_SMALL_BATCH rows the batched forward IS the solo forward
	@# (the routed experts sum in the router's order on both paths since
	@# 2026-10-07; a speculation verify batch must not move a near-tie)
	$(PYTHON) -c "import json,sys; a=json.load(open('test-qwen4exp.a.out'))['logprobs']; b=json.load(open('test-qwen4exp.b.out'))['logprobs']; d=max(abs(x-y) for x,y in zip(a,b)); print('qwen4exp batched vs solo score: %d positions, max %.2e' % (len(a), d)); sys.exit(0 if len(a)==len(b)==56 and d == 0 else 1)"
	rm -f test-qwen4exp.a.out test-qwen4exp.b.out
	./$(TEST_ATTN_SCALE) test-gemma3-62.gguf test-gemma3-26.gguf
	./$(TEST_DOT_ROUTE) test-q8.gguf
	$(PYTHON) scripts/check-ptx-roster.py src/cuda.c src/kernels_ptx.h
	$(PYTHON) scripts/check-ci-roster.py
	$(PYTHON) scripts/check-fpclass.py
	$(PYTHON) scripts/template-conformance.py --cites-only
	$(PYTHON) scripts/check-agent-docs.py --self-test
	$(PYTHON) scripts/check-agent-docs.py AGENTS.md CLAUDE.md
	./$(TEST_PENALTY_WINDOW) test.gguf
	./$(TEST_REQUEST_STOP)
	./$(TEST_LORA_GRAD)
	./$(TEST_DPO_GRAD)
	./$(TEST_KD_GRAD)
	./$(TEST_KD_GRAD) test-granite.gguf test-lora-granite.full.gguf
	./$(TEST_KD_GRAD) test-gemma4.gguf test-lora-gemma4.full.gguf
	@# and against a QUANTIZED base: the transposed quantized matvec (dx =
	@# W^T dy through frozen Q8_0 rows) is the one genuinely new kernel
	@# family here, so it gets its own gradient gate
	./$(TEST_LORA_GRAD) test-q8.gguf test-lora-q8.full.gguf
	@# and on NoPE layers with the attention temperature live: the recompute
	@# and the backward must match the serving forward (no rope, Q scaled)
	./$(TEST_LORA_GRAD) test-nope.gguf test-lora-nope.full.gguf
	@# and with partial-dimension rope (R8.16): the transpose rotates only
	@# the rope_dim dims, pairs (llama) and split halves (neox)
	./$(TEST_LORA_GRAD) test-prope.gguf test-lora-prope.full.gguf
	./$(TEST_LORA_GRAD) test-prope-neox.gguf test-lora-prope-neox.full.gguf
	@# and through the Gated DeltaNet recurrence (R8.16), qwen35 and clef:
	@# per-slot directions and the exact-zero probe on top ("slots"), since
	@# the whole-adapter direction missed a dropped decay or gate term
	./$(TEST_LORA_GRAD) test-ornith.gguf test-lora-ornith.full.gguf slots
	./$(TEST_LORA_GRAD) test-clef.gguf test-lora-clef.full.gguf slots
	@# R8.9.4: granite's muP scalars (residual and logit scale, fixed
	@# attention scale, tied output) and muse-glimmer's output gate, sandwich
	@# norms, sliding window and softcapped head, all through the adjoint
	./$(TEST_LORA_GRAD) test-granite.gguf test-lora-granite.full.gguf
	./$(TEST_LORA_GRAD) test-muse.gguf test-lora-muse.full.gguf
	@# R8.9.5: gemma3's tanh-GELU gated FFN, the first refusal gemma3/gemma4
	@# hit under --train -- act_f/act_d beside the SiLU pair, gated_act
	@# itself recomputed at the tape site so serving and training agree
	./$(TEST_LORA_GRAD) test-gemma3.gguf test-lora-gemma3.full.gguf
	@# R8.9.6: gemma-4 dense (weightless V norm), absent-V layers, and the
	@# E-series (shared KV + per-layer embeddings) as a stiff fixture: the
	@# whole-adapter directional derivative and the cosine are its checks
	./$(TEST_LORA_GRAD) test-gemma4.gguf test-lora-gemma4.full.gguf
	@# R17.3: an attention-only adapter on a sparse-MoE model, the backward
	@# through the frozen experts and the router (renormalized top-k, and the
	@# plain softmax over all experts), held per coordinate to 0.5%
	./$(TEST_LORA_GRAD) test-moe-train.moe4-train.gguf test-lora-moe4.full.gguf strict
	./$(TEST_LORA_GRAD) test-moe-train.moe4-train-nonorm.gguf test-lora-moe4n.full.gguf strict
	./$(TEST_LORA_GRAD) test-g4v.gguf test-lora-g4v.full.gguf
	./$(TEST_LORA_GRAD) test-es8.gguf test-lora-es8.full.gguf stiff
	@# and with qwen3-style per-head QK norms in the layer: the norm adjoint
	@# sits between the rope adjoint and the projection backward
	./$(TEST_LORA_GRAD) test-qk.gguf test-lora-qk.full.gguf
	@# R8.19: the same gates in --train-gpu's device mode (every weight
	@# product on the GPU, batch-invariant, not the CPU's bits): its own
	@# central differences hold its gradient only if the taped forward and
	@# the recompute see the same values. Forward rounds (and the head's rows
	@# with them) of 5 against the recompute's 14-position window, so a
	@# product whose columns depend on the batch shows. Skips without a CUDA
	@# device.
	RUNNER_TEST_TRAIN_DEV=1 RUNNER_TRAIN_ROWS=5 ./$(TEST_LORA_GRAD)
	RUNNER_TEST_TRAIN_DEV=1 RUNNER_TRAIN_ROWS=5 ./$(TEST_LORA_GRAD) test-q8.gguf test-lora-q8.full.gguf
	RUNNER_TEST_TRAIN_DEV=1 RUNNER_TRAIN_ROWS=5 ./$(TEST_LORA_GRAD) test-qk.gguf test-lora-qk.full.gguf
	RUNNER_TEST_TRAIN_DEV=1 RUNNER_TRAIN_ROWS=5 ./$(TEST_LORA_GRAD) test-nope.gguf test-lora-nope.full.gguf
	RUNNER_TEST_TRAIN_DEV=1 RUNNER_TRAIN_ROWS=5 ./$(TEST_LORA_GRAD) test-prope.gguf test-lora-prope.full.gguf
	RUNNER_TEST_TRAIN_DEV=1 RUNNER_TRAIN_ROWS=5 ./$(TEST_LORA_GRAD) test-ornith.gguf test-lora-ornith.full.gguf slots
	RUNNER_TEST_TRAIN_DEV=1 RUNNER_TRAIN_ROWS=5 ./$(TEST_LORA_GRAD) test-clef.gguf test-lora-clef.full.gguf slots
	RUNNER_TEST_TRAIN_DEV=1 RUNNER_TRAIN_ROWS=5 ./$(TEST_LORA_GRAD) test-granite.gguf test-lora-granite.full.gguf
	RUNNER_TEST_TRAIN_DEV=1 RUNNER_TRAIN_ROWS=5 ./$(TEST_LORA_GRAD) test-muse.gguf test-lora-muse.full.gguf
	RUNNER_TEST_TRAIN_DEV=1 RUNNER_TRAIN_ROWS=5 ./$(TEST_LORA_GRAD) test-gemma3.gguf test-lora-gemma3.full.gguf
	RUNNER_TEST_TRAIN_DEV=1 RUNNER_TRAIN_ROWS=5 ./$(TEST_LORA_GRAD) test-gemma4.gguf test-lora-gemma4.full.gguf
	RUNNER_TEST_TRAIN_DEV=1 RUNNER_TRAIN_ROWS=5 ./$(TEST_LORA_GRAD) test-moe-train.moe4-train.gguf test-lora-moe4.full.gguf strict
	RUNNER_TEST_TRAIN_DEV=1 RUNNER_TRAIN_ROWS=5 ./$(TEST_LORA_GRAD) test-g4v.gguf test-lora-g4v.full.gguf
	RUNNER_TEST_TRAIN_DEV=1 RUNNER_TRAIN_ROWS=5 ./$(TEST_LORA_GRAD) test-es8.gguf test-lora-es8.full.gguf stiff
	./$(TEST_MVT)
	./$(TEST_MVT) test-q8.gguf
	./$(TEST_MVT) test-bf16.gguf
	./$(TEST_TRAIN_GPU)
	./$(TEST_MVCANON)
	./$(TEST_MVCANON) test-q8.gguf
	./$(TEST_BIND)
	./$(TEST_HOST_HEADER)
	./$(TEST_RESIDENCY)
	./$(TEST_CAPACITY)
	./$(TEST_RESTART)
	./$(TEST_PFX_PERSIST)
	./$(TEST_SCHED_TURN)
	./$(TEST_VRAMREG)
	./$(TEST_UPLOAD_REPORT)
	./$(TEST_JSON_SCHEMA)
	./$(TEST_SVAL_WALK)
	./$(TEST_JSON_OOM)
	./$(TEST_SCHEMA_OOM)
	./$(TEST_SAMPLER)
	./$(TEST_TOKENIZER)
	./$(TEST_TOK_MERGE)
	./$(TEST_TOKENIZER_OOM)
	./$(TEST_TOK_CLASSES)
	./$(TEST_TEMPLATE)
	./$(TEST_PROMPT_MARKS) test.gguf
	./$(TEST_TEMPLATE_OOM)
	./$(TEST_TEMPLATE_FAMILIES)
	./$(TEST_TOOLS)
	./$(TEST_BUDGET)
	$(TEST_ATTRIB_RUN)
	$(TEST_MSG_OOM_RUN)
	./$(TEST_SHARED)
	./$(TEST_MLOCK) test-es-mlock.gguf
	./$(TEST_FILE_ID)
	./$(TEST_PREFIX)
	./$(TEST_GRAMMAR_FF)
	./$(TEST_GRAMMAR_FF) test-ornith.gguf
	./$(TEST_CHOICE_MASS)
	./$(TEST_LOOKUP_DRAFT)
	./$(TEST_STOP_CONSTRAINT)
	$(TEST_RESP_SM_RUN)
	./$(TEST_KV_TOL)
	./$(TEST_KV_FP4)
	./$(TEST_CUDA_KVFP4)
	./$(TEST_TC_TOL)
	$(MAKE) --no-print-directory test-tc-overflow
	./$(TEST_I8_TOL)
	./$(TEST_MV_TOL)
	./$(TEST_ATTN_TOL)
	./$(TEST_GPU_ID) --self-test
	./$(TEST_GPU_ID)
	./$(TEST_QUANTS_SIMD)
	./$(TEST_IQ_DECODE)
	./$(TEST_KQ_STAGE)
	./$(TEST_INSTANCES)
	./$(TEST_INSTANCES_OOM)
	./$(TEST_METAL_ADMISSION)
	./$(TEST_TRAY_CORE)
	$(TEST_TRAY_WIN_RUN)
	$(TEST_UTF8_CONSOLE_RUN)
	@# the CPU-only backend stub, which no platform branch ever builds
	$(MAKE) --no-print-directory test-gpu-stub
	@# The split guard was absent from this list entirely, which is how a
	@# target-name collision kept it unbuilt and unnoticed. It self-skips
	@# without CUDA, so it costs a Mac nothing and actually fires on the boxes
	@# that have the backend it guards.
	$(MAKE) --no-print-directory test-split-guard
	$(MAKE) --no-print-directory test-makefile-sane
	@# the fused-vs-eager routing gate needs a fixture whose router is not
	@# zero: the dense-oracle MoE fixtures are 0.5/0.5 either way and can only
	@# compare a routing path with itself (it self-skips on those, correctly)
	$(PYTHON) scripts/make-test-moe.py test-moe-fixture
	./$(TEST_ROPE_YARN) test-moe-fixture.gptoss-yarn.gguf test-yarn.gguf
	./$(TEST_MOE_TOL) test-moe-fixture.moe4.gguf
	@# CPU/GPU agreement on a router that has a BIAS and routes top-1, so the
	@# bias decides which expert runs. Every other MoE fixture here either has
	@# no router bias or selects all its experts, and both hide a dropped bias
	@# under the tolerances -- which is exactly how the CUDA fused path shipped
	@# without one. See the fixture's comment in scripts/make-test-moe.py.
	./$(TEST_GPU_ID) test-moe-fixture.gptoss-top1.gguf
	./$(TEST_MOE_ROUTER) test-moe-fixture
	./$(TEST_PAGING_WARN) test-moe-fixture
	./$(TEST_AUTOFIT) test.gguf
	@# Llama-4 attention knobs: NoPE and the position-dependent temperature
	@mkdir -p test-attn
	$(PYTHON) scripts/make-test-model.py test-attn/k_off.gguf
	$(PYTHON) scripts/make-test-model.py --attn-knobs 1,0.0 test-attn/k_nope.gguf
	$(PYTHON) scripts/make-test-model.py --attn-knobs 2,0.0 test-attn/k_half.gguf
	$(PYTHON) scripts/make-test-model.py --attn-knobs 1,0.1 test-attn/k_temp.gguf
	@# floor_scale 4 instead of llama-4's 8192: at 8192 the temperature is
	@# exactly 1.0 for every position a test prompt reaches, so the knob's
	@# arithmetic is never actually exercised. This fixture makes it live.
	$(PYTHON) scripts/make-test-model.py --attn-knobs 1,5.0,4 test-attn/k_temp_live.gguf
	$(PYTHON) -m pytest -q tests/test_attn_knobs.py
	@# CPU/GPU agreement at logit precision on each knob. Token identity (the
	@# text gates) is blind at fixture scale -- see the test's header.
	./$(TEST_GPU_ID) test-attn/k_off.gguf
	./$(TEST_GPU_ID) test-attn/k_nope.gguf
	./$(TEST_GPU_ID) test-attn/k_half.gguf
	./$(TEST_GPU_ID) test-attn/k_temp_live.gguf
	@# The same comparison with every forward a single token. That is a
	@# backend STATE, not a slower schedule: scratch sized lazily on the first
	@# multi-token batch stays nil for such a run, and the text gates above are
	@# blind to what that does at fixture scale.
	./$(TEST_GPU_ID) test.gguf 0 1
	@# spark2_5 on the device (2026-10-30): Q, K and V as row slices of one
	@# fused tensor, the per-head output gate, the exact-erf GELU and two rope
	@# widths and bases, against the CPU at logit precision, batched and one
	@# token at a time, and with every layer sliding
	$(PYTHON) scripts/make-test-model.py --spark2-5 test-spark.gguf > /dev/null
	$(PYTHON) scripts/make-test-model.py --spark2-5 --spark-all-swa test-spark-swa.gguf > /dev/null
	./$(TEST_GPU_ID) test-spark.gguf
	./$(TEST_GPU_ID) test-spark.gguf 0 1
	./$(TEST_GPU_ID) test-spark-swa.gguf
	./$(TEST_QUANTIZE)
	./$(TEST_VRAM_ROLLBACK)
	./$(TEST_GGUF_GETTERS)
	./$(TEST_HFHUB)
	@mkdir -p test-gguf-split
	$(PYTHON) scripts/make-test-model.py test-gguf-split/whole.gguf
	$(PYTHON) scripts/gguf-split.py test-gguf-split/whole.gguf test-gguf-split/part 3
	./$(TEST_GGUF_SPLIT) test-gguf-split/whole.gguf test-gguf-split/part-00001-of-00003.gguf
	./$(TEST_PARSE)
	./$(TEST_ENVELOPE)
	./$(TEST_SHA256)
	./$(TEST_ED25519)
	./$(TEST_MLDSA)
	./$(TEST_CANON_KERNELS)
	./$(TEST_PMATH)
	./$(TEST_FPCLASS)
	./$(TEST_ACT)
	./$(TEST_ECDSA)
	./$(TEST_WATERMARK)
	./$(TEST_THREAD_DEFAULT)
	./$(TEST_MODEL_LOAD_FAILURE)
	@# a model load with each of its allocations failed in turn, alone and
	@# beside an instance holding the shared weights, over one fixture per
	@# binding shape: dense, qk-norm, sliding, Gemma 4 (plain, mixed heads,
	@# E-series), Muse, Granite, the recurrent hybrids, PLE/MTP, three MoE
	@# routings, a trunk MTP head, a removed sublayer, per-layer FFN widths
	$(PYTHON) scripts/make-test-model.py --gemma4-hetero test-load-oom-g4h.gguf > /dev/null
	$(PYTHON) scripts/make-test-model.py --mtp-layers 1 test-load-oom-mtp.gguf > /dev/null
	$(PYTHON) scripts/make-test-model.py --ffn-widths 128,96 test-load-oom-ffw.gguf > /dev/null
	$(PYTHON) scripts/make-test-moe.py test-load-oom-moe > /dev/null
	$(PYTHON) scripts/make-test-hybrid.py test-load-oom-hybrid > /dev/null
	$(PYTHON) scripts/make-test-nemotron.py test-load-oom-nemotron > /dev/null
	./runner -m test.gguf --quantize test-load-oom-cut.gguf --remove-sublayer attn:1 > /dev/null 2>&1
	./$(TEST_MODEL_LOAD_OOM) test.gguf test-qk.gguf test-gemma3.gguf test-gemma4.gguf \
	    test-load-oom-g4h.gguf test-es.gguf test-muse.gguf test-granite.gguf \
	    test-ornith.gguf test-clef.gguf test-qwen4exp.gguf test-load-oom-moe.gptoss-mxfp4.gguf \
	    test-load-oom-moe.shexpg.gguf test-load-oom-moe.gemma4-moe.gguf \
	    test-load-oom-hybrid.gguf test-load-oom-nemotron.gguf test-load-oom-mtp.gguf \
	    test-load-oom-cut.gguf test-load-oom-ffw.gguf 2> test-model-load-oom.err || \
	    { tail -40 test-model-load-oom.err; exit 1; }
	@tail -1 test-model-load-oom.err
	@# every allocation of six file transformations (requant, type plan,
	@# sublayer removal, expert prune, merge, context surgery) failed in
	@# turn, and three refusals between their stages
	./$(TEST_QUANTIZE_OOM) . test.gguf test-load-oom-moe.moe4.gguf test-yarn.gguf \
	    test-lora.full.gguf 2> test-quantize-oom.err || \
	    { tail -40 test-quantize-oom.err; exit 1; }
	@tail -1 test-quantize-oom.err
	@# the host MoE bridge equals the forward's FFN on every MoE layer: the
	@# routings above, Granite's residual scale, Nemotron-H's gate-less
	@# shared expert
	$(PYTHON) scripts/make-test-hybrid.py test-bridge-nemo --arch nemotron_h_moe > /dev/null
	./$(TEST_MOE_HOST_BRIDGE) test-load-oom-moe.gptoss-mxfp4.gguf test-load-oom-moe.shexpg.gguf \
	    test-load-oom-moe.shexp.gguf test-load-oom-moe.gemma4-moe.gguf test-load-oom-moe.moe4.gguf \
	    test-load-oom-moe.afmoe-plain.gguf test-load-oom-moe.ggroup.gguf test-load-oom-moe.rsigmoid.gguf \
	    test-load-oom-hybrid.gguf test-bridge-nemo.gguf 2> test-moe-host-bridge.err || \
	    { tail -20 test-moe-host-bridge.err; exit 1; }
	@tail -1 test-moe-host-bridge.err
	./$(TEST_LORA_CHAINS)
	./$(TEST_MVT_CHAINS)
	./$(TEST_DN_RECURRENCE)
	./$(TEST_ATTN_BW)
	$(MAKE) --no-print-directory test-bare-invocation
	$(MAKE) --no-print-directory test-help-interface
	$(MAKE) --no-print-directory test-shader-embed
	$(MAKE) --no-print-directory test-metal PYTHON="$(PYTHON)"
	@for f in test.gguf test-q8.gguf test-bf16.gguf test-q4k.gguf; do \
	  ./$(TEST_SPEC_ID) $$f || exit 1; ./$(TEST_SPEC_ID) $$f cpu || exit 1; done
	$(PYTHON) scripts/check-generated.py
	$(MAKE) --no-print-directory test-moe PYTHON="$(PYTHON)"
	$(MAKE) --no-print-directory test-prune-experts PYTHON="$(PYTHON)"

# The microbatch gates: byte identity of the batched forward against the
# sequential one on the F32, Q8_0 and BF16 fixtures (plus the two declines),
# the Metal microbatch identity on two fixtures, and the attention split.
# They are the longest gates and share nothing with the rest, so CI runs them
# as their own job.
test-gates-batch: runner $(TEST_BATCH) $(TEST_BATCH_ID) $(TEST_ATTN_SPLIT) test.gguf test-q8.gguf test-bf16.gguf test-q4k.gguf test-muse.gguf
	./$(TEST_BATCH)
	@# and again on a QUANTIZED fixture: the F32 run above cannot reach a
	@# quantized matvec at all, so on its own it gates nothing real
	./$(TEST_BATCH) test-q8.gguf
	@# BF16 has separate width-classed CUDA twins; cover both x4 and x8.
	./$(TEST_BATCH) test-bf16.gguf
	@# A rope-enabled dense Mamba-2 hybrid must decline the CUDA microbatch:
	@# fwd_batch has no recurrent/skip-layer path and would treat its NULL
	@# attention projections as dense weights. The sequential fallback remains
	@# the byte-identity reference until a real recurrent batch loop exists.
	$(PYTHON) scripts/make-test-hybrid.py test-hybrid-batch --dense --rope
	./$(TEST_BATCH) test-hybrid-batch.gguf 2
	@# Apertus is dense but ungated: fwd_batch's gated FFN would pass its NULL
	@# w_gate to enc_mv_batch. It must use the sequential GPU fallback until
	@# that loop grows an explicit xIELU branch.
	$(PYTHON) scripts/make-test-model.py --apertus REAL test-apertus-batch.gguf
	./$(TEST_BATCH) test-apertus-batch.gguf 2
	./$(TEST_BATCH_ID) test.gguf
	./$(TEST_BATCH_ID) test-q4k.gguf
	$(MAKE) --no-print-directory test-attn-split

# The python half: the client tests and the root pytest. Its prerequisites
# are the fixtures and gate binaries the root tests read by name: they skip,
# not fail, without them (a local -n 3 run without test-instances and
# test-tc-tol lost six cases to skips), so they are listed here instead of
# being left to the old recipe's order.
# PYTEST_WORKERS=N (or auto) runs the root pytest under pytest-xdist, whole
# files per worker: every module's server fixture starts once per file.
# The gate binaries the root tests drive or skip without: test-instances
# (lease interop), test-tc-tol (TC eligibility, iquants), test-gpu-identity
# (iquants), test-lora-grad (DeltaNet autograd), the manifest-less runner
# (Windows console tests).
test-python: test-python-deps runner $(DIFFTOK) $(TEST_INSTANCES) $(TEST_TC_TOL) $(TEST_GPU_ID) $(TEST_LORA_GRAD) $(RUNNER_NO_MANIFEST) \
             test.gguf test-lora.full.gguf test-moe-train.moe4-train.gguf test-es.gguf
	@[ -f test-moe-fixture.moe4.gguf ] || $(PYTHON) scripts/make-test-moe.py test-moe-fixture > /dev/null
	PYTHONPATH=python/src $(PYTHON) -m pytest python/tests/
	$(PYTHON) scripts/ci-run.py --timeout 2400 root-pytest -- $(PYTHON) -m pytest $(PYTEST_XDIST) --durations=25 -ra --junitxml=.build/ci/root.xml -q tests/test_ci_contracts.py tests/test_cuda_act_probe.py tests/test_fit_check.py tests/test_apertus.py tests/test_ornith_cpu.py tests/test_ornith_reference.py tests/test_compat_matrix.py tests/test_arch_admission.py tests/test_hybrid_admission.py tests/test_hostile_geometry.py tests/test_certify_envelope.py tests/test_cpu_cuda_margin.py tests/test_envelope_gate.py tests/test_envelope_swap.py tests/test_cli_files.py tests/test_hf_fetch.py tests/test_pull.py tests/test_stop_specials.py tests/test_muse_auto_toolchoice.py tests/test_chat_template_flag.py tests/test_server_banner.py tests/test_split_gguf.py tests/test_metal_coverage.py tests/test_metal_mv_default.py tests/test_gpu_declines.py tests/test_caps.py tests/test_tool_info.py tests/test_bench_json.py tests/test_mtp_admission.py tests/test_mtp_consume.py tests/test_compare_llamacpp.py tests/test_release_check.py tests/test_glibc_release.py tests/test_eseries.py tests/test_stress_models.py tests/test_moe_prune_plan.py tests/test_kld_compare.py tests/test_kld_margin.py tests/test_quant_fidelity.py tests/test_token_divergence.py tests/test_verify_gguf.py tests/test_type_plan_size.py tests/test_stress_context.py tests/test_cert_greedy_identity.py tests/test_tokenizer_corpus.py tests/test_batch_bench.py tests/test_spec_telemetry.py tests/test_draft_required.py tests/test_draft_lookup.py tests/test_kv_reachable.py tests/test_kv_ring.py tests/test_tiedv.py tests/test_moe_mm_flips.py tests/test_load_prefetch.py tests/test_spec_gpu.py tests/test_request_disconnect.py tests/test_score.py tests/test_decide.py tests/test_clef.py tests/test_lfm2.py tests/test_spark2_5.py tests/test_flint.py tests/test_k2_horizon.py tests/test_yarn_betas.py tests/test_hf_spec_bounds.py tests/test_doc_counts.py tests/test_lora.py tests/test_lora_qwen35.py tests/test_train.py tests/test_train_teacher.py tests/test_train_recipe.py tests/test_deltanet_autograd.py tests/test_rollout_identity.py tests/test_merge.py tests/test_lineage.py tests/test_eval_record.py tests/test_moe_lora.py tests/test_transcript.py tests/test_oms.py tests/test_kv_quality.py tests/test_tool_choice_boundary.py tests/test_nvfp4_scale.py tests/test_remove_sublayer.py tests/test_rewind_under_refused_prefix.py tests/test_server_penalty_exemptions.py tests/test_lora_identity_alpha.py tests/test_ttl_releases_draft.py tests/test_depth_slice.py tests/test_device_evidence.py tests/test_model_tiers.py tests/test_site_tiers.py tests/test_tier_smoke.py tests/test_difftok.py tests/test_unicode_classes.py tests/test_gemma4_untyped_fallback.py tests/test_xml_schema_fallback.py tests/test_gen_quality_metrics.py tests/test_granite.py tests/test_phi3_rope.py tests/test_iquants.py tests/test_metal_moe_batch.py tests/test_muse_glimmer.py tests/test_receipts.py tests/test_truncation_benchmark.py tests/test_type_plan.py tests/test_unload_honesty.py tests/test_tray_not_raised_on_refusal.py tests/test_tray_not_raised_without_console.py tests/test_shadow_mode_flag.py tests/test_record_sign.py tests/test_qwen3_coder_tools.py tests/test_native_xml_routing.py tests/test_sampling_defaults.py tests/test_coverage_inventory.py tests/test_turn_mark.py tests/test_cuda_iq_grids.py tests/test_metal_iq_kernels.py tests/test_tc_gate_eligibility.py tests/test_decide_calibrate.py tests/test_gpu_split_order.py tests/test_cuda_tensor_roles.py tests/test_cuda_buffer_lifetime.py tests/test_cuda_upload_report.py tests/test_tooluse_shifted.py tests/test_reasoning_budget.py tests/test_reasoning_sampling.py tests/test_loop_guard.py tests/test_model_id_long_name.py tests/test_gguf_blockorder.py tests/test_schema_close_api.py tests/test_embeddings_pooling.py tests/test_provenance.py tests/test_rerank.py tests/test_echo_logprobs.py tests/test_top_logprobs_keys.py tests/test_repeated_tool_calls.py tests/test_named_contexts.py tests/test_responses_store.py tests/test_adapter_routing.py tests/test_bundle.py tests/test_serve_receipts.py tests/test_adapter_signature.py tests/test_sign_model.py tests/test_agent_transcripts.py tests/test_watermark.py tests/test_kv_snapshots.py tests/test_session_images.py tests/test_parent_pid.py tests/test_lease_interop.py tests/test_gold_logits.py tests/test_mv_fuse.py tests/test_stream_utf8.py tests/test_tool_schema_validity.py tests/test_doctor.py tests/test_metal_attn_tile.py tests/test_metal_attn_gqa.py tests/test_server_sessions.py tests/test_confirm_below.py tests/test_evidence_pack.py tests/test_closure_provenance.py tests/test_reasoning_reserve.py tests/test_hermes_json_families.py tests/test_adapt_info.py tests/test_expect_resident.py tests/test_health_requests.py tests/test_server_shared_pool.py tests/test_completion_token_ids.py tests/test_cli_smokes.py tests/test_utf8_io.py tests/test_cli_lifecycle.py tests/test_cli_surface.py tests/test_completion_sequences.py tests/test_reasoning_start.py tests/test_generation_fault_framing.py tests/test_stop_latency.py tests/test_agent_torture.py tests/test_torture_compare.py tests/test_competitor_freshness.py tests/test_write_stall_experiment.py tests/test_idle_coexistence.py tests/test_family_fixtures.py tests/test_bench_corpus.py

# --------------------------------------------------------------------------
# Template conformance: runner's native C chat renderer vs the model's OWN
# jinja chat template, byte for byte.
#
# Runner renders chat templates in C rather than executing jinja. That is
# deliberate, but it makes silent drift the DEFAULT failure mode, and
# tests/test_template.c cannot catch it: its goldens were written from
# runner's own output, so they prove self-consistency and nothing else. This
# target is the missing oracle.
#
# NOT part of `make test`, on purpose. Three reasons, in order of weight:
#
#   1. It needs an oracle that is not in the tree. The HF-backed families are
#      fetched over the network; the rest are read out of multi-GB GGUFs under
#      models/. `make test` is the offline gate, and a gate that goes red
#      because huggingface.co blinked is a gate people learn to ignore --
#      which is exactly how the goldens rotted in the first place.
#   2. The cache cannot rescue it either: the oracles live under .build/, and
#      `make clean` wipes .build. `make clean && make test` would therefore
#      always need the network back.
#   3. It reports THREE outcomes, not two (exit 2 = "not checked"), because an
#      unavailable oracle must never be counted as conformance. `make test` is
#      binary. Folding a three-state gate into it loses the state that matters.
#
# So it gets its own target, and CI runs it as its own job (.github/workflows/
# ci.yml, job `template-conformance`) restricted to the network-backed
# families, where a red is unambiguous.
#
#   make template-conformance           run the gate (fetches missing oracles)
#   make template-conformance-refresh   re-fetch every oracle first
#   make template-conformance-baseline  re-record the known-difference backlog
#
# The backlog (scripts/template-conformance-baseline.json) holds the
# differences that are BUGS. The gate fails on NEW drift; the backlog keeps
# the existing ones counted and visible instead of quietly allowlisted.
# Intentional deviations go in scripts/template-conformance-allowlist.json,
# and only with a source citation the harness re-verifies on every run.
TEMPLATE_CONFORMANCE = scripts/template-conformance.py

template-conformance:
	$(PYTHON) $(TEMPLATE_CONFORMANCE)

template-conformance-refresh:
	$(PYTHON) $(TEMPLATE_CONFORMANCE) --refresh

template-conformance-baseline:
	$(PYTHON) $(TEMPLATE_CONFORMANCE) --write-baseline

# Harmony's oracle is the openai-harmony LIBRARY, not the jinja in the GGUF:
# that jinja is a reimplementation of the spec with gaps it structurally
# cannot close (no content_type field, so it can never emit <|constrain|>).
# This target records the library's renders so a machine without it still
# checks harmony instead of skipping it. Needs `pip install openai-harmony`,
# or RUNNER_HARMONY_PYTHON pointing at an interpreter that has it.
template-conformance-harmony-oracle:
	$(PYTHON) $(TEMPLATE_CONFORMANCE) --write-harmony-oracle

smoke: runner test.gguf
	./$(RUNNER_EXE) --version
	./$(RUNNER_EXE) --caps
	./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; p=json.load(sys.stdin)['sampling_presets']; assert {x['name'] for x in p} >= {'generic','qwen3','llama3','gemma3','phi3'} and all(x['source'] for x in p); print('preset table ok')"
	./$(RUNNER_EXE) -m test.gguf -p "hello" -n 8 --temp 0 --gpu off
	./$(RUNNER_EXE) -m test.gguf -p "hi" -n 24 --temp 0 --json --gpu off 2>/dev/null | $(PYTHON) -c "import json,sys; json.load(sys.stdin); print('valid json')"
	./$(RUNNER_EXE) --caps | $(PYTHON) -c "import json,sys; c=json.load(sys.stdin); assert c['kv_types'] == ['f16','q8','k8v4','fp4'], c['kv_types']; assert c['kv_type_default'] == 'f16', 'q8 KV is lossy: f16 must stay the default'; print('kv cache types ok')"
	./$(RUNNER_EXE) -m test.gguf -p "hello" -n 8 --temp 0 --gpu off --kv q8 2>&1 | grep -q "head_dim not a multiple of 32" && echo "kv q8 fallback ok"

release-check: runner
	@set -e; \
	tag="$${TAG:-v$$(./$(RUNNER_EXE) --version | sed 's/^runner //')}"; \
	tmp="$$(mktemp)"; \
	trap 'rm -f "$$tmp"' EXIT; \
	printf '%s\n' "$$(./$(RUNNER_EXE) --version)" > "$$tmp"; \
	printf 'tag:        %s\ncommit:     %s\nbuilt:      local\n' "$$tag" "$$(git rev-parse HEAD 2>/dev/null || echo unknown)" >> "$$tmp"; \
	$(PYTHON) scripts/check-release.py --tag "$$tag" --binary ./$(RUNNER_EXE) \
		--build-info "$$tmp" --commit "$$(git rev-parse HEAD 2>/dev/null || echo unknown)"

# Truncation-recovery regression gate. Runner's headline is that a schema-
# constrained tool call stays a parseable tool_calls entry when the token
# budget runs out mid-object (finish_reason "length"), and completes at the
# 64-token control. This spawns Runner on the CPU fixture, drives the pinned
# token ladder, and fails red if any rung loses the property. The property is
# an engine guarantee (grammar + closer), not model quality, so the tiny
# fixture exhibits it with no GPU or competitor. See docs/truncation-benchmark.md
# for the full head-to-head recipe and the committed raw results.
test-truncation: runner test.gguf
	$(PYTHON) scripts/truncation-benchmark.py --model test.gguf --assert

# Real models through every surface (scripts/family-sweep.py). Not part of
# `make test`: it needs a shelf of real GGUFs and, for the gpu rows, a device.
# The roster is the set of families whose behaviour a fixture cannot model:
# thinking blocks, native tool protocols, constrained turns (R4.12.27; every
# defect of 2026-10-02 passed every fixture test). Exit 0 clean, 1 an engine
# FAIL, 2 a model missing from the shelf.
#   make sweep-thinking RUNNER_SWEEP_MODELS=/path/to/models SWEEP_BACKENDS=cpu,gpu
SWEEP_BACKENDS ?= cpu
SWEEP_OUT      ?= family-sweep-out
SWEEP_THREADS  ?= 0
.PHONY: sweep-thinking
sweep-thinking: runner
	@test -n "$(RUNNER_SWEEP_MODELS)" || { echo "error: set RUNNER_SWEEP_MODELS to the model shelf"; exit 2; }
	RUNNER_SWEEP_MODELS="$(RUNNER_SWEEP_MODELS)" $(PYTHON) scripts/family-sweep.py \
		--roster scripts/sweep-rosters/thinking.json --only $(SWEEP_BACKENDS) \
		--threads $(SWEEP_THREADS) --out-dir $(SWEEP_OUT)

# Optional ecosystem gate. Install the pinned Python and Node dependencies in
# tests/compatibility first; Runner itself remains dependency-free.
compat-consumers: runner test.gguf
	$(PYTHON) scripts/consumer_compat.py

# ---------------------------------------------------------------- fuzzing
#
# libFuzzer harnesses for the hand-written parsers that eat untrusted input.
# clang-only: `make fuzz` prints a notice and succeeds when clang is absent
# (the Windows dev box is msys2/gcc), so it never breaks a normal build.
#
# Runs are deliberately short and memory-capped so CI can afford them. Seeds
# are the committed corpora under tests/fuzz/corpus/<target>/; libFuzzer's own
# discoveries and any crash artifacts go to the throwaway fuzz-corpus/ tree
# rather than dirtying the checkout.
FUZZ_CLANG   ?= clang
FUZZ_TIME    ?= 20
FUZZ_RSS_MB  ?= 2048
FUZZ_TARGETS = json_parse schema_compile sval_feed sval_trial jsonv_feed \
               gguf_open http_request tokenizer model_load gguf_split
# -O1 -g: libFuzzer wants speed but ASan reports want frames.
# No -march=native and no -ffast-math: the point here is defined behaviour,
# and UBSan must abort rather than warn or the run cannot gate anything.
FUZZ_INSTRUMENTATION = -fsanitize=fuzzer,address,undefined
FUZZ_FLAGS = -g -O1 -std=gnu11 -Wall -Wextra -Wno-unused-parameter -I src \
             -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=undefined \
             -fno-omit-frame-pointer

FUZZ_SRC_json_parse     = src/json.c
FUZZ_SRC_schema_compile = src/json.c $(SCHEMA_SRC) src/jsonmode.c
FUZZ_SRC_sval_feed      = src/json.c $(SCHEMA_SRC) src/jsonmode.c
FUZZ_SRC_sval_trial     = src/json.c $(SCHEMA_SRC) src/jsonmode.c
FUZZ_SRC_jsonv_feed     = src/jsonmode.c
FUZZ_SRC_gguf_open      = src/gguf.c src/compat.c src/quants.c
FUZZ_SRC_http_request   = src/http.c src/json.c src/compat.c
FUZZ_SRC_tokenizer      = src/tokenizer.c src/gguf.c src/compat.c src/quants.c
FUZZ_SRC_model_load     = src/model.c src/gguf.c src/compat.c src/quants.c src/vramreg.c src/gpu_none.c
FUZZ_SRC_gguf_split     = src/gguf.c src/compat.c src/quants.c

fuzz-%: tests/fuzz/fuzz_%.c tests/fuzz/fuzz_files.h $(wildcard src/*.c) $(HDR)
	$(FUZZ_CLANG) $(FUZZ_FLAGS) $(FUZZ_EXTRA_FLAGS) tests/fuzz/fuzz_$*.c $(FUZZ_SRC_$*) -o $@ -lm -lpthread $(FUZZ_EXTRA_FLAGS)

# build only; useful on its own to check the harnesses still compile
fuzz-build: $(addprefix fuzz-,$(FUZZ_TARGETS))

# allocator_may_return_null: a size read straight out of an untrusted file can
# ask for tens of GB. Aborting on that turns every run into the same
# already-known resource finding and hides everything behind it; returning NULL
# instead makes the allocation *fail*, which is the behaviour on any host
# without memory overcommit and which these parsers are supposed to handle. It
# also means the OOM paths get fuzzed rather than skipped.
FUZZ_SAN_OPTS = allocator_may_return_null=1:max_allocation_size_mb=1024

# gguf_open mutes its own stderr per call (see the harness); log_path keeps
# sanitizer reports that are raised inside the muted window.
FUZZ_ENV_gguf_open = ASAN_OPTIONS=$(FUZZ_SAN_OPTS):log_path=.build/fuzz-findings/gguf_open/asan \
                     UBSAN_OPTIONS=log_path=.build/fuzz-findings/gguf_open/ubsan
# a valid GGUF header is ~8 KB; without a cap libFuzzer sizes inputs from the
# largest seed and spends the budget copying weights instead of parsing
FUZZ_ARGS_gguf_open = -max_len=16384 -dict=tests/fuzz/gguf.dict
# a request header is kilobytes at most; without a cap the mutator spends the
# budget on multi-megabyte buffers that reach no branch the small ones miss
FUZZ_ARGS_http_request = -max_len=16384 -dict=tests/fuzz/http.dict
FUZZ_ARGS_tokenizer = -max_len=8192
FUZZ_ARGS_model_load = -max_len=1048576 -dict=tests/fuzz/gguf.dict
FUZZ_ARGS_gguf_split = -max_len=1048576
FUZZ_ENV_model_load = ASAN_OPTIONS=$(FUZZ_SAN_OPTS):log_path=.build/fuzz-findings/model_load/asan UBSAN_OPTIONS=halt_on_error=1:log_path=.build/fuzz-findings/model_load/ubsan
FUZZ_ENV_gguf_split = ASAN_OPTIONS=$(FUZZ_SAN_OPTS):log_path=.build/fuzz-findings/gguf_split/asan UBSAN_OPTIONS=halt_on_error=1:log_path=.build/fuzz-findings/gguf_split/ubsan

# $(foreach) not a shell loop: the per-target FUZZ_ENV_*/FUZZ_ARGS_* lookups
# have to happen while make is expanding, which `for t in ...; $(VAR_$$t)`
# cannot do (make would resolve the name before the shell ever sets $t)
fuzz-run: fuzz-build
	$(PYTHON) scripts/fuzz-seeds.py fuzz-corpus
	@$(foreach t,$(FUZZ_TARGETS), \
		echo "== fuzzing $(t) for $(FUZZ_TIME)s =="; \
		mkdir -p fuzz-corpus/$(t) .build/fuzz-findings/$(t); \
		env ASAN_OPTIONS=$(FUZZ_SAN_OPTS) $(FUZZ_ENV_$(t)) \
		    ./fuzz-$(t) fuzz-corpus/$(t) tests/fuzz/corpus/$(t) \
			-max_total_time=$(FUZZ_TIME) -rss_limit_mb=$(FUZZ_RSS_MB) \
			-malloc_limit_mb=1024 \
			-timeout=25 -artifact_prefix=.build/fuzz-findings/$(t)/ \
			-print_final_stats=1 $(FUZZ_ARGS_$(t)) \
			|| { cat .build/fuzz-findings/$(t)/asan.* .build/fuzz-findings/$(t)/ubsan.* 2>/dev/null; exit 1; }; \
	)
	@echo "fuzz: all targets clean"

fuzz:
	@if command -v $(FUZZ_CLANG) > /dev/null 2>&1; then \
		$(MAKE) --no-print-directory fuzz-run; \
	else \
		echo "make fuzz: skipped -- '$(FUZZ_CLANG)' is not on PATH."; \
		echo "            libFuzzer needs clang; install it or set FUZZ_CLANG=<path>."; \
	fi

clean:
	rm -f test-moe-fixture.*.gguf test-q8.gguf test-bf16.gguf runner runner-debug runner-no-manifest.exe $(TEST_UTF8_CONSOLE) $(TEST_JSON_SCHEMA) $(TEST_SVAL_WALK) $(TEST_JSON_OOM) \
		$(TEST_TEMPLATE_OOM) $(TEST_TEMPLATE_FAMILIES) \
	      $(TEST_SCHEMA_OOM) $(TEST_SAMPLER) $(TEST_TOKENIZER) \
	      $(TEST_TOKENIZER_OOM) $(TEST_TOK_CLASSES) $(TEST_TEMPLATE) $(TEST_PROMPT_MARKS) $(TEST_SHARED) $(TEST_MLOCK) \
	      $(TEST_BATCH) $(TEST_BIND) $(TEST_HOST_HEADER) $(TEST_VRAMREG) $(TEST_UPLOAD_REPORT) $(TEST_TRAIN_GPU) test-shared-asan-bin \
	      $(TEST_KV_TOL) $(TEST_KV_FP4) $(TEST_CUDA_KVFP4) $(TEST_TC_TOL) $(TEST_I8_TOL) $(TEST_MV_TOL) $(TEST_ATTN_TOL) $(TEST_GPU_ID) $(TEST_MOE_TOL) $(TEST_MOE_ROUTER) $(TEST_PAGING_WARN) $(TEST_AUTOFIT) $(TEST_RESP_SM) $(TEST_PREFIX) $(TEST_GRAMMAR_FF) $(TEST_CHOICE_MASS) $(TEST_TOOLS) $(DIFFTOK) \
	      $(TEST_QUANTS_SIMD) $(TEST_INSTANCES) $(TEST_INSTANCES_OOM) $(TEST_METAL_ADMISSION) $(TEST_TRAY_CORE) \
	      $(TEST_QUANTIZE) $(TEST_VRAM_ROLLBACK) $(TEST_GGUF_GETTERS) $(TEST_HFHUB) \
	      $(TEST_PARSE) $(TEST_THREAD_DEFAULT) $(TEST_CAPACITY) $(TEST_METAL_OWNERSHIP) $(TEST_METAL_SHADERS) $(TEST_METAL_KQUANTS) $(TEST_MODEL_LOAD_FAILURE) $(TEST_MODEL_LOAD_OOM) test-load-oom-* test-model-load-oom.err $(TEST_QUANTIZE_OOM) test-quantize-oom* $(TEST_MOE_HOST_BRIDGE) test-bridge-nemo* test-moe-host-bridge.err $(TEST_LORA_CHAINS) $(TEST_MVT_CHAINS) $(TEST_DN_RECURRENCE) $(TEST_ATTN_BW) \
	      $(TEST_FILE_ID) test-file-identity.tmp \
	      $(TEST_BUDGET) $(TEST_ATTRIB) $(TEST_MSG_OOM) $(TEST_STOP_CONSTRAINT) $(TEST_ROPE_YARN) $(TEST_ATTN_SCALE) $(TEST_DOT_ROUTE) \
	      $(TMPL_CONF_RENDER) \
	      $(TEST_SPLIT_GUARD) split-guard.out test-swap-race-bin test-tokenizer-race-bin test-decide-race-bin test-decide-race-gpu-bin \
	      runner-gpu-stub
	rm -rf test-attn
	rm -rf .build
	rm -f shared-noid.out
	rm -f metal-cpu.out metal-fallback.out metal-fallback.err
	rm -f metal-init-fallback.out metal-init-fallback.err
	rm -f metal-prefill-loop.out metal-prefill-native.out metal-prefill-native.err
	rm -f metal-decode1-cpu.out metal-decode1-gpu.out metal-decode1-gpu.err
	rm -f metal-split-cpu.out metal-split-gpu.out metal-split-gpu.err
	rm -f metal-split-whole.out metal-split-whole.err
	rm -f metal-bind-cpu.out metal-bind-gpu.out metal-bind-gpu.err
	rm -f metal-kv-q8.err metal-moe-dense.out metal-moe1.out metal-moe1.err
	rm -f metal-moe2.out metal-moe2.err
	rm -f metal-gptoss-moe-cpu.out metal-gptoss-moe-gpu.out metal-gptoss-moe-gpu.err
	rm -f metal-gemma4-moe-cpu.out metal-gemma4-moe-gpu.out metal-gemma4-moe-gpu.err
	rm -f metal-swa-cpu.out metal-swa-gpu.out metal-swa-gpu.err test-swa.gguf
	rm -f test-spark.gguf test-spark-swa.gguf test-spark-q8.gguf test-spark-q40.gguf
	rm -f metal-es-cpu.out metal-es-gpu.out metal-es-gpu.err test-es.gguf test-es-cfg.gguf
	rm -f $(addprefix fuzz-,$(FUZZ_TARGETS))
	rm -rf fuzz-corpus

# regenerate the committed PTX headers. Normal builds and CI use the committed
# src/kernels_ptx.h and src/kernels_ptx80.h. Run it through
# tools/ptx-toolchain/make-ptx.sh, which builds the pinned toolchain
# (tools/ptx-toolchain/ptx-linux-64.lock) and reproduces the committed headers
# byte for byte; another nvcc, other runtime headers or another host compiler
# change kernels (R8.23). The ptx-reproducible workflow checks it.
NVCC ?= nvcc
NVCC_CCBIN ?=
ptx: src/kernels.cu src/kernels_sm80.cu src/kernels_tc2.cuh src/kernels_tc2_iq.cuh \
     src/kernels_iq4_table.cuh src/kernels_iq_grids.cuh src/kernels_iq_stage.cuh src/iq_decode.h \
     src/kernels_kq_stage.cuh src/kernels_tc2_kq.cuh src/kq_stage.h src/kernels_train.cuh src/kernel_args.h \
     src/tdn_args.h
	$(NVCC) $(if $(NVCC_CCBIN),-ccbin $(NVCC_CCBIN)) -ptx -arch=compute_75 -O3 -o src/kernels.ptx src/kernels.cu
	$(NVCC) $(if $(NVCC_CCBIN),-ccbin $(NVCC_CCBIN)) -ptx -arch=compute_80 -O3 -o src/kernels_sm80.ptx src/kernels_sm80.cu
	python3 scripts/embed-ptx.py || python scripts/embed-ptx.py
	EMBED_IN=src/kernels_sm80.ptx EMBED_OUT=src/kernels_ptx80.h EMBED_NAME=k_ptx80_src python3 scripts/embed-ptx.py || \
	EMBED_IN=src/kernels_sm80.ptx EMBED_OUT=src/kernels_ptx80.h EMBED_NAME=k_ptx80_src python scripts/embed-ptx.py

# A duplicate target name makes make DISCARD one recipe and say so — but only
# as a warning, scrolling past in the build noise. That is exactly how
# test-split-guard sat unbuilt and unreferenced for as long as it existed: the
# diagnosis was printed on every single build and nobody was reading it.
#
# "overriding commands for target" is never benign. It always means a recipe
# was silently thrown away. Promote it to a failure rather than a warning.
#
# -n on a do-nothing target is enough: these warnings are emitted while make
# PARSES the makefile, before any recipe runs, so this costs a parse and no
# work at all.
makefile-noop:
	@:

FORCE:
	@:

test-python-deps:
	@$(PYTHON) -c "import pytest" >/dev/null 2>&1 || { \
		echo "error: make test requires pytest for $(PYTHON)"; \
		echo "install it with '$(PYTHON) -m pip install pytest' or set PYTHON to an interpreter that already has it"; \
		exit 1; \
	}
	@[ -z "$(PYTEST_WORKERS)" ] || $(PYTHON) -c "import xdist" >/dev/null 2>&1 || { \
		echo "error: PYTEST_WORKERS=$(PYTEST_WORKERS) needs pytest-xdist for $(PYTHON) (tests/ci/requirements.txt)"; \
		exit 1; \
	}
	@echo "python test dependencies ok ($(PYTHON))"

# Local commit-msg hook for the publication rule (AGENTS.md); CI runs the same
# script over every pull request, this only moves the refusal to commit time.
hooks:
	git config core.hooksPath .githooks
	@echo "commit-msg hook installed (core.hooksPath = .githooks)"

test-makefile-sane:
	@out=$$($(MAKE) -n --no-print-directory makefile-noop 2>&1); \
	case "$$out" in \
	  *"overriding commands"*) \
	    echo "FAIL: duplicate make target — a recipe is being discarded:"; \
	    echo "$$out" | grep -E "overriding commands|ignoring old commands"; \
	    exit 1;; \
	esac; \
	qline=$$($(MAKE) -Bn --no-print-directory runner | grep -- ' -c src/quants.c '); \
	test -n "$$qline" || { echo "FAIL: quants.c is not a separate translation unit"; exit 1; }; \
	echo "$$qline" | grep -q -- ' -fno-fast-math ' || { echo "FAIL: quants.c lacks -fno-fast-math"; exit 1; }; \
	case "$$qline" in *" -ffast-math "*) echo "FAIL: quants.c still has -ffast-math"; exit 1;; esac; \
	zline=$$($(MAKE) -Bn --no-print-directory runner | grep -- ' -c src/quantize.c '); \
	test -n "$$zline" || { echo "FAIL: quantize.c is not a separate translation unit"; exit 1; }; \
	echo "$$zline" | grep -q -- ' -fno-fast-math ' || { echo "FAIL: quantize.c lacks -fno-fast-math"; exit 1; }; \
	case "$$zline" in *" -ffast-math "*) echo "FAIL: quantize.c still has -ffast-math"; exit 1;; esac; \
	bline=$$($(MAKE) -Bn --no-print-directory CFLAGS=-O0 runner | grep -- ' src/model.c '); \
	test -n "$$bline" || { echo "FAIL: release-style build has no model.c compile line"; exit 1; }; \
	echo "$$bline" | grep -q -- ' $(GPU_BACKEND_DEF) ' || { \
		echo "FAIL: command-line CFLAGS dropped backend identity $(GPU_BACKEND_DEF)"; \
		exit 1; \
	}; \
	if grep -q 'system(' src/tray.c src/tray_*.c src/tray_*.m; then echo "FAIL: tray launches through a shell"; exit 1; fi; \
	for f in src/model.c src/engine.c src/completion.c src/server.c src/scheduler.c src/registry.c src/provenance.c src/lineage.c src/respstore.c src/receipts.c src/watermark.c src/kvsnap.c src/session.c src/api_responses.c src/api_anthropic.c src/template.c $(SCHEMA_SRC) src/jsonmode.c src/tokenizer.c src/sample.c src/envelope.c src/oms.c src/http.c src/json.c src/gguf.c; do \
	  $(CC) -fsyntax-only -std=gnu11 -Werror=vla -DRUNNER_GPU_CUDA -I src $$f 2>/dev/null || { \
	    echo "FAIL: $$f declares a variable-length stack array (the forward runs on 512 KB server threads; size by file values goes on the heap)"; exit 1; }; done; \
	tline=$$($(MAKE) -Bn --no-print-directory T3=1 CFLAGS="-O3 -ffast-math" runner | grep -- ' src/model.c '); \
	test -n "$$tline" || { echo "FAIL: T3 build has no model.c compile line"; exit 1; }; \
	case "$$tline" in *" -ffast-math "*) echo "FAIL: T3=1 left -ffast-math in the engine build"; exit 1;; esac; \
	for f in -fno-fast-math -ffp-contract=off -DRUNNER_PORTABLE_MATH -DRUNNER_CANON_KERNELS -DRUNNER_T3_BUILD; do \
	  echo "$$tline" | grep -q -- " $$f " || { echo "FAIL: T3=1 build lacks $$f"; exit 1; }; done; \
	echo "makefile ok (no discarded recipes; T3 switch reaches the engine)"


.PHONY: template-conformance template-conformance-refresh template-conformance-baseline template-conformance-harmony-oracle
# The pytest cases that skip without a CUDA device, which hosted CI never has,
# for a CUDA host: a case still skipped there for want of CUDA fails the
# target. scripts/cuda-gate.sh runs it as a step. (Until it did, a failing
# split-order test went three weeks without a run.)
CUDA_GATED_TESTS = tests/test_apertus.py tests/test_gpu_declines.py \
                   tests/test_gpu_split_order.py tests/test_hybrid_admission.py \
                   tests/test_moe.py tests/test_ornith_cpu.py tests/test_train.py \
                   tests/test_train_teacher.py tests/test_bench_json.py tests/test_clef.py \
                   tests/test_remove_sublayer.py tests/test_lora_qwen35.py \
                   tests/test_transcript.py tests/test_spec_gpu.py \
                   tests/test_server_sessions.py
# test_spec_gpu.py reads test.gguf from the tree root (the others build their
# fixtures in a temporary directory), so the target builds it: on a fresh
# tree the gate's first run otherwise fails the three spec cases.
test-cuda-gated: runner test.gguf
	@mkdir -p .build/ci
	$(PYTHON) -m pytest -q -rs --junitxml=.build/ci/cuda-gated.xml $(CUDA_GATED_TESTS)
	$(PYTHON) scripts/check-junit.py --no-skips-matching '(?i)cuda|device|gpu' \
	    --allow-skip 'CUDA runtime is available' .build/ci/cuda-gated.xml

.PHONY: test-gpu-stub test-cuda-nvfp4 test-cuda-qwen4exp test-ts-client test-cuda-gated test-cuda-batch-identity test-cuda-spec-identity
.PHONY: test-gates test-gates-core test-gates-batch test-python
.PHONY: test-metal-kv-fp4 FORCE makefile-noop test-python-deps test-makefile-sane test-cuda-iquants test-tc-overflow fixture-scale-note clean debug ptx test test-bare-invocation test-help-interface test-shader-embed test-metal-shader-gate test-apertus test-moe test-prune-experts test-metal-fallback test-metal-prefill test-metal-kquant test-metal-decode-only test-metal-split test-metal-bind-failure test-metal-kv-q8 test-metal-moe test-metal-gptoss-moe test-metal-gemma4-moe test-metal-gemma4-hetero test-metal-bigmodel test-metal-bigmodel-multibuf test-metal-moe-em test-metal-moe-mm test-metal-fuse test-metal-gelu-overflow test-metal-eseries test-metal-swa smoke release-check test-truncation fuzz fuzz-build fuzz-run test-shared-asan test-shared-noid test-split-guard test-swap-race

# The TypeScript client (R10.7): type-checked with the pinned compiler and run
# against a live runner on the CI fixture. Needs Node 18+ and npm (the pinned
# typescript is fetched by `npm ci`), so it is its own target and a CI step in
# the consumer-compatibility job rather than part of `make test`.
test-ts-client: $(RUNNER_EXE)
	cd clients/typescript && npm ci --ignore-scripts && npx tsc -p tsconfig.json && \
	  node --test test/client.test.mjs

# Soak harness for the startup/SIGTERM race (test_signal_during_startup). Not
# in `make test` — it is a diagnostic soak (thousands of spawns), run on demand
# or as a CI soak job. Exits 1 on a survivor, retaining its startup output.
.PHONY: repro-startup-signal
repro-startup-signal: runner test.gguf
	$(PYTHON) scripts/repro-startup-signal.py --iterations 6000 --concurrency 12 --load

# CI-only race lanes: CPU stub avoids driver-owned threads and allocations.
TSAN_FLAGS = -O1 -g -fsanitize=thread -fno-omit-frame-pointer -fno-fast-math -std=gnu11 -Wall -I src
test-pool-race:
	$(CC) $(TSAN_FLAGS) tests/test_thread_default.c src/compat.c src/quants.c -o test-pool-race-bin $(LDFLAGS)
	TSAN_OPTIONS=halt_on_error=1 ./test-pool-race-bin

test-server-race-tsan: test.gguf
	$(CC) $(TSAN_FLAGS) $(filter-out $(GPU_SRC),$(TEST_SWAP_RACE_SRC)) src/gpu_none.c -o test-server-race-tsan-bin $(LDFLAGS)
	TSAN_OPTIONS=halt_on_error=1 ./test-server-race-tsan-bin test.gguf

# /v1/decide and /v1/rerank number their answers from counters every slot
# shares; slots answer on their own threads. TSan halting on an unguarded
# increment is the gate (tests/test_decide_race.c). CI runs it beside the two
# above; MinGW has no TSan.
test-decide-race: test.gguf
	$(CC) $(TSAN_FLAGS) tests/test_decide_race.c $(filter-out tests/test_swap_race.c $(GPU_SRC),$(TEST_SWAP_RACE_SRC)) src/gpu_none.c -o test-decide-race-bin $(LDFLAGS)
	TSAN_OPTIONS=halt_on_error=1 ./test-decide-race-bin test.gguf
# The same with the build's GPU backend under the four engines (Metal on a
# Mac, CUDA where the build has it), so the backend's own first-use state is
# raced too: kernel-selection flags, dispatch counters, once-only notices,
# the driver load. Needs the device, so CI's hosted runners cannot run it.
# MinGW has no TSan: there it builds plain and is the functional half only
# (four engines on the GPU at once, every answer and every id distinct).
ifeq ($(BUILD_OS),Windows)
DECIDE_RACE_GPU_FLAGS = -O1 -g -std=gnu11 -Wall -I src
DECIDE_RACE_GPU_RUN   =
else
DECIDE_RACE_GPU_FLAGS = $(TSAN_FLAGS)
DECIDE_RACE_GPU_RUN   = TSAN_OPTIONS=halt_on_error=1
endif
test-decide-race-gpu: test.gguf
	$(CC) $(DECIDE_RACE_GPU_FLAGS) $(GPU_BACKEND_DEF) tests/test_decide_race.c $(filter-out tests/test_swap_race.c,$(TEST_SWAP_RACE_SRC)) -o test-decide-race-gpu-bin $(LDFLAGS)
	$(DECIDE_RACE_GPU_RUN) ./test-decide-race-gpu-bin test.gguf gpu

.PHONY: test-pool-race test-server-race-tsan test-decide-race test-decide-race-gpu

fuzz-replay-%: tests/fuzz/fuzz_%.c tests/fuzz/replay.c tests/fuzz/fuzz_files.h $(HDR)
	$(FUZZ_CLANG) $(filter-out $(FUZZ_INSTRUMENTATION),$(FUZZ_FLAGS)) -fsanitize=address,undefined tests/fuzz/replay.c tests/fuzz/fuzz_$*.c $(FUZZ_SRC_$*) -o $@ -lm -lpthread

