#!/bin/sh
# R6.91: run tests/test_sample_nan.c's three builds (default, portable, strict),
# made by `make test-sample-nan`, and check:
#   - no build picks a token whose logit is NaN (each binary's own check);
#   - the finite picks' digest is the same in all three builds;
#   - it is the digest of main's sampler (BASE, the fix's base) compiled the same
#     way with the default flags, when that commit is in the clone. A shallow CI
#     clone may not hold it; then only the three builds' agreement is checked.
# usage: check-sample-nan.sh EXT BASE CC CFLAGS... -- LDFLAGS...
set -u
ext=$1; base=$2; cc=$3; shift 3
cflags=""; while [ $# -gt 0 ] && [ "$1" != "--" ]; do cflags="$cflags $1"; shift; done
[ $# -gt 0 ] && shift
ldflags="$*"
out=.sample-nan
mkdir -p "$out"
fail=0
for k in default portable strict; do
    ./test-sample-nan-$k$ext > "$out/$k.txt" || fail=1
    grep -E '^(FAIL|digest)' "$out/$k.txt"
done
if [ $fail != 0 ]; then echo "sample-nan: a build picked a NaN token"; exit 1; fi
d=$(sed -n 's/^digest default: .* \([0-9a-f]*\)$/\1/p' "$out/default.txt")
for k in portable strict; do
    if ! grep -q " $d\$" "$out/$k.txt"; then
        echo "sample-nan: the $k build's finite picks differ from the default build's ($d)"; exit 1
    fi
done
if git cat-file -e "$base:src/sample.c" 2>/dev/null; then
    git show "$base:src/sample.c" > "$out/base-sample.c"
    # shellcheck disable=SC2086
    $cc $cflags -I src tests/test_sample_nan.c "$out/base-sample.c" -o "$out/base$ext" $ldflags || exit 1
    b=$("$out/base$ext" | sed -n 's/^digest default: .* \([0-9a-f]*\)$/\1/p')
    if [ "$b" != "$d" ]; then
        echo "sample-nan: the default build's finite picks moved: $d, main's sampler gives $b"; exit 1
    fi
    echo "sample-nan: ok: no NaN token in the default, portable and strict builds; finite picks $d in all three, as main's sampler gives"
else
    echo "sample-nan: ok: no NaN token in three builds; finite picks $d in all three (main's sampler is not in this clone)"
fi
