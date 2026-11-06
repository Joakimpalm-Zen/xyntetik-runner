#!/bin/sh
# The R=8 row, pinned side load against pageable, three interleaved rounds. Each
# server: request 0 cold, 1-2 warm clean, 3-4 warm under the side load.
cd $R330_DIR
export SIDE_FROM=3 RUNNER_BIN=$R330_BASE EXPECT_RAM=4.5-7.5
for r in 1 2 3; do
  case $r in 2) L="pin page";; *) L="page pin";; esac
  for c in $L; do date -u +%H:%M:%S
    case $c in
      pin) python serve_read.py p$r-pin out/hot-m2.txt 5 128 --rate 380 --seconds 7 --list out/adm-r8.txt --from-pinned 2>&1 | grep -v pagein;;
      page) python serve_read.py p$r-page out/hot-m2.txt 5 128 --rate 380 --seconds 7 --list out/adm-r8.txt 2>&1 | grep -v pagein;;
    esac
  done
done
date -u +%H:%M:%S
