#!/bin/sh
cd $R330_DIR
B=$R330_BASE; L=$R330_REL; PR="$R330_DIR/out/prime-s1.txt"
one() { # tag bin prime
  if [ -n "$3" ]; then RUNNER_BIN=$2 EXPECT_RAM=$4 PRIME=$PR python serve_read.py $1 out/hot-all.txt 3 128 2>&1
  else RUNNER_BIN=$2 EXPECT_RAM=$4 python serve_read.py $1 out/hot-all.txt 3 128 2>&1; fi
}
for r in 1 2 3; do
  case $r in 1) L4="base rel basep relp";; 2) L4="relp basep rel base";; 3) L4="rel basep base relp";; esac
  for c in $L4; do date -u +%H:%M:%S
    case $c in base) one z$r-base $B "" 4.5-7.5;; rel) one z$r-rel $L "" 9.5-12;; basep) one z$r-basep $B 1 4.5-7.5;; relp) one z$r-relp $L 1 9.5-12;; esac
  done
done
date -u +%H:%M:%S
