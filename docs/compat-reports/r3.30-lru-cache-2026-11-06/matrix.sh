#!/bin/sh
# Three interleaved rounds. Each server: request 0 cold, 1-2 warm clean, 3-4 warm under the side load.
cd $R330_DIR
export SIDE_FROM=3
one() { # tag hot rate list
  if [ -n "$3" ]; then python serve_read.py $1 $2 5 128 --rate $3 --seconds 7 --list $4 2>&1 | grep -v pagein
  else python serve_read.py $1 $2 3 128 2>&1 | grep -v pagein; fi
}
for r in 1 2 3; do
  case $r in 1) L="all auto m1 m2 m3";; 2) L="m3 m2 auto m1 all";; 3) L="m1 all m3 auto m2";; esac
  for c in $L; do
    date -u +%H:%M:%S
    case $c in
      all) one x$r-all out/hot-all.txt;;
      auto) one x$r-auto auto;;
      m1) one x$r-m1 out/hot-m1.txt 180 out/adm-r8.txt;;
      m2) one x$r-m2 out/hot-m2.txt 380 out/adm-r8.txt;;
      m3) one x$r-m3 out/hot-m3.txt 550 out/adm-r16.txt;;
    esac
  done
done
date -u +%H:%M:%S
