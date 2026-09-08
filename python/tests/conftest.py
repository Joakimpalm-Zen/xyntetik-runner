# The shadow fixtures carry real pytest files on purpose (a frozen repair
# task and its protected tests); the verifier runs those in a scratch copy.
# They are not this suite's tests, so collection must not descend into them.
collect_ignore_glob = ["fixtures/*"]
