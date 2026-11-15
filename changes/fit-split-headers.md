- **`--info fit` answers for a model in several parts.** A model published as `<prefix>-00001-of-0000N.gguf` keeps each
  part's tensors in that part's own table. The fit check read the first part alone, so it answered "WILL NOT LOAD: the
  loader refuses this file: missing tensor ..." for a set the loader takes. Found on both split files of the support
  list, on a complete 2-part download.
  - The header read now covers every part of the set, from any part's path, and reads the data of none, so partial
    downloads still work as long as each part has its header.
  - A part that is missing is named ("missing or unreadable split GGUF part 2/2") and no verdict is given from the
    parts at hand.
  - The same read serves the other header-only callers (derive's parent metadata, the draft-head check).
