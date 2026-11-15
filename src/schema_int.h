// What schema.c's compiler lends the native tool grammars in schema_tools.c:
// the node constructor, the enum literal order, the length and item bound
// reader, and the JSON-schema compiler itself. The validator and the closer
// keep their state private to schema.c.
#ifndef RUNNER_SCHEMA_INT_H
#define RUNNER_SCHEMA_INT_H

#include <stdbool.h>

#include "json.h"
#include "schema.h"

snode *sn_new(int kind);                 // NULL on allocation failure
bool   enum_index(snode *n);             // sorts n->lits into n->order
bool   compile_bound(jv *v, int dflt, int *out);
snode *compile_node(jv *s, char *err, int errcap, int depth);
// schema_compile without the $ref pass, for the wrappers in schema_tools.c
snode *schema_compile_raw(jv *schema, char *err, int errcap);

// R2.18: one public compile under way, for the per-schema cap on compiled
// pattern automata (schema.c); every public entry brackets its compile.
// R2.19: the work of the last bounded-number decision on this thread (digit
// pairs compared, spelling family members and exponent widths tried), and
// how many exact bound expansions this thread has built (cached per bounds)
unsigned long schema_witness_work(void);
unsigned long schema_number_expansions(void);
void schema_automata_begin(void);
void schema_automata_end(void);

#endif // RUNNER_SCHEMA_INT_H
