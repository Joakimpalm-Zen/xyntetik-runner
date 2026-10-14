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

#endif // RUNNER_SCHEMA_INT_H
