// Native Inkling 2 framing, from the publisher's chat_template.jinja.
// Included by template.c so the prompt-marker discipline stays private.
static const jv *inkling_kwarg(const jv *req) {
    jv *kw = jv_get((jv *)req, "chat_template_kwargs");
    jv *v = jv_get(kw, "reasoning_effort");
    return v ? v : jv_get((jv *)req, "reasoning_effort");
}

static bool inkling_effort(const jv *v, double *number) {
    *number = 0.9;
    if (!v || v->type == J_NULL) return true;
    if (v->type == J_STR) {
        static const char *const names[] = {"none", "minimal", "low", "medium", "high", "xhigh", "max"};
        static const double levels[] = {0, 0.1, 0.2, 0.7, 0.9, 0.99, 0.99};
        const char *s = v->str;
        size_t n = strlen(s);
        py_strip(&s, &n);
        for (size_t i = 0; i < sizeof levels / sizeof *levels; i++)
            if (strlen(names[i]) == n && !memcmp(names[i], s, n)) {
                *number = levels[i]; return true;
            }
        return false;
    }
    // jinja's float filter defaults to zero for a value it cannot convert.
    *number = v->type == J_NUM ? v->num : v->type == J_BOOL && v->b ? 1 : 0;
    return isfinite(*number) && *number >= 0 && *number <= 0.99;
}

static void inkling_json(const jv *v, sbuf *b, bool sorted) {
    if (v && v->type == J_OBJ) {
        int *order = malloc(sizeof(int) * (size_t)(v->n ? v->n : 1));
        if (!order) { b->failed = true; return; }
        for (int i = 0; i < v->n; i++) {
            int at = i;
            while (sorted && at > 0 && strcmp(v->keys[order[at - 1]], v->keys[i]) > 0) {
                order[at] = order[at - 1]; at--;
            }
            order[at] = i;
        }
        sb_lit(b, "{");
        for (int i = 0; i < v->n; i++) {
            int k = order[i];
            if (i) sb_lit(b, ",");
            sb_str(b, v->keys[k]); sb_lit(b, ":");
            inkling_json(v->items[k], b, sorted);
        }
        sb_lit(b, "}"); free(order);
    } else if (v && v->type == J_ARR) {
        sb_lit(b, "[");
        for (int i = 0; i < v->n; i++) {
            if (i) sb_lit(b, ",");
            inkling_json(v->items[i], b, sorted);
        }
        sb_lit(b, "]");
    } else {
        jv_dump(v, b);
        if (v && v->type == J_NUM && !v->num_int && !v->num_text &&
            v->num > -1e16 && v->num < 1e16 && v->num == trunc(v->num))
            sb_lit(b, ".0");
    }
}

static void inkling_header(sbuf *b, const char *role, const char *name, const char *channel) {
    const char *tag = !strcmp(role, "system") ? "<|message_system|>"
                    : !strcmp(role, "tool") ? "<|message_tool|>"
                    : !strcmp(role, "assistant") ? "<|message_model|>" : "<|message_user|>";
    pl_lit(b, tag);
    if (name) sb_put(b, name, strlen(name));
    pl_lit(b, channel);
}

static void inkling_text(sbuf *b, const char *role, const char *name, const char *text) {
    inkling_header(b, role, name, "<|content_text|>");
    if (text) sb_put(b, text, strlen(text));
    pl_lit(b, "<|end_message|>");
}

static size_t render_inkling(const chat_msg *msgs, int n_msgs, const jv *tools,
                              char *out, size_t cap) {
    sbuf b = {0};
    if (tools && tools->type == J_ARR && tools->n) {
        pl_lit(&b, "<|message_system|>tool_declare<|content_xml|>");
        sb_lit(&b, "[");
        for (int i = 0; i < tools->n; i++) {
            const jv *tool = tools->items[i], *fn = jv_get((jv *)tool, "function");
            if (!fn) fn = tool;
            if (i) sb_lit(&b, ",");
            sb_lit(&b, "{\"description\":");
            const char *desc = jv_str(jv_get((jv *)fn, "description"), "");
            sb_str(&b, desc);
            sb_lit(&b, ",\"name\":"); jv_dump(jv_get((jv *)fn, "name"), &b);
            sb_lit(&b, ",\"parameters\":");
            jv *params = jv_get((jv *)fn, "parameters");
            if (params && params->n) inkling_json(params, &b, true); else sb_lit(&b, "{}");
            sb_lit(&b, ",\"type\":");
            sb_str(&b, jv_str(jv_get((jv *)tool, "type"), "function"));
            sb_lit(&b, "}");
        }
        sb_lit(&b, "]"); pl_lit(&b, "<|end_message|>");
    }
    bool effort_emitted = false;
    const jv *effort = n_msgs ? msgs[0].effort : NULL;
    for (int i = 0; i <= n_msgs; i++) {
        if (!effort_emitted && (i == n_msgs || strcmp(msgs[i].role, "system"))) {
            double level;
            if (!inkling_effort(effort, &level)) { free(b.s); return SIZE_MAX; }
            char number[32]; json_format_number(level, number);
            pl_lit(&b, "<|message_system|><|content_text|>Thinking effort level: ");
            sb_put(&b, number, strlen(number)); pl_lit(&b, "<|end_message|>");
            effort_emitted = true;
        }
        if (i == n_msgs) break;
        const chat_msg *m = msgs + i;
        const jv *raw = m->source;
        const char *name = m->name;
        if (raw && !strcmp(m->role, "assistant"))
            name = jv_str(jv_get((jv *)raw, "name"), NULL);
        const char *reason = raw ? jv_str(jv_get((jv *)raw, "reasoning_content"), NULL) : NULL;
        const jv *calls = raw ? jv_get((jv *)raw, "tool_calls") : NULL;
        if (!strcmp(m->role, "assistant") && reason && reason[0]) {
            inkling_header(&b, m->role, name, "<|content_thinking|>");
            sb_put(&b, reason, strlen(reason)); pl_lit(&b, "<|end_message|>");
        }
        const jv *content = raw ? jv_get((jv *)raw, "content") : NULL;
        if (!raw) inkling_text(&b, m->role, name, m->content);
        else if (content && content->type == J_STR) {
            if (content->str[0] || strcmp(m->role, "assistant") ||
                !((calls && calls->n) || (reason && reason[0])))
                inkling_text(&b, m->role, name, content->str);
        } else if (content && content->type == J_ARR) {
            for (int k = 0; k < content->n; k++) {
                jv *part = content->items[k];
                const char *type = jv_str(jv_get(part, "type"), "text");
                if (part->type == J_STR) inkling_text(&b, m->role, name, part->str);
                else if (!strcmp(type, "text") || !strcmp(type, "input_text"))
                    inkling_text(&b, m->role, name, jv_str(jv_get(part, "text"), ""));
                else if (!strcmp(type, "image") || !strcmp(type, "input_image") || !strcmp(type, "image_url")) {
                    if (strcmp(m->role, "user") && strcmp(m->role, "tool")) {
                        free(b.s); return SIZE_MAX;
                    }
                    // Template placeholder only; this is not image encoding.
                    inkling_header(&b, m->role, name, "<|content_image|>");
                    pl_lit(&b, "<|unused_200054|><|end_message|>");
                } else { free(b.s); return SIZE_MAX; }
            }
        }
        if (!strcmp(m->role, "assistant")) {
            for (int k = 0; calls && calls->type == J_ARR && k < calls->n; k++) {
                jv *fn = jv_get(calls->items[k], "function");
                const char *called = jv_str(jv_get(fn, "name"), "");
                jv *args = jv_get(fn, "arguments"), *parsed = NULL;
                if (args && args->type == J_STR && !args->str[0]) args = NULL;
                if (args && args->type == J_STR) {
                    parsed = json_parse(args->str, strlen(args->str));
                    if (!parsed) { free(b.s); return SIZE_MAX; }
                    args = parsed;
                }
                if ((args && args->type != J_OBJ) || (name && name[0] && strcmp(name, called))) {
                    jv_free(parsed); free(b.s); return SIZE_MAX;
                }
                inkling_header(&b, "assistant", called, "<|content_invoke_tool_json|>");
                sb_lit(&b, "{\"name\":"); sb_str(&b, called); sb_lit(&b, ",\"args\":");
                // dictsort applies only to the argument object's outer keys.
                if (args) {
                    int *order = malloc(sizeof(int) * (size_t)(args->n ? args->n : 1));
                    if (!order) b.failed = true;
                    else {
                        for (int j = 0; j < args->n; j++) {
                            int at = j;
                            while (at > 0 && strcmp(args->keys[order[at - 1]], args->keys[j]) > 0) {
                                order[at] = order[at - 1]; at--;
                            }
                            order[at] = j;
                        }
                        sb_lit(&b, "{");
                        for (int j = 0; j < args->n; j++) {
                            int at = order[j]; if (j) sb_lit(&b, ",");
                            sb_str(&b, args->keys[at]); sb_lit(&b, ":");
                            inkling_json(args->items[at], &b, false);
                        }
                        sb_lit(&b, "}"); free(order);
                    }
                } else sb_lit(&b, "{}");
                sb_lit(&b, "}"); pl_lit(&b, "<|end_message|>"); jv_free(parsed);
            }
            pl_lit(&b, "<|content_model_end_sampling|>");
        }
    }
    if (b.failed) { free(b.s); return SIZE_MAX; }
    size_t n = b.n;
    if (cap) { size_t copy = n < cap - 1 ? n : cap - 1; memcpy(out, b.s, copy); out[copy] = 0; }
    free(b.s); return n;
}
