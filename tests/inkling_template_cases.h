// Absolute framing anchors from thinkingmachines/Inkling-2-Mini's
// chat_template.jinja at b85c3cfa615fc413cc9af3379f491b963f3b586a.
static const char INKLING_HISTORY[] =
    "[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"A\"},\"B\"]},"
    "{\"role\":\"assistant\",\"content\":\"\",\"reasoning_content\":\"why\",\"tool_calls\":["
    "{\"id\":\"c1\",\"function\":{\"name\":\"get_weather\",\"arguments\":{"
    "\"z\":1,\"a\":{\"z\":3,\"B\":4},\"A\":2}}}]},"
    "{\"role\":\"tool\",\"name\":\"get_weather\",\"tool_call_id\":\"c1\",\"content\":\"ok\"}]";

static void test_inkling_source_contract(void) {
    jv *source = json_parse(INKLING_HISTORY, strlen(INKLING_HISTORY));
    jv *effort = json_parse("\"low\"", 5);
    CHECK(source && effort, "Inkling structured fixtures parse");
    if (!source || !effort) { jv_free(source); jv_free(effort); return; }
    chat_msg m[3];
    for (int i = 0; i < 3; i++)
        m[i] = (chat_msg){ .role = jv_str(jv_get(source->items[i], "role"), ""),
                          .content = "", .source = source->items[i], .effort = effort,
                          .name = i == 2 ? "get_weather" : NULL };
    const char *want =
        "<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|>"
        "<|message_user|><|content_text|>A<|end_message|>"
        "<|message_user|><|content_text|>B<|end_message|>"
        "<|message_model|><|content_thinking|>why<|end_message|>"
        "<|message_model|>get_weather<|content_invoke_tool_json|>"
        "{\"name\":\"get_weather\",\"args\":{\"A\":2,\"a\":{\"z\":3,\"B\":4},\"z\":1}}"
        "<|end_message|><|content_model_end_sampling|>"
        "<|message_tool|>get_weather<|content_text|>ok<|end_message|>";
    render(TMPL_INKLING2, m, 3, true, THINK_ON, NULL, A, sizeof A);
    tok_strip_marks(A);
    CHECK(!strcmp(A, want), "Inkling parts/reason/call sorting/effort differ from publisher anchor");
    // Negative cases: losing each structured field must change the anchor.
    for (int field = 0; field < 4; field++) {
        const char *key = field == 0 ? "content" : field == 1 ? "reasoning_content" : "tool_calls";
        jv *owner = field == 0 ? source->items[0] : source->items[1];
        jv *value = jv_get(owner, key);
        int at = -1;
        for (int i = 0; i < owner->n; i++) if (!strcmp(owner->keys[i], key)) at = i;
        if (field < 3) owner->items[at] = NULL;
        else m[0].effort = NULL;
        render(TMPL_INKLING2, m, 3, true, THINK_ON, NULL, B, sizeof B);
        tok_strip_marks(B);
        CHECK(strcmp(B, want), "Inkling anchor ignores structured field %d", field);
        if (field < 3) owner->items[at] = value;
        else m[0].effort = effort;
    }
    render_state st;
    render_messages_state(TMPL_INKLING2, m, 3, true, THINK_ON, NULL, B, sizeof B, &st);
    CHECK(!st.reasoning_open && !st.opener, "Inkling must not invent a generation prelude");
    jv_free(source); jv_free(effort);
}

static void test_inkling_edge_parts(void) {
    const char *image = "{\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"not-read\"}}]}";
    jv *source = json_parse(image, strlen(image));
    CHECK(source != NULL, "Inkling image fixture parses");
    if (!source) return;
    chat_msg m = { .role = "user", .content = "", .source = source };
    for (int i = 0; i < 2; i++) {
        m.role = i ? "tool" : "user";
        render(TMPL_INKLING2, &m, 1, true, THINK_DEFAULT, NULL, A, sizeof A);
        tok_strip_marks(A);
        CHECK(strstr(A, "<|content_image|><|unused_200054|><|end_message|>") != NULL,
              "Inkling image placeholder differs from publisher");
        CHECK(!strstr(A, "not-read"), "Inkling placeholder must not insert the image URL");
    }
    m.role = "assistant";
    CHECK(render(TMPL_INKLING2, &m, 1, true, THINK_DEFAULT, NULL, A, sizeof A) == SIZE_MAX,
          "Inkling assistant image must be refused");
    jv_free(source);
    const char *empty = "{\"content\":\"\",\"tool_calls\":[{\"function\":{\"name\":\"f\",\"arguments\":\"\"}}]}";
    source = json_parse(empty, strlen(empty));
    CHECK(source != NULL, "Inkling empty arguments fixture parses");
    if (!source) return;
    m = (chat_msg){ .role = "assistant", .content = "", .source = source };
    render(TMPL_INKLING2, &m, 1, true, THINK_DEFAULT, NULL, A, sizeof A);
    tok_strip_marks(A);
    CHECK(strstr(A, "{\"name\":\"f\",\"args\":{}}<|end_message|>") != NULL,
          "Inkling empty argument string is the publisher's empty object");
    jv_free(source);
}
