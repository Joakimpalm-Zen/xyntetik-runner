"""scripts/comment-only-check.py: a change to comments must not change code.

The comment diet (R6.70) rewrites thousands of comment lines in files where a
dropped character is a bug. The check strips comments from both revisions and
compares what is left, so it has to tell a comment from a `//` inside a string.
"""
import importlib.util
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "comment-only-check.py"
spec = importlib.util.spec_from_file_location("comment_only_check", SCRIPT)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def test_comments_go_and_code_stays():
    src = ('// a header\n'
           'int a = 1;   // trailing\n'
           '\n'
           '/* a block\n   over two lines */\n'
           'int b = a /* inline */ + 2;\n')
    assert check.code_lines(src) == ["int a = 1;", "int b = a   + 2;"]


def test_a_marker_inside_a_literal_is_code():
    src = ('const char *u = "http://host/*path*/";  // the url\n'
           "char c = '\"';  // a quote\n"
           'const char *e = "a \\" // not a comment";\n')
    assert check.code_lines(src) == ['const char *u = "http://host/*path*/";',
                                     "char c = '\"';",
                                     'const char *e = "a \\" // not a comment";']


def test_a_changed_token_is_seen_and_a_changed_comment_is_not():
    before = "int n = 0;  // counts rows\nreturn n;\n"
    assert check.code_lines(before) == check.code_lines("// rows\nint n = 0;\nreturn n;  // done\n")
    assert check.code_lines(before) != check.code_lines("int n = 1;  // counts rows\nreturn n;\n")
    # a comment that held code: taking the markers off is a code change
    assert check.code_lines("// n++;\nreturn n;\n") != check.code_lines("n++;\nreturn n;\n")


def test_the_command_line_names_a_file_whose_code_differs(tmp_path):
    def git(*a):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *a],
                       cwd=tmp_path, check=True, capture_output=True)
    git("init", "-q")
    (tmp_path / "a.c").write_text("int a = 1;  // one\n")
    (tmp_path / "b.c").write_text("int b = 2;  // two\n")
    git("add", "a.c", "b.c")
    git("commit", "-q", "-m", "base")
    (tmp_path / "a.c").write_text("// one, reworded\nint a = 1;\n")
    (tmp_path / "b.c").write_text("int b = 3;  // two\n")
    r = subprocess.run([sys.executable, str(SCRIPT), "--base", "HEAD", "a.c", "b.c"],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "ok: a.c" in r.stdout and "CODE CHANGED: b.c" in r.stdout, r.stdout
