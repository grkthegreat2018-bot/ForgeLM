"""Tests for new agent tools and sidebar collapse."""
import pytest
from pathlib import Path

from forge_gui.api.agent_tools import ToolSandbox


class TestNewFileTools:
    @pytest.fixture()
    def workspace(self, tmp_path):
        (tmp_path / "test.py").write_text("print('hello')\n", encoding="utf-8")
        (tmp_path / "subdir").mkdir()
        return tmp_path

    def test_rename_file(self, workspace):
        sb = ToolSandbox(workspace)
        rec = sb.execute("rename_file", {"old_path": "test.py", "new_path": "renamed.py"})
        assert rec["ok"]
        assert not workspace.joinpath("test.py").exists()
        assert workspace.joinpath("renamed.py").exists()

    def test_rename_file_not_found(self, workspace):
        sb = ToolSandbox(workspace)
        rec = sb.execute("rename_file", {"old_path": "nonexistent.py", "new_path": "x.py"})
        assert not rec["ok"]

    def test_rename_file_dest_exists(self, workspace):
        (workspace / "other.py").write_text("# other", encoding="utf-8")
        sb = ToolSandbox(workspace)
        rec = sb.execute("rename_file", {"old_path": "test.py", "new_path": "other.py"})
        assert not rec["ok"]
        assert "exists" in rec["result"]["error"]

    def test_create_dir(self, workspace):
        sb = ToolSandbox(workspace)
        rec = sb.execute("create_dir", {"path": "new/nested/dir"})
        assert rec["ok"]
        assert workspace.joinpath("new/nested/dir").is_dir()

    def test_file_info(self, workspace):
        sb = ToolSandbox(workspace)
        rec = sb.execute("file_info", {"path": "test.py"})
        assert rec["ok"]
        info = rec["result"]
        assert info["size"] > 0
        assert info["lines"] == 1
        assert info["ext"] == ".py"

    def test_dir_tree(self, workspace):
        sb = ToolSandbox(workspace)
        rec = sb.execute("dir_tree", {"path": "."})
        assert rec["ok"]
        tree = rec["result"]["tree"]
        assert tree["type"] == "dir"
        assert "children" in tree

    def test_dir_tree_max_depth(self, workspace):
        (workspace / "a/b/c").mkdir(parents=True)
        sb = ToolSandbox(workspace)
        rec = sb.execute("dir_tree", {"path": ".", "max_depth": 1})
        assert rec["ok"]


class TestCodeIntelligence:
    @pytest.fixture()
    def workspace(self, tmp_path):
        (tmp_path / "code.py").write_text(
            "def foo():\n    pass\n\nclass Bar:\n    def method(self):\n        pass\n"
            "# TODO: fix this\n# FIXME: broken\n", encoding="utf-8")
        (tmp_path / "other.py").write_text(
            "x = foo()\ny = Bar()\n", encoding="utf-8")
        return tmp_path

    def test_find_references(self, workspace):
        sb = ToolSandbox(workspace)
        rec = sb.execute("find_references", {"symbol": "foo"})
        assert rec["ok"]
        matches = rec["result"]["matches"]
        # should find the definition and the reference
        assert len(matches) >= 2

    def test_find_definitions(self, workspace):
        sb = ToolSandbox(workspace)
        rec = sb.execute("find_definitions", {"path": "code.py"})
        assert rec["ok"]
        defs = rec["result"]["definitions"]
        names = [d["name"] for d in defs]
        assert "foo" in names
        assert "Bar" in names

    def test_find_definitions_filter(self, workspace):
        sb = ToolSandbox(workspace)
        rec = sb.execute("find_definitions", {"path": "code.py", "symbol": "Bar"})
        assert rec["ok"]
        defs = rec["result"]["definitions"]
        assert all(d["name"] == "Bar" for d in defs)

    def test_find_todos(self, workspace):
        sb = ToolSandbox(workspace)
        rec = sb.execute("find_todos", {})
        assert rec["ok"]
        todos = rec["result"]["todos"]
        tags = [t["tag"] for t in todos]
        assert "TODO" in tags
        assert "FIXME" in tags

    def test_line_count_file(self, workspace):
        sb = ToolSandbox(workspace)
        rec = sb.execute("line_count", {"path": "code.py"})
        assert rec["ok"]
        lc = rec["result"]
        assert lc["total"] > 0
        assert lc["code"] > 0
        assert lc["comment"] >= 2  # TODO + FIXME lines

    def test_syntax_check_valid(self, workspace):
        sb = ToolSandbox(workspace)
        rec = sb.execute("syntax_check", {"path": "code.py"})
        assert rec["ok"]
        assert rec["result"]["valid"] is True

    def test_syntax_check_invalid(self, workspace):
        (workspace / "bad.py").write_text("def foo(:\n    pass\n", encoding="utf-8")
        sb = ToolSandbox(workspace)
        rec = sb.execute("syntax_check", {"path": "bad.py"})
        assert rec["ok"]
        assert rec["result"]["valid"] is False
        assert len(rec["result"]["errors"]) > 0

    def test_syntax_check_non_python(self, workspace):
        (workspace / "file.js").write_text("console.log('hi')", encoding="utf-8")
        sb = ToolSandbox(workspace)
        rec = sb.execute("syntax_check", {"path": "file.js"})
        assert not rec["ok"]


class TestProjectSearchReplace:
    @pytest.fixture()
    def workspace(self, tmp_path):
        (tmp_path / "a.py").write_text("old_name = 1\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("x = old_name + 1\n", encoding="utf-8")
        return tmp_path

    def test_project_replace(self, workspace):
        sb = ToolSandbox(workspace)
        rec = sb.execute("project_search_replace", {
            "old_text": "old_name", "new_text": "new_name"})
        assert rec["ok"]
        assert rec["result"]["total_files"] == 2
        assert "new_name" in workspace.joinpath("a.py").read_text()
        assert "new_name" in workspace.joinpath("b.py").read_text()

    def test_project_replace_no_match(self, workspace):
        sb = ToolSandbox(workspace)
        rec = sb.execute("project_search_replace", {
            "old_text": "nonexistent", "new_text": "whatever"})
        assert rec["ok"]
        assert rec["result"]["total_files"] == 0


class TestUndoEdit:
    @pytest.fixture()
    def workspace(self, tmp_path):
        (tmp_path / "test.py").write_text("original\n", encoding="utf-8")
        return tmp_path

    def test_undo_write_file(self, workspace):
        sb = ToolSandbox(workspace)
        # write new content
        sb.execute("write_file", {"path": "test.py", "content": "modified\n"})
        assert "modified" in workspace.joinpath("test.py").read_text()
        # undo
        rec = sb.execute("undo_edit", {"path": "test.py"})
        assert rec["ok"]
        assert "original" in workspace.joinpath("test.py").read_text()

    def test_undo_search_replace(self, workspace):
        sb = ToolSandbox(workspace)
        sb.execute("search_replace", {
            "path": "test.py", "old_text": "original", "new_text": "replaced"})
        assert "replaced" in workspace.joinpath("test.py").read_text()
        rec = sb.execute("undo_edit", {"path": "test.py"})
        assert rec["ok"]
        assert "original" in workspace.joinpath("test.py").read_text()

    def test_undo_no_history(self, workspace):
        sb = ToolSandbox(workspace)
        rec = sb.execute("undo_edit", {"path": "test.py"})
        assert not rec["ok"]
        assert "no undo history" in rec["result"]["error"]


class TestGitTools:
    @pytest.fixture()
    def workspace(self, tmp_path):
        (tmp_path / "file.py").write_text("# code", encoding="utf-8")
        return tmp_path

    def test_git_branch_list(self, workspace):
        sb = ToolSandbox(workspace)
        rec = sb.execute("git_branch", {"action": "list"})
        # may not have git, but shouldn't crash
        assert "exit_code" in rec["result"] or "error" in rec["result"]

    def test_git_stash_list(self, workspace):
        sb = ToolSandbox(workspace)
        rec = sb.execute("git_stash", {"action": "list"})
        assert "exit_code" in rec["result"] or "error" in rec["result"]


class TestRunTests:
    def test_run_tests(self, tmp_path):
        (tmp_path / "test_simple.py").write_text(
            "def test_pass():\n    assert True\n", encoding="utf-8")
        sb = ToolSandbox(tmp_path)
        rec = sb.execute("run_tests", {"path": "test_simple.py"})
        # should run and pass (if pytest available)
        assert "exit_code" in rec["result"] or "error" in rec["result"]


class TestSidebarCollapse:
    def test_sidebar_collapse_toggle(self):
        from PySide6.QtWidgets import QApplication
        import sys
        app = QApplication.instance() or QApplication(sys.argv)
        from forge_gui.widgets.sidebar import NavSidebar
        sidebar = NavSidebar([
            ("__section__", "Test"),
            ("Page1", "◎"),
            ("Page2", "✉"),
        ])
        assert not sidebar.is_collapsed
        assert sidebar.width() == 232
        sidebar.toggle_collapse()
        assert sidebar.is_collapsed
        assert sidebar.width() == 48
        sidebar.toggle_collapse()
        assert not sidebar.is_collapsed
        assert sidebar.width() == 232

    def test_sidebar_set_collapsed(self):
        from PySide6.QtWidgets import QApplication
        import sys
        app = QApplication.instance() or QApplication(sys.argv)
        from forge_gui.widgets.sidebar import NavSidebar
        sidebar = NavSidebar([("Page1", "◎")])
        sidebar.set_collapsed(True)
        assert sidebar.is_collapsed
        sidebar.set_collapsed(False)
        assert not sidebar.is_collapsed

    def test_sidebar_collapsed_signal(self):
        from PySide6.QtWidgets import QApplication
        import sys
        app = QApplication.instance() or QApplication(sys.argv)
        from forge_gui.widgets.sidebar import NavSidebar
        sidebar = NavSidebar([("Page1", "◎")])
        signals = []
        sidebar.collapsed_changed.connect(lambda c: signals.append(c))
        sidebar.toggle_collapse()
        assert signals == [True]
        sidebar.toggle_collapse()
        assert signals == [True, False]


class TestToolHarnessNewTools:
    def test_all_new_tools_in_defs(self):
        from forge_gui.api.tool_harness import ToolHarness
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            harness = ToolHarness(td)
            defs = harness.tool_defs()
            names = {d["function"]["name"] for d in defs}
            for expected in ("rename_file", "create_dir", "file_info", "dir_tree",
                             "find_references", "find_definitions", "find_todos",
                             "line_count", "syntax_check", "project_search_replace",
                             "git_branch", "git_stash", "run_tests", "undo_edit"):
                assert expected in names, f"missing tool: {expected}"

    def test_new_tools_hidden_in_read_only(self):
        from forge_gui.api.tool_harness import ToolHarness
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            harness = ToolHarness(td, read_only=True)
            defs = harness.tool_defs()
            names = {d["function"]["name"] for d in defs}
            # side-effect tools should be hidden
            assert "rename_file" not in names
            assert "project_search_replace" not in names
            assert "undo_edit" not in names
            assert "run_tests" not in names
            # read-only tools should still be present
            assert "file_info" in names
            assert "dir_tree" in names
            assert "find_references" in names
            assert "find_todos" in names
            assert "line_count" in names
            assert "syntax_check" in names
