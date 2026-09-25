"""Packaging contract for the bundle that carries the EditApart agent preset.

Harness ≥ 0.1.7 does not read user-preset DIRECTORIES
(`$DSH_HOME/.agent-presets/<id>/{preset.yml,agent.cordis.yml}`) any more: a preset
arrives only as an `@deepseek-ai/dsh-agent-preset` declaration row supplied by a
bundle patch. This test is the guard against drifting back to the retired shape.

The failure mode it exists for is silent, which is why it is worth a test: a row
specifier resolves against the **profile** directory (`ctx.baseUrl`), so a
relative specifier such as `./plugins/edit-apart.mjs` looks for
`<profile>/plugins/edit-apart.mjs`. The preset still mounts — it just arrives
without its own plugin or its method skill, and every tool call fails with
"unknown tool" instead of anything pointing at packaging.

Needs PyYAML (`pip install pyyaml`; listed in requirements.txt).
"""
from __future__ import annotations

import json
import os
import re
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PATCH = os.path.join(ROOT, "cordis.patch.yml")
MANIFEST = os.path.join(ROOT, "package.json")
PACKAGE_NAME = "dsh-edit-apart"
PRESET_ID = "ai-video-editor"

try:
    import yaml
except ImportError:  # pragma: no cover - environment problem, not a code path
    yaml = None


class _Lenient(yaml.SafeLoader if yaml else object):
    """The harness's YAML dialect turns `!!js` scalars into expression nodes.

    A plain parser rejects the tag outright (`tag:yaml.org,2002:js`), so a
    packaging test has to read it the way the loader does: keep the expression
    text. Anything the test asserts about an expression is asserted on that text.
    """


if yaml:
    _Lenient.add_constructor(
        "tag:yaml.org,2002:js", lambda loader, node: loader.construct_scalar(node))


def _load_patch() -> dict:
    with open(PATCH, encoding="utf-8") as fh:
        return yaml.load(fh.read(), Loader=_Lenient)


def _declaration(patch: dict) -> dict:
    inserts = patch[0]["insert"]
    assert len(inserts) == 1
    return inserts[0]


def _composition(patch: dict) -> list[dict]:
    return _declaration(patch)["config"]["plugins"]


def _tracked_files() -> list[str]:
    """Every file git knows about (the packaging surface a user receives)."""
    import subprocess
    out = subprocess.run(["git", "-C", ROOT, "ls-files"], capture_output=True, text=True)
    if out.returncode != 0:  # not a checkout (release tarball): walk instead
        found = []
        for base, dirs, files in os.walk(ROOT):
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "node_modules")]
            found += [os.path.relpath(os.path.join(base, f), ROOT) for f in files]
        return found
    return [line for line in out.stdout.splitlines() if line.strip()]


@unittest.skipUnless(yaml is not None, "PyYAML not installed (pip install pyyaml)")
class TestBundlePackaging(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.load(open(MANIFEST, encoding="utf-8"))
        cls.patch = _load_patch()

    # ── the bundle manifest ──────────────────────────────────────────────────

    def test_manifest_declares_a_bundle_patch(self):
        self.assertEqual(self.manifest["name"], PACKAGE_NAME)
        self.assertTrue(self.manifest["private"] is False,
                        "a store-listed bundle must be publishable")
        dsh = self.manifest.get("dsh", {})
        self.assertIn("bundle", dsh, "harness reads dsh.bundle from package.json")
        patch = dsh["bundle"]["patch"]
        self.assertEqual(patch, ["./cordis.patch.yml"])
        self.assertTrue(os.path.isfile(PATCH), "declared patch file is missing")

    # ── the declaration row ─────────────────────────────────────────────────

    def test_patch_declares_exactly_one_agent_preset(self):
        decl = _declaration(self.patch)
        self.assertEqual(decl["name"], "@deepseek-ai/dsh-agent-preset")
        cfg = decl["config"]
        self.assertEqual(cfg["id"], PRESET_ID)
        self.assertEqual(cfg["name"], "EditApart (AI taste editor)")
        self.assertIsInstance(cfg.get("order"), int)
        self.assertTrue(cfg.get("description"), "the picker shows the description")
        self.assertIsInstance(cfg["plugins"], list)

    def test_composition_row_ids_are_unique(self):
        ids = [row.get("id") for row in _composition(self.patch)]
        self.assertEqual(len(ids), len(set(ids)), f"duplicate row ids: {ids}")

    # ── the defect class: specifiers that cannot travel with the bundle ─────

    def test_no_relative_row_specifiers(self):
        """A relative row specifier resolves against the PROFILE directory."""
        relative = [row["name"] for row in _composition(self.patch)
                    if isinstance(row.get("name"), str)
                    and (row["name"].startswith(".") or row["name"].startswith("/"))]
        self.assertEqual(relative, [],
                         f"these rows would look for {relative} inside the profile")

    def test_this_packages_own_plugin_is_named_by_package_subpath(self):
        row = next(r for r in _composition(self.patch) if r["id"] == "edit-apart")
        self.assertEqual(row["name"], f"{PACKAGE_NAME}/plugins/edit-apart.mjs")
        target = os.path.join(ROOT, "plugins", "edit-apart.mjs")
        self.assertTrue(os.path.isfile(target), "the named plugin is not in the package")

    def test_skills_root_resolves_through_the_profile(self):
        """The preset's own method skill must travel with the bundle.

        `baseUrl` is the profile directory, so the installed package is reachable
        as `<profile>/node_modules/<pkg>/skills`.
        """
        row = next(r for r in _composition(self.patch) if r["id"] == "skill-filesystem")
        roots = row["config"]["customSkillDirs"]
        self.assertEqual(len(roots), 1)
        expr = roots[0]
        self.assertIn("baseUrl", expr, "must resolve relative to the profile, not an absolute path")
        self.assertIn(f"node_modules/{PACKAGE_NAME}/skills/", expr)
        self.assertTrue(os.path.isfile(os.path.join(ROOT, "skills", "edit-apart", "SKILL.md")),
                        "the skill the expression points at is not in the package")

    def test_no_machine_local_paths_anywhere(self):
        """Nothing a user receives may carry a path from a development machine."""
        offenders = []
        patterns = (re.compile(r"/home/[A-Za-z0-9._-]+/", re.I),
                    re.compile(r"/Users/[A-Za-z0-9._-]+/"),
                    re.compile(r"file:///", re.I),
                    re.compile(r"[A-Za-z]:\\\\?(?:Users|Documents|dev)\\\\", re.I))
        for rel in _tracked_files():
            if rel.endswith((".png", ".jpg", ".mp4", ".gguf", ".pdf")):
                continue
            path = os.path.join(ROOT, rel)
            try:
                text = open(path, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            for pattern in patterns:
                for match in pattern.finditer(text):
                    offenders.append(f"{rel}: {match.group(0)}")
        self.assertEqual(offenders, [], "machine-local paths in shipped files:\n" +
                         "\n".join(sorted(set(offenders))))

    def test_retired_user_preset_layout_is_gone(self):
        """The 0.1.6-era directory mechanism is dead; shipping its files again
        would imply an install path the harness no longer honours."""
        self.assertFalse(os.path.exists(os.path.join(ROOT, "agent.cordis.yml")))
        self.assertFalse(os.path.exists(os.path.join(ROOT, "preset.yml")))
        for rel in _tracked_files():
            self.assertNotIn(".agent-presets/", rel)

    def test_readme_documents_the_bundle_install(self):
        text = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
        self.assertIn("dsh plugin", text,
                      "the README must show the install command a user actually runs")
        self.assertNotIn(".agent-presets/ai-video-editor", text,
                         "the README still advertises the retired directory mechanism")


if __name__ == "__main__":
    unittest.main(verbosity=2)
