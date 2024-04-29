# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Lightplane"
copyright = "2024, Meta AI Research"
author = "Meta AI Research"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = []

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "scratch", "README.md"]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]
autosummary_generate = True

# -- Configurations for plugins ------------
napoleon_google_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_special_with_doc = True
napoleon_numpy_docstring = False
# napoleon_use_param = False
napoleon_use_rtype = False
autodoc_inherit_docstrings = False
autodoc_member_order = "bysource"
napoleon_custom_sections = [("Returns", "params_style")]

# myst settings
myst_heading_anchors = 3

# make sure we can find Lightplane
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

master_doc = "index"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_theme = 'piccolo_theme'
html_static_path = ["_static"]


######
# import files into sphinx:
######

LIGHTPLANE_GIT_ROOT = "https://github.com/facebookresearch/lightplane"

def _process_introduction_lines(lines):
    lines_out = []
    for line in lines:
        if line.startswith(
            "# Lightplane"
        ):  # rename the main title to "Introduction"
            line = "# Introduction"
        # replace important paths
        line = (
            line.replace(
                "docs/_static/assets/lightplane_splash.png",
                "_static/assets/lightplane_splash.png",
            )
            .replace(
                "(./examples/README.md)",
                f"(./examples.md)",
            )
            .replace(
                "(LICENSE)",
                f"({LIGHTPLANE_GIT_ROOT}/LICENSE)",
            )
        )
        lines_out.append(line)
    return lines_out


def _process_examples_lines(lines):
    lines_out = []
    for line in lines:
        # replace all markdown links to local files with references to github
        md_link = line.find("](./")
        if md_link != -1:
            md_link_end = line[md_link:].rfind(")")
            link_with_brackets = line[md_link:md_link+md_link_end+1]
            link = link_with_brackets[2:-1]
            line = line.replace(
                link_with_brackets,
                f"](https://github.com/facebookresearch/lightplane/examples/{link})"
            )
        lines_out.append(line)
    return lines_out


def _copy_file_to_sphinx_docs(input_file, output_file):
    input_lines = input_file.read_text().split("\n")
    if str(output_file).endswith("introduction.md"):
        output_lines = _process_introduction_lines(input_lines)
    elif str(output_file).endswith("examples.md"):
        output_lines = _process_examples_lines(input_lines)
    with open(output_file, "w") as f:
        f.write("\n".join(output_lines))
        

def _copy_files_to_sphinx_docs():
    import pathlib
    lightplane_root = pathlib.Path(__file__).parent.resolve().parent
    
    # README.md -> docs/introduction.md
    readme_path = lightplane_root / "README.md"
    introduction_path = lightplane_root / "docs" / "introduction.md"
    _copy_file_to_sphinx_docs(readme_path, introduction_path)
    
    # examples/README.md -> docs/examples.md
    examples_readme_path = lightplane_root / "examples" / "README.md"
    docs_examples_readme_path = lightplane_root / "docs" / "examples.md"
    _copy_file_to_sphinx_docs(examples_readme_path, docs_examples_readme_path)

#### copy existing .md files from the main repo to sphinx docs
_copy_files_to_sphinx_docs()
####
