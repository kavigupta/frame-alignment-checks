"""
Delete this file once all other tests pass. It will fail if any template tags are used in the code.
It will fail if it exists, because it includes the template tag "_OF_PACKAGE" which should not be used.
"""

import subprocess
import unittest

from parameterized import parameterized


def allFilesTrackedByGit():
    return (
        subprocess.check_output(
            ["git", "ls-files"], universal_newlines=True
        )
        .strip()
        .split("\n")
    )


class TestTemplateTagsNoLongerExist(unittest.TestCase):

    @parameterized.expand(
        [(file_path,) for file_path in allFilesTrackedByGit()]
    )
    def test_template_tags_no_longer_exist(self, file):
        """
        Go through all files in this directory and ensure that no template tags are used.
        """
        try:
            with open(file) as f:
                contents = f.read()
        except UnicodeDecodeError:
            return
        self.assertNotIn("_OF_PACKAGE", contents)

    @parameterized.expand([(file_path,) for file_path in allFilesTrackedByGit()])
    def test_no_template_tags_in_names(self, file):
        """
        Go through all files in this directory and ensure that no template tags are used in names.
        """
        self.assertNotIn("_OF_PACKAGE", file)
