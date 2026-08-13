import unittest

from src.chunking import split_markdown_for_chunks, utf16_offsets


class ChunkingTests(unittest.TestCase):
    def test_splits_on_page_boundary_near_limit(self):
        markdown = "A" * 385 + "\n## Page 2\n" + "B" * 100
        chunks = split_markdown_for_chunks(markdown, 100, 10)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].end_offset, 395)
        self.assertTrue(chunks[1].text.startswith("## Page 2"))

    def test_empty_document_gets_stable_placeholder(self):
        chunks = split_markdown_for_chunks("", 900, 120)

        self.assertEqual(chunks[0].text, "(empty document)")
        self.assertEqual(chunks[0].start_offset, 0)
        self.assertEqual(chunks[0].end_offset, 1)

    def test_utf16_offsets_match_javascript_string_offsets(self):
        offsets = utf16_offsets("a😀b")

        self.assertEqual(offsets, [0, 1, 3, 4])


if __name__ == "__main__":
    unittest.main()
