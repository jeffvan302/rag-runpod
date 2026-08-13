import unittest

from src.embedding_contract import (
    QUERY_PREFIX,
    format_embedding_text,
    normalize_output_dimensions,
)


class EmbeddingContractTests(unittest.TestCase):
    def test_accepts_mrl_dimensions_for_qwen_4b(self):
        self.assertEqual(
            normalize_output_dimensions("Qwen/Qwen3-Embedding-4B", 1024),
            1024,
        )
        self.assertEqual(
            normalize_output_dimensions("Qwen/Qwen3-Embedding-4B", 2560),
            2560,
        )

    def test_rejects_unconfigured_dimensions(self):
        with self.assertRaisesRegex(ValueError, "supports configured output dimensions"):
            normalize_output_dimensions("Qwen/Qwen3-Embedding-4B", 1280)

    def test_adds_query_instruction_once(self):
        formatted = format_embedding_text("What is the notice period?", "query")
        self.assertEqual(formatted, f"{QUERY_PREFIX}What is the notice period?")
        self.assertEqual(format_embedding_text(formatted, "query"), formatted)

    def test_document_text_has_no_query_instruction(self):
        self.assertEqual(format_embedding_text("  Contract terms  ", "document"), "Contract terms")


if __name__ == "__main__":
    unittest.main()
