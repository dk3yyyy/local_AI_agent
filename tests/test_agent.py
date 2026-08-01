import unittest

from langchain_core.documents import Document

from agent import (
    CITATION_VALIDATION_MESSAGE,
    NO_MATCH_MESSAGE,
    answer_question,
)


class FakeModel:
    def __init__(
        self, response: str = "The crust receives strong praise [review-1]."
    ) -> None:
        self.response = response
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class SequenceModel:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


class FakeStore:
    def __init__(self, results: list[tuple[Document, float]]) -> None:
        self.results = results

    def get(self, *, include: list[str]) -> dict[str, list[str]]:
        return {"ids": [str(index) for index in range(len(self.results))]}

    def similarity_search_with_score(
        self, query: str, *, k: int
    ) -> list[tuple[Document, float]]:
        return self.results[:k]


class AnswerQuestionTest(unittest.TestCase):
    def test_builds_grounded_prompt_and_returns_cited_sources(self) -> None:
        document = Document(
            page_content="Great crust Crisp and flavorful.",
            metadata={"rating": 5, "date": "2024-01-10"},
            id="review-1",
        )
        model = FakeModel()

        result = answer_question(
            "What do guests say about the crust?",
            vector_store=FakeStore([(document, 0.12)]),
            model=model,
        )

        self.assertEqual(result.answer, "The crust receives strong praise [1].")
        self.assertEqual(result.retrieved_source_ids, ("review-1",))
        self.assertFalse(result.abstained)
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0].citation_number, 1)
        self.assertEqual(result.sources[0].document.id, "review-1")
        self.assertIn("[1]", model.prompts[0])
        self.assertIn("Source ID: review-1", model.prompts[0])
        self.assertIn("Great crust Crisp and flavorful.", model.prompts[0])
        self.assertIn("only the supplied reviews", model.prompts[0])
        self.assertIn(
            "answer every part that at least one supplied review supports",
            model.prompts[0],
        )
        self.assertIn("Mixed or incomplete evidence is", model.prompts[0])
        self.assertIn("not a reason to abstain", model.prompts[0])
        self.assertIn("Never use INSUFFICIENT_EVIDENCE as prose", model.prompts[0])

    def test_repairs_an_uncited_answer_once_and_preserves_diagnostics(self) -> None:
        document = Document(
            page_content="The crust was perfectly crispy.",
            metadata={"source_id": "review-1"},
            id="review-1",
        )
        model = SequenceModel(
            "Guests praise the crispy crust.",
            "Guests praise the crispy crust [1].",
        )

        result = answer_question(
            "What do guests say about the crust?",
            vector_store=FakeStore([(document, 0.5)]),
            model=model,
        )

        self.assertEqual(result.answer, "Guests praise the crispy crust [1].")
        self.assertEqual(result.raw_response, "Guests praise the crispy crust.")
        self.assertEqual(result.repair_response, "Guests praise the crispy crust [1].")
        self.assertEqual(result.initial_failure_reason, "missing_citations")
        self.assertIsNone(result.failure_reason)
        self.assertTrue(result.repair_attempted)
        self.assertEqual(len(model.prompts), 2)
        self.assertIn("Guests praise the crispy crust.", model.prompts[1])
        self.assertIn("rewrite it once", model.prompts[1])

    def test_failed_repair_records_the_final_structured_reason(self) -> None:
        document = Document(
            page_content="The crust was perfectly crispy.",
            metadata={"source_id": "review-1"},
            id="review-1",
        )
        model = SequenceModel("Unsupported [2].", "Still unsupported [3].")

        result = answer_question(
            "What do guests say about the crust?",
            vector_store=FakeStore([(document, 0.5)]),
            model=model,
        )

        self.assertEqual(result.answer, CITATION_VALIDATION_MESSAGE)
        self.assertEqual(result.initial_failure_reason, "out_of_range_citation")
        self.assertEqual(result.failure_reason, "out_of_range_citation")
        self.assertEqual(result.raw_response, "Unsupported [2].")
        self.assertEqual(result.repair_response, "Still unsupported [3].")
        self.assertTrue(result.repair_attempted)
        self.assertEqual(len(model.prompts), 2)

    def test_repair_can_return_a_clean_abstention_without_a_third_attempt(self) -> None:
        document = Document(
            page_content="A review about pizza crust.",
            metadata={"source_id": "review-1"},
            id="review-1",
        )
        model = SequenceModel(
            "Parking is available.",
            "  INSUFFICIENT_EVIDENCE  ",
        )

        result = answer_question(
            "Is parking available?",
            vector_store=FakeStore([(document, 0.5)]),
            model=model,
        )

        self.assertEqual(result.answer, NO_MATCH_MESSAGE)
        self.assertEqual(result.sources, ())
        self.assertTrue(result.abstained)
        self.assertEqual(result.initial_failure_reason, "missing_citations")
        self.assertEqual(result.failure_reason, "clean_abstention")
        self.assertTrue(result.repair_attempted)
        self.assertEqual(len(model.prompts), 2)

    def test_accepts_numeric_citation_for_the_matching_retrieved_review(self) -> None:
        document = Document(
            page_content="Phenomenal crust Crispy and flavorful.",
            metadata={"source_id": "review-long-content-hash"},
            id="review-long-content-hash",
        )

        result = answer_question(
            "What did the review say about the phenomenal crust?",
            vector_store=FakeStore([(document, 0.08)]),
            model=FakeModel("The crust was crispy and flavorful [1]."),
        )

        self.assertEqual(result.answer, "The crust was crispy and flavorful [1].")
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0].document.id, "review-long-content-hash")

    def test_numeric_citations_resolve_to_the_correct_multiple_reviews(self) -> None:
        first = Document(
            page_content="The crust was crispy.",
            metadata={"source_id": "review-a"},
            id="review-a",
        )
        second = Document(
            page_content="The crust was thin.",
            metadata={"source_id": "review-b"},
            id="review-b",
        )

        result = answer_question(
            "What did guests say about the crust?",
            vector_store=FakeStore([(first, 0.08), (second, 0.1)]),
            model=FakeModel(
                "One guest called it thin [2]; another called it crispy [1]."
            ),
        )

        self.assertEqual(
            result.answer,
            "One guest called it thin [1]; another called it crispy [2].",
        )
        self.assertEqual(
            [source.document.id for source in result.sources],
            ["review-b", "review-a"],
        )

    def test_rejects_numeric_citation_outside_the_retrieved_set(self) -> None:
        document = Document(
            page_content="Phenomenal crust Crispy and flavorful.",
            metadata={"source_id": "review-long-content-hash"},
            id="review-long-content-hash",
        )

        result = answer_question(
            "What did the review say about the phenomenal crust?",
            vector_store=FakeStore([(document, 0.08)]),
            model=FakeModel("Unsupported statement [2]."),
        )

        self.assertEqual(result.answer, CITATION_VALIDATION_MESSAGE)
        self.assertEqual(result.sources, ())

    def test_rejects_citations_that_were_not_retrieved(self) -> None:
        document = Document(
            page_content="Great crust Crisp and flavorful.",
            metadata={"source_id": "review-1"},
            id="review-1",
        )

        result = answer_question(
            "What do guests say about the crust?",
            vector_store=FakeStore([(document, 0.12)]),
            model=FakeModel("Unsupported claim [review-999]."),
        )

        self.assertEqual(result.answer, CITATION_VALIDATION_MESSAGE)
        self.assertEqual(result.sources, ())

    def test_rejects_uncited_model_answers(self) -> None:
        document = Document(
            page_content="Great crust Crisp and flavorful.",
            metadata={"source_id": "review-1"},
            id="review-1",
        )

        result = answer_question(
            "What do guests say about the crust?",
            vector_store=FakeStore([(document, 0.12)]),
            model=FakeModel("The crust receives strong praise."),
        )

        self.assertEqual(result.answer, CITATION_VALIDATION_MESSAGE)
        self.assertEqual(result.sources, ())

    def test_model_can_abstain_when_retrieved_reviews_are_insufficient(self) -> None:
        document = Document(
            page_content="A review about pizza crust.",
            metadata={"source_id": "review-1"},
            id="review-1",
        )

        model = FakeModel("\n  INSUFFICIENT_EVIDENCE  \n")
        result = answer_question(
            "Is parking available?",
            vector_store=FakeStore([(document, 0.5)]),
            model=model,
        )

        self.assertEqual(result.answer, NO_MATCH_MESSAGE)
        self.assertEqual(result.sources, ())
        self.assertEqual(result.retrieved_source_ids, ("review-1",))
        self.assertTrue(result.abstained)
        self.assertEqual(result.failure_reason, "clean_abstention")
        self.assertFalse(result.repair_attempted)
        self.assertEqual(len(model.prompts), 1)

    def test_accepts_cited_answer_with_standalone_insufficient_token_line(
        self,
    ) -> None:
        document = Document(
            page_content="The crust was perfectly crispy.",
            metadata={"source_id": "review-1"},
            id="review-1",
        )

        responses = {
            "followed": "Guests praise the crispy crust [1].\n\nINSUFFICIENT_EVIDENCE",
            "preceded": "INSUFFICIENT_EVIDENCE\nGuests praise the crispy crust [1].",
            "multiline_whitespace": (
                "\n  INSUFFICIENT_EVIDENCE  \n\nGuests praise the crispy crust [1].\n"
            ),
        }

        for position, response in responses.items():
            with self.subTest(position=position):
                result = answer_question(
                    "What do guests say about the crust?",
                    vector_store=FakeStore([(document, 0.5)]),
                    model=FakeModel(response),
                )

                self.assertEqual(result.answer, "Guests praise the crispy crust [1].")
                self.assertNotIn("INSUFFICIENT_EVIDENCE", result.answer)
                self.assertEqual(len(result.sources), 1)
                self.assertEqual(result.sources[0].document.id, "review-1")
                self.assertEqual(result.retrieved_source_ids, ("review-1",))
                self.assertFalse(result.abstained)

    def test_rejects_insufficient_evidence_token_embedded_in_prose(self) -> None:
        document = Document(
            page_content="The crust was perfectly crispy.",
            metadata={"source_id": "review-1"},
            id="review-1",
        )

        model = SequenceModel(
            "The raw marker INSUFFICIENT_EVIDENCE must not be shown [1].",
            "This response must never be requested [1].",
        )
        result = answer_question(
            "What do guests say about the crust?",
            vector_store=FakeStore([(document, 0.5)]),
            model=model,
        )

        self.assertEqual(result.answer, CITATION_VALIDATION_MESSAGE)
        self.assertNotIn("INSUFFICIENT_EVIDENCE", result.answer)
        self.assertEqual(result.sources, ())
        self.assertFalse(result.abstained)
        self.assertEqual(result.failure_reason, "embedded_control_token")
        self.assertFalse(result.repair_attempted)
        self.assertEqual(len(model.prompts), 1)

    def test_rejects_empty_remainder_without_attempting_repair(self) -> None:
        document = Document(
            page_content="The crust was perfectly crispy.",
            metadata={"source_id": "review-1"},
            id="review-1",
        )
        model = SequenceModel(
            "INSUFFICIENT_EVIDENCE\nINSUFFICIENT_EVIDENCE",
            "This response must never be requested [1].",
        )

        result = answer_question(
            "What do guests say about the crust?",
            vector_store=FakeStore([(document, 0.5)]),
            model=model,
        )

        self.assertEqual(result.answer, CITATION_VALIDATION_MESSAGE)
        self.assertEqual(result.sources, ())
        self.assertEqual(result.failure_reason, "invalid_remainder")
        self.assertFalse(result.repair_attempted)
        self.assertEqual(len(model.prompts), 1)

    def test_rejects_uncited_answer_after_removing_control_token_line(self) -> None:
        document = Document(
            page_content="The crust was perfectly crispy.",
            metadata={"source_id": "review-1"},
            id="review-1",
        )

        result = answer_question(
            "What do guests say about the crust?",
            vector_store=FakeStore([(document, 0.5)]),
            model=FakeModel("Guests praise the crust.\nINSUFFICIENT_EVIDENCE"),
        )

        self.assertEqual(result.answer, CITATION_VALIDATION_MESSAGE)
        self.assertEqual(result.sources, ())
        self.assertFalse(result.abstained)
        self.assertEqual(result.failure_reason, "missing_citations")

    def test_does_not_call_model_when_filters_match_no_reviews(self) -> None:
        model = FakeModel()

        result = answer_question(
            "Anything recent?",
            vector_store=FakeStore([]),
            model=model,
        )

        self.assertEqual(result.answer, NO_MATCH_MESSAGE)
        self.assertEqual(result.sources, ())
        self.assertEqual(result.retrieved_source_ids, ())
        self.assertFalse(result.abstained)
        self.assertEqual(model.prompts, [])
        self.assertEqual(result.failure_reason, "empty_retrieval")


if __name__ == "__main__":
    unittest.main()
