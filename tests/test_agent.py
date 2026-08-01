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
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return next(self.responses)


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

        result = answer_question(
            "Is parking available?",
            vector_store=FakeStore([(document, 0.5)]),
            model=FakeModel("\n  INSUFFICIENT_EVIDENCE  \n"),
        )

        self.assertEqual(result.answer, NO_MATCH_MESSAGE)
        self.assertEqual(result.sources, ())
        self.assertEqual(result.retrieved_source_ids, ("review-1",))
        self.assertTrue(result.abstained)

    def test_retries_false_abstention_and_accepts_repaired_cited_answer(self) -> None:
        document = Document(
            page_content="The white pizza uses ricotta, mozzarella, and garlic.",
            metadata={"source_id": "review-1"},
            id="review-1",
        )
        model = SequenceModel(
            [
                "INSUFFICIENT_EVIDENCE",
                "Guests describe ricotta, mozzarella, and garlic [1].",
            ]
        )

        result = answer_question(
            "Which ingredients are mentioned in the white pizza?",
            vector_store=FakeStore([(document, 0.1)]),
            model=model,
        )

        self.assertEqual(
            result.answer,
            "Guests describe ricotta, mozzarella, and garlic [1].",
        )
        self.assertFalse(result.abstained)
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.raw_response, "INSUFFICIENT_EVIDENCE")
        self.assertEqual(
            result.repair_response,
            "Guests describe ricotta, mozzarella, and garlic [1].",
        )
        self.assertEqual(result.initial_failure_reason, "clean_abstention")
        self.assertIsNone(result.failure_reason)
        self.assertTrue(result.repair_attempted)
        self.assertEqual(len(model.prompts), 2)
        self.assertIn("at least one supplied review", model.prompts[1])

    def test_retries_invalid_citations_and_accepts_repaired_answer(self) -> None:
        document = Document(
            page_content="Delivery was late and the pizza arrived cold.",
            metadata={"source_id": "review-1"},
            id="review-1",
        )
        model = SequenceModel(
            [
                "Delivery was late [9].",
                "Guests report late delivery and cold pizza [1].",
            ]
        )

        result = answer_question(
            "What delivery problems did guests report?",
            vector_store=FakeStore([(document, 0.1)]),
            model=model,
        )

        self.assertEqual(
            result.answer,
            "Guests report late delivery and cold pizza [1].",
        )
        self.assertFalse(result.abstained)
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.raw_response, "Delivery was late [9].")
        self.assertEqual(
            result.repair_response,
            "Guests report late delivery and cold pizza [1].",
        )
        self.assertEqual(result.initial_failure_reason, "out_of_range_citation")
        self.assertIsNone(result.failure_reason)
        self.assertTrue(result.repair_attempted)
        self.assertEqual(len(model.prompts), 2)
        self.assertIn("could not be accepted", model.prompts[1])

    def test_stops_after_one_repair_when_evidence_is_still_insufficient(self) -> None:
        document = Document(
            page_content="A review about pizza crust.",
            metadata={"source_id": "review-1"},
            id="review-1",
        )
        model = SequenceModel(["INSUFFICIENT_EVIDENCE", "INSUFFICIENT_EVIDENCE"])

        result = answer_question(
            "Is parking available?",
            vector_store=FakeStore([(document, 0.5)]),
            model=model,
        )

        self.assertEqual(result.answer, NO_MATCH_MESSAGE)
        self.assertTrue(result.abstained)
        self.assertEqual(result.sources, ())
        self.assertEqual(result.initial_failure_reason, "clean_abstention")
        self.assertEqual(result.failure_reason, "clean_abstention")
        self.assertTrue(result.repair_attempted)
        self.assertEqual(len(model.prompts), 2)

    def test_failed_repair_preserves_initial_valid_abstention(self) -> None:
        document = Document(
            page_content="A review about pizza crust.",
            metadata={"source_id": "review-1", "title": "Pizza"},
        )
        model = SequenceModel(
            ["INSUFFICIENT_EVIDENCE", "Unsupported parking claim [9]."]
        )
        result = answer_question(
            "Is parking available?",
            vector_store=FakeStore([(document, 0.5)]),
            model=model,
        )

        self.assertEqual(result.answer, NO_MATCH_MESSAGE)
        self.assertTrue(result.abstained)
        self.assertEqual(result.sources, ())
        self.assertEqual(result.raw_response, "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result.repair_response, "Unsupported parking claim [9].")
        self.assertEqual(result.initial_failure_reason, "clean_abstention")
        self.assertEqual(result.failure_reason, "out_of_range_citation")
        self.assertTrue(result.repair_attempted)
        self.assertEqual(len(model.prompts), 2)

    def test_unsupported_benchmark_topics_remain_abstentions(self) -> None:
        document = Document(
            page_content="Guests discuss pizza crust and toppings only.",
            metadata={"source_id": "review-1", "title": "Pizza"},
        )
        unsupported_questions = (
            "Is parking available?",
            "Is the restaurant wheelchair accessible?",
            "Does the restaurant accept reservations?",
            "Is Wi-Fi available?",
            "How much is the delivery fee?",
        )

        for question in unsupported_questions:
            with self.subTest(question=question):
                model = SequenceModel(
                    ["INSUFFICIENT_EVIDENCE", "INSUFFICIENT_EVIDENCE"]
                )
                result = answer_question(
                    question,
                    vector_store=FakeStore([(document, 0.5)]),
                    model=model,
                )

                self.assertEqual(result.answer, NO_MATCH_MESSAGE)
                self.assertTrue(result.abstained)
                self.assertEqual(result.sources, ())
                self.assertEqual(result.initial_failure_reason, "clean_abstention")
                self.assertEqual(result.failure_reason, "clean_abstention")
                self.assertEqual(len(model.prompts), 2)

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
            [
                "The raw marker INSUFFICIENT_EVIDENCE must not be shown [1].",
                "Guests praise the crispy crust [1].",
            ]
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
        self.assertTrue(result.repair_attempted)

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
        self.assertEqual(result.failure_reason, "empty_retrieval")
        self.assertEqual(model.prompts, [])


if __name__ == "__main__":
    unittest.main()
