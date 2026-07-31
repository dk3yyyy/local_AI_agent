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
        self.assertIn("[review-1]", model.prompts[0])
        self.assertIn("Great crust Crisp and flavorful.", model.prompts[0])
        self.assertIn("only the supplied reviews", model.prompts[0])

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
            model=FakeModel("INSUFFICIENT_EVIDENCE"),
        )

        self.assertEqual(result.answer, NO_MATCH_MESSAGE)
        self.assertEqual(result.sources, ())
        self.assertEqual(result.retrieved_source_ids, ("review-1",))
        self.assertTrue(result.abstained)

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


if __name__ == "__main__":
    unittest.main()
