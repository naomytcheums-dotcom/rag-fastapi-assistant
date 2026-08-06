# Judge calibration set

`human_annotations.json` is currently empty. To calibrate the LLM judge
(`src/llm_judge.py`) against human judgment:

1. Once `ANTHROPIC_API_KEY` has credit, run `python src/generation.py "<question>"`
   for ~20 questions spread across `data/test_set.json` (mix of easy/medium/hard).
2. For each one, manually read the answer against its retrieved context and
   score it 1-5 on faithfulness and relevance yourself.
3. Add one entry per example to `human_annotations.json`:

```json
{
  "id": "q003",
  "question": "How do you declare a POST route that accepts a JSON request body?",
  "context": "<the context block generation.py actually sent to the model>",
  "answer": "<the model's actual answer>",
  "human_faithfulness": 5,
  "human_relevance": 5
}
```

4. Run `python src/llm_judge.py` -- it scores the same examples and reports
   agreement with your human scores. The project targets >90% agreement
   (judge within 1 point of the human score) before trusting the judge to
   score the rest of the test set unsupervised.

This can't be filled in without real API credit and real manual review --
see `AUDIT.md` for why it wasn't fabricated.
