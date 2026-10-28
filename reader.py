import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

MODEL_NAME = "deepset/roberta-base-squad2"


class SpanReader:
    """Finds the best answer span to a question inside each of several passages."""

    def __init__(self, model_name=MODEL_NAME, device="cpu", max_answer_tokens=30, max_length=384):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.max_answer_tokens = max_answer_tokens
        self.max_length = max_length

    @torch.no_grad()
    def read(self, question, passages):
        """One dict per passage: the answer text, its score, and the score of 'no answer'."""
        enc = self.tokenizer(
            [question] * len(passages),
            passages,
            truncation="only_second",
            max_length=self.max_length,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")
        sequence_ids = [enc.sequence_ids(i) for i in range(len(passages))]
        outputs = self.model(**{k: v.to(self.device) for k, v in enc.items()})
        start_all = outputs.start_logits.float().cpu()
        end_all = outputs.end_logits.float().cpu()

        results = []
        for i, passage in enumerate(passages):
            # Allow the first token (it stands for "no answer") and the passage tokens only
            allowed = torch.tensor([s == 1 for s in sequence_ids[i]])
            allowed[0] = True
            start_p = torch.softmax(start_all[i].masked_fill(~allowed, float("-inf")), dim=0)
            end_p = torch.softmax(end_all[i].masked_fill(~allowed, float("-inf")), dim=0)

            null_score = float(start_p[0] * end_p[0])
            grid = start_p[:, None] * end_p[None, :]
            grid = torch.triu(grid)  # the end must not come before the start
            grid = torch.tril(grid, diagonal=self.max_answer_tokens - 1)
            grid[0, :] = 0
            grid[:, 0] = 0

            flat = int(grid.argmax())
            start, end = divmod(flat, grid.shape[1])
            score = float(grid[start, end])
            if score > 0:
                first = int(offsets[i][start][0])
                last = int(offsets[i][end][1])
                answer = passage[first:last].strip()
            else:
                answer = ""
            results.append({"answer": answer, "score": score, "null_score": null_score})
        return results