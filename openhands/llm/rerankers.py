from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BartForConditionalGeneration


class BaseReRanker(ABC):
    @abstractmethod
    def __call__(self, texts: List[str]) -> List[float]:
        pass


class CosineSimilarityReRanker(BaseReRanker):
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def __call__(self, texts: List[str]) -> List[float]:
        # Compute embeddings
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors='pt'
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)

        # Compute pairwise similarities
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        similarities = torch.mm(normalized_embeddings, normalized_embeddings.t())

        # Average similarity for each text with all others
        avg_similarities = (similarities.sum(dim=1) - 1) / (len(texts) - 1)
        return avg_similarities.tolist()


class BARTScoreReRanker(BaseReRanker):
    def __init__(self, model_name: str):
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, texts: List[str]) -> List[float]:
        scores = []
        for hyp in texts:
            score = 0
            for ref in texts:
                if hyp != ref:
                    inputs = self.tokenizer(ref, return_tensors='pt', truncation=True)
                    with torch.no_grad():
                        score += self.model(**inputs).logits.mean().item()
            scores.append(score / (len(texts) - 1))
        return scores


class CombReranker(BaseReRanker):
    def __init__(self, rerankers: List[BaseReRanker]):
        self.rerankers = rerankers

    def __call__(self, texts: List[str]) -> List[float]:
        all_scores = []
        for reranker in self.rerankers:
            scores = reranker(texts)
            all_scores.append(scores)

        # Average scores from all rerankers
        return [sum(scores) / len(scores) for scores in zip(*all_scores)]
