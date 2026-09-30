"""Multi-task training entry point: L_total = L_CRF + lambda_wsd * L_WSD.

Trains the encoder + role scorer + WSD head + CRF jointly against
``SRLExample``s built from a CoNLL-U corpus (via
``src.data_pipeline.hdtb_parser.build_examples_from_conllu``), using AdamW
with differential learning rates (backbone vs. BiLSTM/role-scorer/CRF/WSD
heads) and a linear-warmup/cosine-decay schedule. Validation reports
span-level Macro-F1 via ``seqeval``, with early stopping.

Usage:
    python scripts/train.py --train data/processed/train.conllu \
        --val data/processed/val.conllu
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyiwn
import torch
import torch.nn as nn
import yaml
from seqeval.metrics import classification_report, f1_score
from transformers import get_cosine_schedule_with_warmup

_REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(_REPO_ROOT))

from src.data_pipeline.dataset import LABEL_PAD_ID, SRLCollator, SRLDataset, SRLExample
from src.data_pipeline.hdtb_parser import build_examples_from_conllu
from src.data_pipeline.morphology_extractor import MorphologyExtractor
from src.inference.viterbi_decoder import decode, subword_labels_to_word_labels
from src.models.crf_layer import LinearChainCRF
from src.models.encoder import SRLEncoder, load_model_config
from src.models.role_scorer import RoleScorer, pool_predicate_hidden_state
from src.models.wsd_head import WSDHead
from src.ontology.gloss_cache import GlossEmbeddingCache
from src.ontology.hypernym_engine import HypernymEngine
from src.ontology.indowordnet_client import IndoWordNetClient
from src.ontology.mask_generator import MaskGenerator

_MODEL_CONFIG_PATH = _REPO_ROOT / "config" / "model_config.yaml"
_TRAINING_CONFIG_PATH = _REPO_ROOT / "config" / "training_config.yaml"


class SRLModel(nn.Module):
    """Bundles the encoder, role scorer, WSD head, and CRF as one module."""

    def __init__(self, model_config: Dict, num_labels: int, gloss_dim: int) -> None:
        super().__init__()
        self.encoder = SRLEncoder(model_config)
        self.role_scorer = RoleScorer(
            hidden_dim=self.encoder.output_dim,
            mlp_hidden_dim=model_config["role_scorer"]["mlp_hidden_dim"],
            num_labels=num_labels,
            dropout=model_config["role_scorer"]["dropout"],
        )
        self.wsd_head = WSDHead(
            hidden_dim=self.encoder.output_dim,
            gloss_dim=gloss_dim,
            projection_dim=model_config["wsd_head"]["projection_dim"],
        )
        self.crf = LinearChainCRF(num_labels=num_labels)

    def encode_and_score(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = self.encoder(
            batch["input_ids"],
            batch["attention_mask"],
            batch["case_marker_ids"],
            batch["clitic_ids"],
            batch["predicate_mask"],
        )
        neural_emissions = self.role_scorer(hidden_states, batch["predicate_mask"])
        predicate_hidden = pool_predicate_hidden_state(
            hidden_states, batch["predicate_mask"]
        )
        return hidden_states, neural_emissions, predicate_hidden


def build_optimizer(model: SRLModel, training_config: Dict) -> torch.optim.AdamW:
    opt_cfg = training_config["optimizer"]
    backbone_params = list(model.encoder.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": opt_cfg["backbone_lr"]},
            {"params": head_params, "lr": opt_cfg["head_lr"]},
        ],
        weight_decay=opt_cfg["weight_decay"],
        eps=opt_cfg["adam_epsilon"],
    )


def iterate_batches(
    dataset: SRLDataset, collator: SRLCollator, batch_size: int, shuffle: bool
) -> List[Tuple[List[SRLExample], Dict[str, object]]]:
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)
    batches = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        examples = [dataset.examples[i] for i in batch_indices]
        collated = collator([dataset[i] for i in batch_indices])
        batches.append((examples, collated))
    return batches


def build_batch_ontology_mask(
    examples: List[SRLExample],
    word_ids_batch: List[List[Optional[int]]],
    seq_len: int,
    mask_generator: MaskGenerator,
    label_list: List[str],
) -> torch.Tensor:
    batch_size = len(examples)
    mask = torch.zeros(batch_size, seq_len, len(label_list))
    for b, example in enumerate(examples):
        lemmas = example.lemmas or example.tokens
        case_markers = _case_markers_for(example)
        word_level_mask = mask_generator.generate(
            lemmas, example.predicate_lemma or lemmas[example.predicate_index], case_markers
        )
        for i, word_id in enumerate(word_ids_batch[b]):
            if word_id is not None:
                mask[b, i] = word_level_mask[word_id]
    return mask


_MORPHOLOGY_EXTRACTOR = MorphologyExtractor()


def _case_markers_for(example: SRLExample) -> List[Optional[str]]:
    return [m.case_marker for m in _MORPHOLOGY_EXTRACTOR.extract(example.tokens)]


def build_wsd_targets(
    examples: List[SRLExample], gloss_cache: GlossEmbeddingCache, max_candidates: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    all_candidates, all_embeddings, gold_indices = [], [], []
    max_k = 1
    for example in examples:
        lemma = example.predicate_lemma or example.tokens[example.predicate_index]
        candidates, embeddings = gloss_cache.embed_candidates(
            lemma, pyiwn.PosTag.VERB, max_candidates=max_candidates
        )
        all_candidates.append(candidates)
        all_embeddings.append(embeddings)
        max_k = max(max_k, len(candidates))

        gold_index = LABEL_PAD_ID
        if example.predicate_synset_id is not None:
            for i, candidate in enumerate(candidates):
                if candidate.synset_id() == example.predicate_synset_id:
                    gold_index = i
                    break
        gold_indices.append(gold_index)

    gloss_dim = gloss_cache.embedding_dim
    batch_size = len(examples)
    padded_embeddings = torch.zeros(batch_size, max_k, gloss_dim)
    candidate_mask = torch.zeros(batch_size, max_k)
    for b, embeddings in enumerate(all_embeddings):
        k = embeddings.shape[0]
        if k > 0:
            padded_embeddings[b, :k] = embeddings
            candidate_mask[b, :k] = 1

    return padded_embeddings, candidate_mask, torch.tensor(gold_indices, dtype=torch.long)


def compute_loss(
    model: SRLModel,
    batch: Dict[str, torch.Tensor],
    examples: List[SRLExample],
    mask_generator: MaskGenerator,
    gloss_cache: GlossEmbeddingCache,
    label_list: List[str],
    max_candidates: int,
    lambda_wsd: float,
) -> torch.Tensor:
    _, neural_emissions, predicate_hidden = model.encode_and_score(batch)

    ontology_mask = build_batch_ontology_mask(
        examples, batch["word_ids"], batch["input_ids"].shape[1], mask_generator, label_list
    )

    tags = batch["label_ids"].clone()
    tags[tags == LABEL_PAD_ID] = 0  # CRF requires valid indices at masked positions
    crf_loss = model.crf(
        neural_emissions, tags, batch["attention_mask"].bool(), ontology_mask=ontology_mask
    )

    gloss_embeddings, candidate_mask, gold_indices = build_wsd_targets(
        examples, gloss_cache, max_candidates
    )
    _, wsd_logits = model.wsd_head(predicate_hidden, gloss_embeddings, candidate_mask)
    has_gold = gold_indices != LABEL_PAD_ID
    if has_gold.any():
        wsd_loss = nn.functional.cross_entropy(
            wsd_logits[has_gold], gold_indices[has_gold], reduction="mean"
        )
    else:
        wsd_loss = torch.tensor(0.0, device=neural_emissions.device)

    return crf_loss + lambda_wsd * wsd_loss


@torch.no_grad()
def evaluate(
    model: SRLModel,
    dataset: SRLDataset,
    collator: SRLCollator,
    mask_generator: MaskGenerator,
    label_list: List[str],
) -> Dict[str, object]:
    model.eval()
    gold_sequences, pred_sequences = [], []

    for idx, example in enumerate(dataset.examples):
        item = dataset[idx]
        batch = collator([item])
        _, neural_emissions, _ = model.encode_and_score(batch)

        lemmas = example.lemmas or example.tokens
        case_markers = _case_markers_for(example)
        word_level_mask = mask_generator.generate(
            lemmas, example.predicate_lemma or lemmas[example.predicate_index], case_markers
        )
        word_ids = batch["word_ids"][0]
        subword_mask = torch.zeros(1, neural_emissions.shape[1], len(label_list))
        for i, word_id in enumerate(word_ids):
            if word_id is not None:
                subword_mask[0, i] = word_level_mask[word_id]

        phi = model.crf.combine_emissions(neural_emissions, subword_mask)[0]
        subword_labels = decode(
            phi,
            model.crf.transitions,
            model.crf.start_transitions,
            model.crf.end_transitions,
            label_list,
            mask=batch["attention_mask"][0],
        )
        word_labels = subword_labels_to_word_labels(
            subword_labels, word_ids, len(example.tokens)
        )

        gold_sequences.append(example.bio_tags)
        pred_sequences.append(word_labels)

    model.train()
    macro_f1 = f1_score(gold_sequences, pred_sequences, average="macro")
    report = classification_report(gold_sequences, pred_sequences, zero_division=0)
    return {"macro_f1": macro_f1, "report": report}


def train(
    train_path: str,
    val_path: Optional[str] = None,
    num_epochs: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
) -> Dict[str, object]:
    with open(_MODEL_CONFIG_PATH, encoding="utf-8") as fh:
        model_config = yaml.safe_load(fh)
    with open(_TRAINING_CONFIG_PATH, encoding="utf-8") as fh:
        training_config = yaml.safe_load(fh)

    torch.manual_seed(model_config["seed"])
    random.seed(model_config["seed"])

    label_list = model_config["crf"]["label_set"]
    label_to_id = {label: i for i, label in enumerate(label_list)}

    train_examples = build_examples_from_conllu(Path(train_path).read_text(encoding="utf-8"))
    train_dataset = SRLDataset(train_examples, label_to_id=label_to_id)
    collator = SRLCollator(tokenizer=train_dataset.tokenizer)

    val_dataset = None
    if val_path is not None:
        val_examples = build_examples_from_conllu(Path(val_path).read_text(encoding="utf-8"))
        val_dataset = SRLDataset(val_examples, label_to_id=label_to_id, tokenizer=train_dataset.tokenizer)

    iwn_client = IndoWordNetClient()
    hypernym_engine = HypernymEngine(client=iwn_client)
    mask_generator = MaskGenerator(hypernym_engine=hypernym_engine)
    gloss_cache = GlossEmbeddingCache(iwn_client=iwn_client)

    model = SRLModel(model_config, len(label_list), gloss_cache.embedding_dim)
    optimizer = build_optimizer(model, training_config)

    batch_size = training_config["data"]["batch_size"]
    num_epochs = num_epochs or training_config["loop"]["num_epochs"]
    steps_per_epoch = max(1, (len(train_dataset) + batch_size - 1) // batch_size)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(training_config["schedule"]["warmup_ratio"] * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    lambda_wsd = training_config["loss"]["lambda_wsd"]
    max_candidates = model_config["wsd_head"]["max_candidate_senses"]
    max_grad_norm = training_config["optimizer"]["max_grad_norm"]

    best_macro_f1 = -1.0
    patience = training_config["loop"]["early_stopping_patience"]
    epochs_without_improvement = 0
    history = []

    for epoch in range(num_epochs):
        model.train()
        batches = iterate_batches(train_dataset, collator, batch_size, shuffle=True)
        epoch_loss = 0.0
        for examples, batch in batches:
            optimizer.zero_grad()
            loss = compute_loss(
                model, batch, examples, mask_generator, gloss_cache, label_list,
                max_candidates, lambda_wsd,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(batches)
        epoch_result = {"epoch": epoch, "train_loss": avg_loss}

        if val_dataset is not None:
            metrics = evaluate(model, val_dataset, collator, mask_generator, label_list)
            epoch_result["val_macro_f1"] = metrics["macro_f1"]
            if metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]
                epochs_without_improvement = 0
                if checkpoint_dir is not None:
                    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), Path(checkpoint_dir) / "best_model.pt")
            else:
                epochs_without_improvement += 1

        history.append(epoch_result)

        if val_dataset is not None and epochs_without_improvement >= patience:
            break

    return {"model": model, "history": history, "best_val_macro_f1": best_macro_f1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Path to training CoNLL-U file")
    parser.add_argument("--val", default=None, help="Path to validation CoNLL-U file")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args()

    result = train(args.train, args.val, args.epochs, args.checkpoint_dir)
    for epoch_result in result["history"]:
        print(epoch_result)


if __name__ == "__main__":
    main()
