"""Create a tiny offline encoder, tokenizer, and VA folds for smoke testing."""

import argparse
import csv
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import (
    DistilBertConfig,
    DistilBertModel,
    PreTrainedTokenizerFast,
    XLMRobertaConfig,
    XLMRobertaModel,
)


SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]
FOLD_ROWS = {
    "full_dataset_fold1.csv": [
        (0, "calm morning", "smoke", 0.78, 0.24),
        (1, "terrible angry storm", "smoke", 0.08, 0.91),
        (2, "quiet ordinary simple room", "smoke", 0.52, 0.30),
        (3, "joyful exciting music", "smoke", 0.90, 0.83),
    ],
    "full_dataset_fold2.csv": [
        (0, "sad evening", "smoke", 0.15, 0.42),
        (1, "pleasant gentle breeze", "smoke", 0.82, 0.28),
        (2, "frightening very sudden noise", "smoke", 0.12, 0.95),
        (3, "neutral simple sentence", "smoke", 0.50, 0.48),
    ],
}


def create_tokenizer(model_dir):
    words = sorted(
        {
            word
            for rows in FOLD_ROWS.values()
            for _, text, _, _, _ in rows
            for word in text.split()
        }
    )
    vocabulary = {token: index for index, token in enumerate(SPECIAL_TOKENS + words)}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", vocabulary["[CLS]"]),
            ("[SEP]", vocabulary["[SEP]"]),
        ],
    )
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        unk_token="[UNK]",
        mask_token="[MASK]",
    )
    wrapped.save_pretrained(model_dir)
    return len(vocabulary)


def create_model(model_dir, vocab_size, hidden_size=32):
    torch.manual_seed(42)
    config = XLMRobertaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=hidden_size * 2,
        max_position_embeddings=64,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    XLMRobertaModel(config).save_pretrained(model_dir)


def create_distilbert_model(model_dir, vocab_size):
    torch.manual_seed(42)
    config = DistilBertConfig(
        vocab_size=vocab_size,
        dim=32,
        hidden_dim=64,
        n_layers=1,
        n_heads=4,
        max_position_embeddings=64,
        dropout=0.0,
        attention_dropout=0.0,
        seq_classif_dropout=0.0,
        pad_token_id=0,
    )
    DistilBertModel(config).save_pretrained(model_dir)


def create_data(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in FOLD_ROWS.items():
        with open(data_dir / filename, "w", newline="", encoding="utf-8") as output_file:
            writer = csv.writer(
                output_file,
                delimiter="\t",
                quoting=csv.QUOTE_NONE,
                escapechar="\\",
            )
            writer.writerow(["index", "text", "dataset_of_origin", "valence", "arousal"])
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="artifacts/gaze_strategy_smoke/assets")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    model_dir = output_root / "tiny_encoder"
    distilbert_dir = output_root / "tiny_distilbert"
    xlm_roberta_large_dir = output_root / "tiny_xlm_roberta_large"
    data_dir = output_root / "data"
    model_dir.mkdir(parents=True, exist_ok=True)
    distilbert_dir.mkdir(parents=True, exist_ok=True)
    xlm_roberta_large_dir.mkdir(parents=True, exist_ok=True)
    vocab_size = create_tokenizer(model_dir)
    create_tokenizer(distilbert_dir)
    create_tokenizer(xlm_roberta_large_dir)
    create_model(model_dir, vocab_size)
    create_distilbert_model(distilbert_dir, vocab_size)
    create_model(xlm_roberta_large_dir, vocab_size, hidden_size=48)
    create_data(data_dir)
    manifest = {
        "model_dir": str(model_dir.resolve()),
        "distilbert_model_dir": str(distilbert_dir.resolve()),
        "xlm_roberta_large_model_dir": str(xlm_roberta_large_dir.resolve()),
        "data_dir": str(data_dir.resolve()),
        "fold_rows": {name: len(rows) for name, rows in FOLD_ROWS.items()},
    }
    with open(output_root / "manifest.json", "w", encoding="utf-8") as output_file:
        json.dump(manifest, output_file, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
