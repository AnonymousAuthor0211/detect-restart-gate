#!/usr/bin/env python3
"""
fetch_datasets.py - Download and preprocess datasets for HyperNet-XL experiments

This script downloads the required datasets (SST-2, QNLI, MNLI, XSum, HotpotQA, SAMSum, COCO, Flickr30k, AudioCaps, Clotho, AIME, AIME 2024, Minerva MATH, OmniMATH, LiveMathBench, AMC, etc.)
and performs any necessary preprocessing.
"""

import argparse
import json
import logging
import subprocess
import tempfile
import shutil
import urllib.request
from pathlib import Path

import datasets
from datasets import Dataset, DatasetDict, load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_glue_tasks(output_dir: Path) -> None:
    """Download GLUE tasks (SST-2, QNLI, MNLI, MRPC, QQP, CoLA, RTE, WNLI)."""
    glue_tasks = ["sst2", "qnli", "mnli", "mrpc", "qqp", "cola", "rte", "wnli"]

    for task in glue_tasks:
        logger.info(f"Downloading GLUE task: {task}")

        try:
            dataset = load_dataset("glue", task)

            # Save to disk
            task_dir = output_dir / task
            task_dir.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(task_dir))

            logger.info(f"Saved {task} to {task_dir}")

            # Print dataset info
            logger.info(f"{task} dataset info:")
            for split, data in dataset.items():
                logger.info(f"  {split}: {len(data)} examples")

        except Exception as e:
            logger.error(f"Failed to download {task}: {e}")


def download_pawsx(output_dir: Path) -> None:
    """Download PAWS-X dataset (multilingual paraphrase detection)."""
    logger.info("Downloading PAWS-X dataset")

    try:
        # Load English version of PAWS-X
        dataset = load_dataset("paws-x", "en")

        # Save to disk
        pawsx_dir = output_dir / "pawsx"
        pawsx_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(pawsx_dir))

        logger.info(f"Saved PAWS-X to {pawsx_dir}")

        # Print dataset info
        logger.info("PAWS-X dataset info:")
        for split, data in dataset.items():
            logger.info(f"  {split}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download PAWS-X: {e}")


def download_boolq(output_dir: Path) -> None:
    """Download BoolQ dataset."""
    logger.info("Downloading BoolQ dataset")

    try:
        dataset = load_dataset("boolq")

        # Save to disk
        boolq_dir = output_dir / "boolq"
        boolq_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(boolq_dir))

        logger.info(f"Saved BoolQ to {boolq_dir}")

        # Print dataset info
        logger.info("BoolQ dataset info:")
        for split, data in dataset.items():
            logger.info(f"  {split}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download BoolQ: {e}")


def download_agnews(output_dir: Path) -> None:
    """Download AG News dataset."""
    logger.info("Downloading AG News dataset")

    try:
        dataset = load_dataset("ag_news")

        # Save to disk
        agnews_dir = output_dir / "agnews"
        agnews_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(agnews_dir))

        logger.info(f"Saved AG News to {agnews_dir}")

        # Print dataset info
        logger.info("AG News dataset info:")
        for split, data in dataset.items():
            logger.info(f"  {split}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download AG News: {e}")


def download_xsum(output_dir: Path) -> None:
    """Download XSum dataset."""
    logger.info("Downloading XSum dataset")

    try:
        dataset = load_dataset("xsum")

        # Save to disk
        xsum_dir = output_dir / "xsum"
        xsum_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(xsum_dir))

        logger.info(f"Saved XSum to {xsum_dir}")

        # Print dataset info
        logger.info("XSum dataset info:")
        for split, data in dataset.items():
            logger.info(f"  {split}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download XSum: {e}")


def download_hotpotqa(output_dir: Path) -> None:
    """Download HotpotQA dataset."""
    logger.info("Downloading HotpotQA dataset")

    try:
        dataset = load_dataset("hotpot_qa", "fullwiki")

        # Save to disk
        hotpot_dir = output_dir / "hotpot"
        hotpot_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(hotpot_dir))

        logger.info(f"Saved HotpotQA to {hotpot_dir}")

        # Print dataset info
        logger.info("HotpotQA dataset info:")
        for split, data in dataset.items():
            logger.info(f"  {split}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download HotpotQA: {e}")


def download_samsum(output_dir: Path) -> None:
    """Download SAMSum dataset."""
    logger.info("Downloading SAMSum dataset")

    try:
        dataset = load_dataset("samsum")

        # Save to disk
        samsum_dir = output_dir / "samsum"
        samsum_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(samsum_dir))

        logger.info(f"Saved SAMSum to {samsum_dir}")

        # Print dataset info
        logger.info("SAMSum dataset info:")
        for split, data in dataset.items():
            logger.info(f"  {split}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download SAMSum: {e}")


def download_coco_captions(output_dir: Path) -> None:
    """Download COCO2017 captions dataset"""
    logger.info("Downloading COCO2017 captions dataset")

    try:
        # Load the dataset
        dataset = load_dataset("yerevann/coco-karpathy")

        
        train_dataset = dataset["train"].shuffle(seed=42)
        val_dataset = dataset["validation"] 

        # Create new dataset dict with sampled training data
        sampled_dataset = datasets.DatasetDict(
            {"train": train_dataset, "validation": val_dataset}
        )

        # Save to disk
        coco_dir = output_dir / "coco_captions"
        coco_dir.mkdir(parents=True, exist_ok=True)
        sampled_dataset.save_to_disk(str(coco_dir))

        logger.info(f"Saved COCO captions to {coco_dir}")

        # Print dataset info
        logger.info("COCO captions dataset info:")
        for split, data in sampled_dataset.items():
            logger.info(f"  {split}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download COCO captions: {e}")


def download_flickr30k(output_dir: Path) -> None:
    """Download Flickr30k dataset"""
    logger.info("Downloading Flickr30k dataset")

    try:
        # Load the dataset
        dataset = load_dataset("nlphuji/flickr30k")

        
        train_dataset = (
            dataset["test"].shuffle(seed=42)
        )  # Flickr30k only has test split

        # Create train/val split from the sampled data
        split_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)

        # Create new dataset dict
        sampled_dataset = datasets.DatasetDict(
            {"train": split_dataset["train"], "validation": split_dataset["test"]}
        )

        # Save to disk
        flickr_dir = output_dir / "flickr30k"
        flickr_dir.mkdir(parents=True, exist_ok=True)
        sampled_dataset.save_to_disk(str(flickr_dir))

        logger.info(f"Saved Flickr30k to {flickr_dir}")

        # Print dataset info
        logger.info("Flickr30k dataset info:")
        for split, data in sampled_dataset.items():
            logger.info(f"  {split}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download Flickr30k: {e}")

def download_vizwiz_vqa(output_dir: Path) -> None:
    """Download VizWiz-VQA dataset with proper train/val splits."""
    logger.info("Downloading VizWiz-VQA dataset")

    try:
        # Load full dataset with splits
        dataset = load_dataset("lmms-lab/VizWiz-VQA")

        

        # The test set will be used to train sicne it has large samples 8k
        dataset_clean = DatasetDict({
            "train": dataset["test"],
            "validation": dataset["val"],
        })

        # Save to disk
        vizwiz_dir = output_dir / "vizwiz_vqa"
        vizwiz_dir.mkdir(parents=True, exist_ok=True)
        dataset_clean.save_to_disk(str(vizwiz_dir))

        logger.info(f"Saved VizWiz-VQA to {vizwiz_dir}")

        # Log stats
        logger.info("VizWiz-VQA dataset info:")
        for split, data in dataset_clean.items():
            logger.info(f"  {split}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download VizWiz-VQA: {e}")

def download_audiocaps(output_dir: Path) -> None:
    """Download full AudioCaps dataset (train, validation, test)."""
    logger.info("Downloading full AudioCaps dataset")

    try:
        # Load full dataset
        dataset = load_dataset("OpenSound/AudioCaps")

        # Optional: explicitly create DatasetDict to ensure type consistency
        full_dataset = DatasetDict(
            {
                "train": dataset["train"],
                "validation": dataset["validation"],
                "test": dataset["test"],
            }
        )

        # Save to disk
        audiocaps_dir = output_dir / "audiocaps"
        audiocaps_dir.mkdir(parents=True, exist_ok=True)
        full_dataset.save_to_disk(str(audiocaps_dir))

        logger.info(f"Saved AudioCaps to {audiocaps_dir}")
        logger.info("AudioCaps dataset info:")
        for split, data in full_dataset.items():
            logger.info(f"  {split}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download AudioCaps: {e}")


def download_clotho(output_dir: Path) -> None:
    """Download Clotho-v2 dataset with benchmark splits (validation, evaluation)."""
    logger.info("Downloading Clotho-v2 benchmark splits")

    try:
        # Load official dataset with Hugging Face tag
        dataset = load_dataset("CLAPv2/Clotho")

        # Print available split keys for safety
        logger.info(f"Available splits: {list(dataset.keys())}")

        # Use validation and evaluation splits from original benchmark
        clotho_dataset = datasets.DatasetDict({
            "train": dataset["train"],
            "validation": dataset["validation"],
            "test": dataset["test"],
        })

        # Save to disk
        clotho_dir = output_dir / "clotho"
        clotho_dir.mkdir(parents=True, exist_ok=True)
        clotho_dataset.save_to_disk(str(clotho_dir))

        logger.info(f"Saved Clotho splits to {clotho_dir}")

        # Print dataset info
        logger.info("Clotho dataset info:")
        for split, data in clotho_dataset.items():
            logger.info(f"  {split}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download Clotho: {e}")

def download_esc50(output_dir: Path) -> None:
    """Download ESC-50 dataset with 5-fold split as per benchmark."""
    logger.info("Downloading ESC-50 dataset")

    try:
        dataset = load_dataset("ashraq/esc50")

        # Organize into a dict of 5 folds
        folds = {}
        for fold in range(1, 6):
            fold_dataset = dataset["train"].filter(lambda ex: ex["fold"] == fold)
            folds[f"fold_{fold}"] = fold_dataset

        esc50_dataset = datasets.DatasetDict(folds)

        # Save to disk
        esc50_dir = output_dir / "esc50"
        esc50_dir.mkdir(parents=True, exist_ok=True)
        esc50_dataset.save_to_disk(str(esc50_dir))

        logger.info(f"Saved ESC-50 with 5-folds to {esc50_dir}")
        for name, data in esc50_dataset.items():
            logger.info(f"  {name}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download ESC-50: {e}")


def download_anli(output_dir: Path) -> None:
    """Download ANLI (Adversarial NLI) dataset."""
    logger.info("Downloading ANLI dataset")

    try:
        # Load R1 round of ANLI
        dataset = load_dataset("anli")

        # Save to disk
        anli_dir = output_dir / "anli"
        anli_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(anli_dir))

        logger.info(f"Saved ANLI to {anli_dir}")

        # Print dataset info
        logger.info("ANLI dataset info:")
        for split, data in dataset.items():
            logger.info(f"  {split}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download ANLI: {e}")


def download_winogrande(output_dir: Path) -> None:
    """Download Winogrande dataset."""
    logger.info("Downloading Winogrande dataset")

    try:
        dataset = load_dataset("winogrande", "winogrande_debiased")

        # Save to disk
        winogrande_dir = output_dir / "winogrande"
        winogrande_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(winogrande_dir))

        logger.info(f"Saved Winogrande to {winogrande_dir}")

        # Print dataset info
        logger.info("Winogrande dataset info:")
        for split, data in dataset.items():
            logger.info(f"  {split}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download Winogrande: {e}")


def download_hellaswag(output_dir: Path) -> None:
    """Download HellaSwag dataset."""
    logger.info("Downloading HellaSwag dataset")

    try:
        dataset = load_dataset("hellaswag")

        # Save to disk
        hellaswag_dir = output_dir / "hellaswag"
        hellaswag_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(hellaswag_dir))

        logger.info(f"Saved HellaSwag to {hellaswag_dir}")

        # Print dataset info
        logger.info("HellaSwag dataset info:")
        for split, data in dataset.items():
            logger.info(f"  {split}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download HellaSwag: {e}")


def download_openbookqa(output_dir: Path) -> None:
    """Download OpenBookQA dataset."""
    logger.info("Downloading OpenBookQA dataset")

    try:
        dataset = load_dataset("openbookqa", "main")

        # Save to disk
        openbookqa_dir = output_dir / "openbookqa"
        openbookqa_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(openbookqa_dir))

        logger.info(f"Saved OpenBookQA to {openbookqa_dir}")

        # Print dataset info
        logger.info("OpenBookQA dataset info:")
        for split, data in dataset.items():
            logger.info(f"  {split}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download OpenBookQA: {e}")


def download_piqa(output_dir: Path) -> None:
    """Download PIQA dataset."""
    logger.info("Downloading PIQA dataset")

    try:
        dataset = load_dataset("piqa")

        # Save to disk
        piqa_dir = output_dir / "piqa"
        piqa_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(piqa_dir))

        logger.info(f"Saved PIQA to {piqa_dir}")

        # Print dataset info
        logger.info("PIQA dataset info:")
        for split, data in dataset.items():
            logger.info(f"  {split}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download PIQA: {e}")


def download_strategyqa(output_dir: Path) -> None:
    """Download StrategyQA dataset."""
    logger.info("Downloading StrategyQA dataset")

    try:
        dataset = load_dataset("tasksource/bigbench", "strategyqa")

        # Save to disk
        strategyqa_dir = output_dir / "strategyqa"
        strategyqa_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(strategyqa_dir))

        logger.info(f"Saved StrategyQA to {strategyqa_dir}")

        # Print dataset info
        logger.info("StrategyQA dataset info:")
        for split, data in dataset.items():
            logger.info(f"  {split}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download StrategyQA: {e}")


def download_gsm8k(output_dir: Path) -> None:
    """Download GSM8K dataset."""
    logger.info("Downloading GSM8K dataset")

    try:
        dataset = load_dataset("gsm8k", "main")

        # Save to disk
        gsm8k_dir = output_dir / "gsm8k"
        gsm8k_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(gsm8k_dir))

        logger.info(f"Saved GSM8K to {gsm8k_dir}")

        # Print dataset info
        logger.info("GSM8K dataset info:")
        for split, data in dataset.items():
            logger.info(f"  {split}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download GSM8K: {e}")


def download_aime(output_dir: Path) -> None:
    """Download AIME dataset (American Invitational Mathematics Examination).

    Uses math-ai/aime25 from Hugging Face. Contains 30 problems with problem, answer, id.
    Only has a test split; we create train/validation via an 80/20 split for consistency.
    """
    logger.info("Downloading AIME dataset")

    try:
        dataset = load_dataset("math-ai/aime25")

        # AIME has only "test" split (30 problems); create train/validation for consistency
        if "test" in dataset and "train" not in dataset:
            test_data = dataset["test"]
            split_dataset = test_data.train_test_split(test_size=0.2, seed=42)
            aime_dataset = DatasetDict({
                "train": split_dataset["train"],
                "validation": split_dataset["test"],
            })
        else:
            aime_dataset = dataset

        # Save to disk
        aime_dir = output_dir / "aime"
        aime_dir.mkdir(parents=True, exist_ok=True)
        aime_dataset.save_to_disk(str(aime_dir))

        logger.info(f"Saved AIME to {aime_dir}")

        # Print dataset info
        logger.info("AIME dataset info:")
        for split, data in aime_dataset.items():
            logger.info(f"  {split}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download AIME: {e}")


def download_aime_2024(output_dir: Path) -> None:
    """Download AIME 2024 dataset (30 problems from 2024 AIME I and II).

    Uses HuggingFaceH4/aime_2024. Only has test split; we create train/validation via 80/20 split.
    """
    logger.info("Downloading AIME 2024 dataset")

    try:
        dataset = load_dataset("HuggingFaceH4/aime_2024")

        if "test" in dataset and "train" not in dataset:
            test_data = dataset["test"]
            split_dataset = test_data.train_test_split(test_size=0.2, seed=42)
            aime_dataset = DatasetDict({
                "train": split_dataset["train"],
                "validation": split_dataset["test"],
            })
        else:
            aime_dataset = dataset

        aime_dir = output_dir / "aime_2024"
        aime_dir.mkdir(parents=True, exist_ok=True)
        aime_dataset.save_to_disk(str(aime_dir))

        logger.info(f"Saved AIME 2024 to {aime_dir}")
        logger.info("AIME 2024 dataset info:")
        for split, data in aime_dataset.items():
            logger.info(f"  {split}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download AIME 2024: {e}")


def download_minerva_math(output_dir: Path) -> None:
    """Download Minerva MATH benchmark (272 challenging competition math problems).

    Uses math-ai/minervamath. Only test split; create train/validation via 80/20 split.
    """
    logger.info("Downloading Minerva MATH dataset")

    try:
        dataset = load_dataset("math-ai/minervamath")

        if "test" in dataset and "train" not in dataset:
            test_data = dataset["test"]
            split_dataset = test_data.train_test_split(test_size=0.2, seed=42)
            minerva_dataset = DatasetDict({
                "train": split_dataset["train"],
                "validation": split_dataset["test"],
            })
        else:
            minerva_dataset = dataset

        minerva_dir = output_dir / "minerva_math"
        minerva_dir.mkdir(parents=True, exist_ok=True)
        minerva_dataset.save_to_disk(str(minerva_dir))

        logger.info(f"Saved Minerva MATH to {minerva_dir}")
        logger.info("Minerva MATH dataset info:")
        for split, data in minerva_dataset.items():
            logger.info(f"  {split}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download Minerva MATH: {e}")


def download_omnimath(output_dir: Path) -> None:
    """Download Omni-MATH dataset (4,428 Olympiad-level math problems, 33 sub-domains).

    Uses KbsdJames/Omni-MATH.
    """
    logger.info("Downloading Omni-MATH dataset")

    try:
        dataset = load_dataset("KbsdJames/Omni-MATH")

        # Use as-is if train/validation/test present; else create splits from single split
        if len(dataset) == 1:
            single = list(dataset.values())[0]
            split_dataset = single.train_test_split(test_size=0.2, seed=42)
            omnimath_dataset = DatasetDict({
                "train": split_dataset["train"],
                "validation": split_dataset["test"],
            })
        else:
            omnimath_dataset = dataset

        omnimath_dir = output_dir / "omnimath"
        omnimath_dir.mkdir(parents=True, exist_ok=True)
        omnimath_dataset.save_to_disk(str(omnimath_dir))

        logger.info(f"Saved Omni-MATH to {omnimath_dir}")
        logger.info("Omni-MATH dataset info:")
        for split, data in omnimath_dataset.items():
            logger.info(f"  {split}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download Omni-MATH: {e}")


def download_livemathbench(output_dir: Path) -> None:
    """Download LiveMathBench dataset (Olympiad-level math from AoPS, contamination-resistant).

    Uses opencompass/LiveMathBench.
    """
    logger.info("Downloading LiveMathBench dataset")

    try:
        dataset = load_dataset("opencompass/LiveMathBench")

        if len(dataset) == 1:
            single = list(dataset.values())[0]
            split_dataset = single.train_test_split(test_size=0.2, seed=42)
            livemath_dataset = DatasetDict({
                "train": split_dataset["train"],
                "validation": split_dataset["test"],
            })
        else:
            livemath_dataset = dataset

        livemath_dir = output_dir / "livemathbench"
        livemath_dir.mkdir(parents=True, exist_ok=True)
        livemath_dataset.save_to_disk(str(livemath_dir))

        logger.info(f"Saved LiveMathBench to {livemath_dir}")
        logger.info("LiveMathBench dataset info:")
        for split, data in livemath_dataset.items():
            logger.info(f"  {split}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download LiveMathBench: {e}")


def download_amc(output_dir: Path) -> None:
    """Download AMC dataset (American Mathematics Competition, 40 problems from AMC 2023).

    Uses math-ai/amc23. Only test split; create train/validation via 80/20 split.
    """
    logger.info("Downloading AMC dataset")

    try:
        dataset = load_dataset("math-ai/amc23")

        if "test" in dataset and "train" not in dataset:
            test_data = dataset["test"]
            split_dataset = test_data.train_test_split(test_size=0.2, seed=42)
            amc_dataset = DatasetDict({
                "train": split_dataset["train"],
                "validation": split_dataset["test"],
            })
        else:
            amc_dataset = dataset

        amc_dir = output_dir / "amc"
        amc_dir.mkdir(parents=True, exist_ok=True)
        amc_dataset.save_to_disk(str(amc_dir))

        logger.info(f"Saved AMC to {amc_dir}")
        logger.info("AMC dataset info:")
        for split, data in amc_dataset.items():
            logger.info(f"  {split}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download AMC: {e}")


def download_math500(output_dir: Path) -> None:
    """Download MATH-500 dataset (subset of MATH dataset with 500 problems)."""
    logger.info("Downloading MATH-500 dataset")
    
    try:
        # Load MATH-500 dataset from HuggingFaceH4
        logger.info("Loading MATH-500 from HuggingFaceH4/MATH-500...")
        dataset = load_dataset("HuggingFaceH4/MATH-500", trust_remote_code=True)
        
        logger.info(f"Successfully loaded MATH-500 dataset with splits: {list(dataset.keys())}")
        
        # Verify we have data
        if dataset and len(dataset) > 0:
            total = sum(len(data) for data in dataset.values())
            logger.info(f"Dataset contains {total} total examples")
            
            if total == 0:
                raise ValueError("MATH-500 dataset is empty")
        else:
            raise ValueError("MATH-500 dataset could not be loaded properly")
        
        # Ensure we have train/val splits
        if len(dataset) == 1:
            # Single split, create train/val (80/20)
            split_data = list(dataset.values())[0]
            split_dataset = split_data.train_test_split(test_size=0.2, seed=42)
            math500_dataset = DatasetDict({
                "train": split_dataset["train"],
                "validation": split_dataset["test"],
            })
        else:
            # Multiple splits, use as-is but ensure we have train/val
            if "train" in dataset and "validation" in dataset:
                math500_dataset = DatasetDict({
                    "train": dataset["train"],
                    "validation": dataset["validation"],
                })
            elif "train" in dataset and "test" in dataset:
                math500_dataset = DatasetDict({
                    "train": dataset["train"],
                    "validation": dataset["test"],
                })
            else:
                # Use first split as train, second as val
                splits = list(dataset.keys())
                math500_dataset = DatasetDict({
                    "train": dataset[splits[0]],
                    "validation": dataset[splits[1]] if len(splits) > 1 else dataset[splits[0]],
                })
        
        # Save to disk
        math500_dir = output_dir / "math500"
        math500_dir.mkdir(parents=True, exist_ok=True)
        math500_dataset.save_to_disk(str(math500_dir))
        
        logger.info(f"Saved MATH-500 to {math500_dir}")
        
        # Print dataset info
        logger.info("MATH-500 dataset info:")
        for split, data in math500_dataset.items():
            logger.info(f"  {split}: {len(data)} examples")
    
    except Exception as e:
        logger.error(f"Failed to download MATH-500: {e}")
        raise


def download_arc(output_dir: Path) -> None:
    """Download ARC (Easy) dataset."""
    logger.info("Downloading ARC (Easy) dataset")

    try:
        dataset = load_dataset("ai2_arc", "ARC-Easy")

        # Save to disk
        arc_dir = output_dir / "arc_easy"
        arc_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(arc_dir))

        logger.info(f"Saved ARC (Easy) to {arc_dir}")

        # Print dataset info
        logger.info("ARC (Easy) dataset info:")
        for split, data in dataset.items():
            logger.info(f"  {split}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download ARC (Easy): {e}")


def download_arc_challenge(output_dir: Path) -> None:
    """Download ARC (Challenge) dataset."""
    logger.info("Downloading ARC (Challenge) dataset")

    try:
        dataset = load_dataset("ai2_arc", "ARC-Challenge")

        # Save to disk
        arc_dir = output_dir / "arc_challenge"
        arc_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(arc_dir))

        logger.info(f"Saved ARC (Challenge) to {arc_dir}")

        # Print dataset info
        logger.info("ARC (Challenge) dataset info:")
        for split, data in dataset.items():
            logger.info(f"  {split}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download ARC (Challenge): {e}")


def download_drop(output_dir: Path) -> None:
    """Download DROP dataset."""
    logger.info("Downloading DROP dataset")

    try:
        dataset = load_dataset("drop")

        # Save to disk
        drop_dir = output_dir / "drop"
        drop_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(drop_dir))

        logger.info(f"Saved DROP to {drop_dir}")

        # Print dataset info
        logger.info("DROP dataset info:")
        for split, data in dataset.items():
            logger.info(f"  {split}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download DROP: {e}")


def download_gpqa_diamond(output_dir: Path) -> None:
    """Download GPQA Diamond dataset (graduate-level STEM knowledge challenge)."""
    logger.info("Downloading GPQA Diamond dataset")

    try:
        # Load GPQA Diamond from Hugging Face
        dataset = load_dataset("test-time-compute/test_gpqa_diamond")

        # Ensure we have train/val splits
        if len(dataset) == 1:
            # Single split, create train/val (80/20)
            split_data = list(dataset.values())[0]
            split_dataset = split_data.train_test_split(test_size=0.2, seed=42)
            gpqa_dataset = DatasetDict({
                "train": split_dataset["train"],
                "validation": split_dataset["test"],
            })
        else:
            # Multiple splits, use as-is but ensure we have train/val
            if "train" in dataset and "validation" in dataset:
                gpqa_dataset = DatasetDict({
                    "train": dataset["train"],
                    "validation": dataset["validation"],
                })
            elif "train" in dataset and "test" in dataset:
                gpqa_dataset = DatasetDict({
                    "train": dataset["train"],
                    "validation": dataset["test"],
                })
            else:
                # Use first split as train, second as val
                splits = list(dataset.keys())
                gpqa_dataset = DatasetDict({
                    "train": dataset[splits[0]],
                    "validation": dataset[splits[1]] if len(splits) > 1 else dataset[splits[0]],
                })

        # Save to disk
        gpqa_dir = output_dir / "gpqa_diamond"
        gpqa_dir.mkdir(parents=True, exist_ok=True)
        gpqa_dataset.save_to_disk(str(gpqa_dir))

        logger.info(f"Saved GPQA Diamond to {gpqa_dir}")

        # Print dataset info
        logger.info("GPQA Diamond dataset info:")
        for split, data in gpqa_dataset.items():
            logger.info(f"  {split}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download GPQA Diamond: {e}")
        raise


def download_triviaqa(output_dir: Path) -> None:
    """Download TriviaQA dataset."""
    logger.info("Downloading TriviaQA dataset")

    try:
        # TriviaQA has multiple configurations, use the web-based version
        dataset = load_dataset("trivia_qa", "rc")

        # Save to disk
        triviaqa_dir = output_dir / "triviaqa"
        triviaqa_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(triviaqa_dir))

        logger.info(f"Saved TriviaQA to {triviaqa_dir}")

        # Print dataset info
        logger.info("TriviaQA dataset info:")
        for split, data in dataset.items():
            logger.info(f"  {split}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download TriviaQA: {e}")


def download_mmlu_stem(output_dir: Path) -> None:
    """Download MMLU-STEM subset (all STEM subjects, all available examples).
    
    Includes all Science, Technology, Engineering, and Mathematics subjects:
    - Physics (college_physics, conceptual_physics, high_school_physics)
    - Chemistry (college_chemistry, high_school_chemistry)
    - Biology (college_biology, high_school_biology, anatomy)
    - Mathematics (college_mathematics, high_school_mathematics, elementary_mathematics, 
                   abstract_algebra, high_school_statistics)
    - Computer Science (college_computer_science, high_school_computer_science, 
                        computer_security, machine_learning)
    - Engineering (electrical_engineering)
    - Astronomy
    """
    logger.info("Downloading MMLU-STEM subset (all STEM subjects)")

    try:
        # Load MMLU dataset - try "all" configuration first
        try:
            full_dataset = load_dataset("cais/mmlu", "all")
        except Exception:
            # Fallback: try loading individual subjects
            logger.info("Trying alternative MMLU loading method...")
            full_dataset = load_dataset("cais/mmlu", "college_physics")
            # This approach would need to be adjusted if "all" doesn't work
        
        # Check available splits
        available_splits = list(full_dataset.keys())
        logger.info(f"Available splits: {available_splits}")
        
        # Identify all STEM subjects
        # Get all unique subjects from test split
        test_subjects = set(full_dataset["test"]["subject"])
        
        # STEM keywords to identify STEM subjects
        stem_keywords = [
            "physics", "chemistry", "chemistry", "biology", "math", "mathematics",
            "computer", "engineering", "anatomy", "astronomy", "biochemistry",
            "machine_learning", "statistics", "calculus", "geometry", "algebra",
            "precalculus", "security"
        ]
        
        stem_subjects = [
            s for s in test_subjects 
            if any(kw in s.lower() for kw in stem_keywords)
        ]
        
        logger.info(f"Found {len(stem_subjects)} STEM subjects: {sorted(stem_subjects)}")
        
        # Combine all available splits (test, validation, dev) to maximize dataset size
        splits_to_use = ["test", "validation", "dev"]
        stem_datasets = []
        subject_counts = {}
        
        for split_name in splits_to_use:
            if split_name in full_dataset:
                split_data = full_dataset[split_name]
                for subject in stem_subjects:
                    subject_data = split_data.filter(
                        lambda x: x["subject"] == subject
                    )
                    if len(subject_data) > 0:
                        stem_datasets.append(subject_data)
                        if subject not in subject_counts:
                            subject_counts[subject] = 0
                        subject_counts[subject] += len(subject_data)
                        logger.info(f"  {split_name}: {len(subject_data)} {subject} examples")
        
        if len(stem_datasets) == 0:
            raise ValueError("No STEM examples found in MMLU dataset")
        
        # Concatenate all STEM datasets
        combined = datasets.concatenate_datasets(stem_datasets)
        
        logger.info(f"Total STEM examples found: {len(combined)}")
        logger.info("Subject breakdown:")
        for subject in sorted(subject_counts.keys()):
            logger.info(f"  {subject}: {subject_counts[subject]} examples")
        
        # Shuffle the combined dataset
        combined_shuffled = combined.shuffle(seed=42)
        
        # Split into train/validation (80/20)
        split_dataset = combined_shuffled.train_test_split(test_size=0.2, seed=42)
        
        mmlu_stem_dataset = DatasetDict({
            "train": split_dataset["train"],
            "validation": split_dataset["test"],
        })

        # Save to disk
        mmlu_stem_dir = output_dir / "mmlu_stem"
        mmlu_stem_dir.mkdir(parents=True, exist_ok=True)
        mmlu_stem_dataset.save_to_disk(str(mmlu_stem_dir))

        logger.info(f"Saved MMLU-STEM to {mmlu_stem_dir}")
        logger.info(f"Total: {len(combined)} examples from {len(stem_subjects)} STEM subjects")

        # Print dataset info
        logger.info("MMLU-STEM dataset info:")
        for split, data in mmlu_stem_dataset.items():
            logger.info(f"  {split}: {len(data)} examples")

    except Exception as e:
        logger.error(f"Failed to download MMLU-STEM: {e}")
        raise


def download_prm800k(output_dir: Path) -> None:
    """Download PRM800K dataset from Hugging Face."""
    logger.info("Downloading PRM800K dataset from Hugging Face...")
    
    prm800k_dir = output_dir / "prm800k"
    prm800k_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download directly from Hugging Face using huggingface_hub
        from huggingface_hub import snapshot_download
        
        repo_id = "tasksource/PRM800K"
        logger.info(f"Downloading from {repo_id}...")
        
        # Download all files from the repository
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(prm800k_dir),
            local_dir_use_symlinks=False
        )
        
        logger.info(f"Downloaded PRM800K to {prm800k_dir}")
        
        # Try to load as dataset
        try:
            dataset = load_dataset("json", data_files={
                "train": str(prm800k_dir / "phase1_train.jsonl"),
                "test": str(prm800k_dir / "phase1_test.jsonl"),
            })
            dataset.save_to_disk(str(prm800k_dir))
            logger.info("PRM800K dataset info:")
            for split, data in dataset.items():
                logger.info(f"  {split}: {len(data)} examples")
        except Exception as load_error:
            logger.warning(f"Could not load as dataset: {load_error}, but files are downloaded to {prm800k_dir}")
            
    except Exception as e:
        logger.error(f"Failed to download PRM800K: {e}")
        raise
        # Try to load from Hugging Face first (if available)
        try:
            logger.info("Attempting to load PRM800K from Hugging Face...")
            # Try with download_mode to reuse cache
            dataset = load_dataset(
                "tasksource/PRM800K", 
                trust_remote_code=True,
                download_mode="reuse_cache_if_exists"
            )
            
            # Save to disk
            prm800k_dir = output_dir / "prm800k"
            prm800k_dir.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(prm800k_dir))
            
            logger.info(f"Saved PRM800K to {prm800k_dir}")
            
            # Print dataset info
            logger.info("PRM800K dataset info:")
            for split, data in dataset.items():
                logger.info(f"  {split}: {len(data)} examples")
                
        except Exception as hf_error:
            # Try to use the dataset builder to access downloaded files
            try:
                logger.info(f"Standard load failed: {hf_error}")
                logger.info("Attempting to use dataset builder to access downloaded files...")
                
                from datasets import load_dataset_builder
                builder = load_dataset_builder("tasksource/PRM800K", trust_remote_code=True)
                
                # Download the data files
                logger.info("Downloading data files via builder...")
                builder.download_and_prepare(download_mode="reuse_cache_if_exists")
                
                # Try to build the dataset
                dataset = builder.as_dataset()
                
                # Save to disk
                prm800k_dir = output_dir / "prm800k"
                prm800k_dir.mkdir(parents=True, exist_ok=True)
                dataset.save_to_disk(str(prm800k_dir))
                
                logger.info(f"Saved PRM800K to {prm800k_dir}")
                logger.info("PRM800K dataset info:")
                for split, data in dataset.items():
                    logger.info(f"  {split}: {len(data)} examples")
                    
            except Exception as builder_error:
                logger.info(f"Builder approach also failed: {builder_error}")
                logger.info(f"Hugging Face dataset generation failed: {hf_error}")
                logger.info("Attempting to load downloaded JSONL files directly...")
            
            # Download from GitHub repository
            prm800k_dir = output_dir / "prm800k"
            prm800k_dir.mkdir(parents=True, exist_ok=True)
            
            # Try to find and load JSONL files from Hugging Face cache
            try:
                import os
                # Try multiple possible cache locations
                possible_cache_dirs = [
                    Path(os.path.expanduser("~/.cache/huggingface/datasets")),
                    Path(os.path.expanduser("~/cache/huggingface/datasets")),
                    Path("/tmp/huggingface/datasets"),
                ]
                
                cache_dir = None
                for possible_dir in possible_cache_dirs:
                    if possible_dir.exists():
                        cache_dir = possible_dir
                        break
                
                if cache_dir is None:
                    cache_dir = Path(os.path.expanduser("~/.cache/huggingface/datasets"))
                
                # Look for JSONL files in the cache directory (more thorough search)
                jsonl_files_found = []
                if cache_dir.exists():
                    logger.info(f"Searching for JSONL files in cache: {cache_dir}")
                    # First, try to find all JSONL files recursively (broader search)
                    all_jsonl_files = list(cache_dir.rglob("*.jsonl"))
                    logger.info(f"Found {len(all_jsonl_files)} total JSONL files in cache")
                    
                    # Filter for PRM800K files
                    for jsonl_file in all_jsonl_files:
                        name_lower = jsonl_file.name.lower()
                        # Check if it looks like a PRM800K file
                        if any(keyword in name_lower for keyword in ["phase1", "phase2"]):
                            jsonl_files_found.append(jsonl_file)
                            logger.info(f"Found PRM800K file: {jsonl_file}")
                    
                    # If still not found, try searching by parent directory
                    if not jsonl_files_found:
                        logger.info("Searching by directory name patterns...")
                        for cache_subdir in cache_dir.iterdir():
                            if cache_subdir.is_dir():
                                name_lower = cache_subdir.name.lower()
                                # More lenient search - look for tasksource or any directory with downloaded files
                                if "tasksource" in name_lower or any(x in name_lower for x in ["prm", "800k"]):
                                    logger.info(f"Checking directory: {cache_subdir}")
                                    # List all files in this directory for debugging
                                    all_files = list(cache_subdir.rglob("*"))
                                    logger.info(f"Found {len(all_files)} total files/dirs in {cache_subdir}")
                                    # Look for any files that might be data files (not just .jsonl)
                                    for file_path in all_files:
                                        if file_path.is_file():
                                            logger.info(f"  File: {file_path.name} (size: {file_path.stat().st_size} bytes)")
                                            # Check if it's a JSONL file (might not have extension)
                                            if file_path.suffix in [".jsonl", ".json"] or "phase" in file_path.name.lower():
                                                jsonl_files_found.append(file_path)
                                                logger.info(f"Found potential data file: {file_path}")
                                    # Also search for .jsonl specifically
                                    for jsonl_file in cache_subdir.rglob("*.jsonl"):
                                        if jsonl_file not in jsonl_files_found:
                                            jsonl_files_found.append(jsonl_file)
                                            logger.info(f"Found JSONL file: {jsonl_file}")
                    
                    # Copy found JSONL files
                    if jsonl_files_found:
                        for jsonl_file in jsonl_files_found:
                            dest_file = prm800k_dir / jsonl_file.name
                            if not dest_file.exists():  # Avoid overwriting
                                shutil.copy2(jsonl_file, dest_file)
                                logger.info(f"Copied {jsonl_file.name} to {dest_file}")
                    else:
                        logger.warning(f"No PRM800K JSONL files found in cache. Searched {len(all_jsonl_files)} total JSONL files.")
                        # List some example files for debugging
                        if all_jsonl_files:
                            logger.info(f"Example files found: {[f.name for f in all_jsonl_files[:5]]}")
                        raise ValueError("No JSONL files found in cache")
                else:
                    logger.warning(f"Cache directory does not exist: {cache_dir}")
                    raise ValueError(f"Cache directory not found: {cache_dir}")
                
                # Try to load JSONL files directly
                jsonl_files = list(prm800k_dir.glob("*.jsonl"))
                if jsonl_files:
                    logger.info(f"Found {len(jsonl_files)} JSONL files, loading as dataset...")
                    # Group files by split
                    data_files = {}
                    for jsonl_file in jsonl_files:
                        name_lower = jsonl_file.stem.lower()
                        if "train" in name_lower:
                            if "train" not in data_files:
                                data_files["train"] = []
                            data_files["train"].append(str(jsonl_file))
                        elif "test" in name_lower:
                            if "test" not in data_files:
                                data_files["test"] = []
                            data_files["test"].append(str(jsonl_file))
                        elif "val" in name_lower:
                            if "validation" not in data_files:
                                data_files["validation"] = []
                            data_files["validation"].append(str(jsonl_file))
                    
                    # Load dataset from JSONL files
                    if data_files:
                        # If multiple files per split, concatenate them
                        final_data_files = {}
                        for split, files in data_files.items():
                            if len(files) == 1:
                                final_data_files[split] = files[0]
                            else:
                                # Multiple files - will need to concatenate
                                final_data_files[split] = files
                        
                        dataset = load_dataset("json", data_files=final_data_files)
                        dataset.save_to_disk(str(prm800k_dir))
                        logger.info(f"Saved PRM800K to {prm800k_dir}")
                        logger.info("PRM800K dataset info:")
                        for split, data in dataset.items():
                            logger.info(f"  {split}: {len(data)} examples")
                        return  # Success, exit early
                    else:
                        raise ValueError("Could not identify train/test/val splits in JSONL files")
                else:
                    raise ValueError("No JSONL files found in cache")
                    
            except Exception as jsonl_error:
                logger.info(f"Loading JSONL files from cache failed: {jsonl_error}")
                logger.info("Attempting to download from GitHub...")
            
            # Check if git-lfs is available
            try:
                subprocess.run(["git-lfs", "version"], check=True, capture_output=True)
                has_git_lfs = True
                logger.info("git-lfs is available")
            except (subprocess.CalledProcessError, FileNotFoundError):
                has_git_lfs = False
                logger.warning("git-lfs is not available, will try shallow clone or direct download")
            
            # Try cloning with different strategies
            clone_success = False
            with tempfile.TemporaryDirectory() as tmpdir:
                repo_path = Path(tmpdir) / "prm800k"
                
                if has_git_lfs:
                    # Try full clone with git-lfs
                    try:
                        logger.info("Cloning PRM800K repository from GitHub (with git-lfs)...")
                        result = subprocess.run(
                            ["git", "clone", "https://github.com/openai/prm800k.git", str(repo_path)],
                            capture_output=True,
                            text=True,
                            timeout=300
                        )
                        if result.returncode == 0:
                            # Initialize and pull LFS files
                            subprocess.run(
                                ["git", "lfs", "pull"],
                                cwd=str(repo_path),
                                check=True,
                                capture_output=True
                            )
                            clone_success = True
                    except Exception as e:
                        logger.warning(f"Full clone with git-lfs failed: {e}")
                
                if not clone_success:
                    # Try shallow clone (without LFS files, then download data files directly)
                    try:
                        logger.info("Trying shallow clone (without LFS files)...")
                        subprocess.run(
                            ["git", "clone", "--depth", "1", "https://github.com/openai/prm800k.git", str(repo_path)],
                            check=True,
                            capture_output=True,
                            text=True,
                            timeout=300
                        )
                        clone_success = True
                        logger.info("Shallow clone successful, downloading data files directly...")
                        
                        # Download data files directly from GitHub raw URLs
                        data_files = [
                            "data/train.jsonl",
                            "data/test.jsonl",
                            "data/val.jsonl",
                        ]
                        
                        for data_file in data_files:
                            raw_url = f"https://raw.githubusercontent.com/openai/prm800k/main/{data_file}"
                            local_file = prm800k_dir / Path(data_file).name
                            try:
                                logger.info(f"Downloading {data_file}...")
                                urllib.request.urlretrieve(raw_url, local_file)
                                logger.info(f"Downloaded {local_file.name}")
                            except Exception as e:
                                logger.warning(f"Failed to download {data_file}: {e}")
                                
                    except Exception as e:
                        logger.warning(f"Shallow clone failed: {e}")
                        # Fall back to downloading specific files directly
                        logger.info("Attempting to download files directly from GitHub...")
                        data_files = [
                            "data/train.jsonl",
                            "data/test.jsonl", 
                            "data/val.jsonl",
                        ]
                        
                        for data_file in data_files:
                            raw_url = f"https://raw.githubusercontent.com/openai/prm800k/main/{data_file}"
                            local_file = prm800k_dir / Path(data_file).name
                            try:
                                logger.info(f"Downloading {data_file}...")
                                urllib.request.urlretrieve(raw_url, local_file)
                                logger.info(f"Downloaded {local_file.name}")
                            except Exception as e:
                                logger.warning(f"Failed to download {data_file}: {e}")
                
                # If clone succeeded, copy files from cloned repo
                if clone_success and repo_path.exists():
                    data_dir = repo_path / "data"
                    if data_dir.exists():
                        # Copy all JSON/JSONL files from data directory
                        for data_file in data_dir.glob("*.json*"):
                            shutil.copy2(data_file, prm800k_dir / data_file.name)
                            logger.info(f"Copied {data_file.name}")
                        
                        # Also copy any subdirectories if they exist
                        for subdir in data_dir.iterdir():
                            if subdir.is_dir():
                                dest_subdir = prm800k_dir / subdir.name
                                if not dest_subdir.exists():
                                    shutil.copytree(subdir, dest_subdir)
                                    logger.info(f"Copied directory {subdir.name}")
                    else:
                        # If data directory doesn't exist, copy all JSON files from root
                        for json_file in repo_path.glob("*.json*"):
                            shutil.copy2(json_file, prm800k_dir / json_file.name)
                            logger.info(f"Copied {json_file.name}")
            
            logger.info(f"Saved PRM800K to {prm800k_dir}")
            logger.info("PRM800K dataset downloaded from GitHub")
            
            # Try to load and show info if possible
            try:
                # Check if we can load the JSON/JSONL files
                json_files = list(prm800k_dir.glob("*.json*"))
                if json_files:
                    logger.info(f"Found {len(json_files)} JSON/JSONL files in PRM800K dataset")
                    for json_file in json_files:
                        try:
                            with open(json_file, 'r') as f:
                                if json_file.suffix == '.jsonl':
                                    # Count lines for JSONL
                                    count = sum(1 for _ in f)
                                    logger.info(f"  {json_file.name}: {count} examples (JSONL)")
                                else:
                                    data = json.load(f)
                                    if isinstance(data, list):
                                        logger.info(f"  {json_file.name}: {len(data)} examples")
                                    elif isinstance(data, dict):
                                        logger.info(f"  {json_file.name}: {len(data)} keys")
                        except Exception as e:
                            logger.debug(f"Could not read {json_file.name}: {e}")
            except Exception as info_error:
                logger.debug(f"Could not load dataset info: {info_error}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone PRM800K repository: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Failed to download PRM800K: {e}")
        raise


def preprocess_datasets(data_dir: Path) -> None:
    """Apply preprocessing to downloaded datasets."""
    logger.info("Applying preprocessing to datasets")

    preprocess_dir = data_dir / "preprocess"
    preprocess_dir.mkdir(exist_ok=True)

    # Preprocess each dataset
    tasks = [
        "sst2",
        "qnli", 
        "mnli",
        "mrpc",
        "qqp",
        "cola",
        "rte",
        "wnli",
        "pawsx",
        "boolq",
        "agnews",
        "xsum",
        "hotpot",
        "samsum",
        "coco_captions",
        "flickr30k",
        "vizwiz_vqa",
        "audiocaps",
        "clotho",
        "anli",
        "winogrande",
        "hellaswag",
        "openbookqa",
        "piqa",
        "strategyqa",
        "gsm8k",
        "arc_easy",
        "arc_challenge",
        "drop",
        "esc50",
            "prm800k",
            "triviaqa",
            "mmlu_stem",
            "math500",
            "gpqa_diamond",
            "aime",
            "aime_2024",
            "minerva_math",
            "omnimath",
            "livemathbench",
            "amc",
    ]

    for task in tasks:
        task_dir = data_dir / task
        if not task_dir.exists():
            logger.warning(f"Dataset directory not found: {task_dir}")
            continue

        logger.info(f"Preprocessing {task}")

        try:
            # Load dataset
            dataset = datasets.load_from_disk(str(task_dir))

            # Task-specific preprocessing
            if task in ["sst2", "qnli", "mnli", "mrpc", "qqp", "cola", "rte", "wnli"]:
                # GLUE tasks - already in good format
                processed_dataset = dataset

            elif task == "pawsx":
                # PAWS-X - already in good format (sentence1, sentence2, label)
                processed_dataset = dataset

            elif task == "boolq":
                # BoolQ - already in good format (question, passage, answer)
                processed_dataset = dataset

            elif task == "agnews":
                # AG News - already in good format (text, label)
                processed_dataset = dataset

            elif task in ["xsum", "samsum"]:
                # Text summarization - ensure consistent column names
                def preprocess_summarization(examples, current_task=task):
                    if current_task == "xsum":
                        return {
                            "document": examples["document"],
                            "summary": examples["summary"],
                            "id": examples["id"],
                        }
                    else:  # samsum
                        return {
                            "document": examples["dialogue"],
                            "summary": examples["summary"],
                            "id": examples["id"],
                        }

                processed_dataset = dataset.map(preprocess_summarization, batched=True)

            elif task == "hotpot":
                # HotpotQA - extract context and answer
                def preprocess_hotpot(examples):
                    contexts = []
                    for context in examples["context"]:
                        # Join all context paragraphs
                        context_text = " ".join([" ".join(para[1]) for para in context])
                        contexts.append(context_text)

                    return {
                        "question": examples["question"],
                        "context": contexts,
                        "answer": examples["answer"],
                        "id": examples["id"],
                    }

                processed_dataset = dataset.map(preprocess_hotpot, batched=True)

            elif task in ["coco_captions", "flickr30k", "vizwiz_vqa"]:
                # Vision captioning/VQA - ensure consistent format
                def preprocess_vision_caption(examples, current_task=task):
                    if current_task == "coco_captions":
                        # COCO has multiple captions per image, take the first one
                        captions = []
                        for caption_list in examples["captions"]:
                            if isinstance(caption_list, list) and len(caption_list) > 0:
                                captions.append(caption_list[0]["text"])
                            else:
                                captions.append("")

                        return {
                            "image": examples["image"],
                            "caption": captions,
                            "id": examples["image_id"],
                        }
                    elif current_task == "flickr30k":
                        # Flickr30k has multiple captions, take the first one
                        captions = []
                        for caption_list in examples["caption"]:
                            if isinstance(caption_list, list) and len(caption_list) > 0:
                                captions.append(caption_list[0])
                            else:
                                captions.append(
                                    caption_list
                                    if isinstance(caption_list, str)
                                    else ""
                                )

                        return {
                            "image": examples["image"],
                            "caption": captions,
                            "id": examples["sentids"],
                        }
                    else:  # vizwiz_vqa
                        # VizWiz-VQA has question, image, and answer(s)
                        answers = []
                        for answer_list in examples["answers"]:
                            if isinstance(answer_list, list) and len(answer_list) > 0:
                                # Take the first answer
                                answers.append(answer_list[0]["answer"] if isinstance(answer_list[0], dict) else answer_list[0])
                            else:
                                answers.append("")

                        return {
                            "image": examples["image"],
                            "question": examples["question"],
                            "answer": answers,
                            "id": examples["image"],
                        }

                processed_dataset = dataset.map(preprocess_vision_caption, batched=True)

            elif task in ["audiocaps", "clotho"]:
                # Audio captioning - ensure consistent format
                def preprocess_audio_caption(examples, current_task=task):
                    if current_task == "audiocaps":
                        return {
                            "audio": examples["audio"],
                            "caption": examples["caption"],
                            "id": examples["audiocap_id"],
                        }
                    else:  # clotho
                        # Clotho has multiple captions, take the first one
                        captions = []
                        for caption_list in examples["captions"]:
                            if isinstance(caption_list, list) and len(caption_list) > 0:
                                captions.append(caption_list[0])
                            else:
                                captions.append("")

                        return {
                            "audio": examples["audio"],
                            "caption": captions,
                            "id": examples["file_name"],
                        }

                processed_dataset = dataset.map(preprocess_audio_caption, batched=True)

            elif task in ["anli", "winogrande", "hellaswag", "openbookqa", "piqa", "strategyqa", "gsm8k", "arc_easy", "arc_challenge", "drop", "prm800k", "triviaqa", "mmlu_stem", "math500", "gpqa_diamond", "aime", "aime_2024", "minerva_math", "omnimath", "livemathbench", "amc"]:
                # Most new tasks are already in good format or need minimal preprocessing
                if task == "anli":
                    # ANLI - already in good format (premise, hypothesis, label)
                    processed_dataset = dataset
                elif task == "winogrande":
                    # Winogrande - already in good format (sentence, option1, option2, answer)
                    processed_dataset = dataset
                elif task == "hellaswag":
                    # HellaSwag - already in good format (ctx, endings, label)
                    processed_dataset = dataset
                elif task == "openbookqa":
                    # OpenBookQA - already in good format (question_stem, choices, answerKey)
                    processed_dataset = dataset
                elif task == "piqa":
                    # PIQA - already in good format (goal, sol1, sol2, label)
                    processed_dataset = dataset
                elif task == "strategyqa":
                    # StrategyQA - already in good format (question, answer)
                    processed_dataset = dataset
                elif task == "gsm8k":
                    # GSM8K - already in good format (question, answer)
                    processed_dataset = dataset
                elif task == "arc_easy":
                    # ARC Easy - already in good format (question, choices, answerKey)
                    processed_dataset = dataset
                elif task == "arc_challenge":
                    # ARC Challenge - already in good format (question, choices, answerKey)
                    processed_dataset = dataset
                elif task == "drop":
                    # DROP - already in good format (passage, question, answers_spans)
                    processed_dataset = dataset
                elif task == "prm800k":
                    # PRM800K - mathematical reasoning dataset, may need format conversion
                    # If it's JSON files, we'll keep them as-is for now
                    processed_dataset = dataset
                elif task == "triviaqa":
                    # TriviaQA - already in good format (question, answer, evidence)
                    processed_dataset = dataset
                elif task == "mmlu_stem":
                    # MMLU-STEM - already in good format (question, choices, answer)
                    processed_dataset = dataset
                elif task == "math500":
                    # MATH-500 - already in good format (problem, solution, or similar)
                    processed_dataset = dataset
                elif task == "gpqa_diamond":
                    # GPQA Diamond - already in good format (question, choices, answer, or similar)
                    processed_dataset = dataset
                elif task == "aime":
                    # AIME - already in good format (problem, answer, id)
                    processed_dataset = dataset
                elif task in ["aime_2024", "minerva_math", "omnimath", "livemathbench", "amc"]:
                    # Math benchmarks - already in good format (problem/question, answer, etc.)
                    processed_dataset = dataset
                elif task == "esc50":
                    # ESC-50 - already organized in 5-fold structure
                    processed_dataset = dataset
                else:
                    # Default: no preprocessing needed
                    processed_dataset = dataset
            else:
                # Default: no preprocessing needed
                processed_dataset = dataset

            # Save preprocessed dataset
            processed_dir = preprocess_dir / task
            processed_dataset.save_to_disk(str(processed_dir))

            logger.info(f"Preprocessed {task} saved to {processed_dir}")

        except Exception as e:
            logger.error(f"Failed to preprocess {task}: {e}")


def create_dataset_info(data_dir: Path) -> None:
    """Create a summary file with dataset information."""
    info_file = data_dir / "dataset_info.json"

    # Find all valid dataset directories (ones that can actually be loaded)
    available_tasks = []
    all_possible_tasks = [
        "sst2",
        "qnli",
        "mnli",
        "mrpc",
        "qqp",
        "cola",
        "rte",
        "wnli",
        "pawsx",
        "boolq",
        "agnews",
        "xsum",
        "hotpot",
        "samsum",
        "coco_captions",
        "flickr30k",
        "vizwiz_vqa",
        "audiocaps",
        "clotho",
        "anli",
        "winogrande",
        "hellaswag",
        "openbookqa",
        "piqa",
        "strategyqa",
        "gsm8k",
        "arc_easy",
        "arc_challenge",
        "drop",
        "esc50",
            "prm800k",
            "triviaqa",
            "mmlu_stem",
            "math500",
            "gpqa_diamond",
            "aime",
            "aime_2024",
            "minerva_math",
            "omnimath",
            "livemathbench",
            "amc",
        ]

    for task in all_possible_tasks:
        task_dir = data_dir / task
        if task_dir.exists() and task_dir.is_dir():
            try:
                # Test if dataset can be loaded (i.e., has valid data files)
                dataset = datasets.load_from_disk(str(task_dir))
                if len(dataset) > 0:  # Check if dataset has splits
                    available_tasks.append(task)
                    logger.debug(f"Found valid dataset: {task}")
            except Exception:
                logger.debug(f"Skipping invalid/incomplete dataset directory: {task}")

    info = {"datasets": {}, "total_size": 0, "tasks": available_tasks}

    for task in available_tasks:
        task_dir = data_dir / task
        try:
            dataset = datasets.load_from_disk(str(task_dir))
            task_info = {
                "splits": {},
                "features": list(dataset[list(dataset.keys())[0]].features.keys()),
            }

            total_examples = 0
            for split, data in dataset.items():
                task_info["splits"][split] = len(data)
                total_examples += len(data)

            task_info["total_examples"] = total_examples
            info["datasets"][task] = task_info
            info["total_size"] += total_examples

            logger.info(f"Gathered info for {task}: {total_examples} examples")

        except Exception as e:
            logger.error(f"Failed to get info for {task}: {e}")

    # Save info file
    with open(info_file, "w") as f:
        json.dump(info, f, indent=2)

    logger.info(
        f"Dataset info saved to {info_file} for {len(available_tasks)} valid datasets"
    )


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess datasets")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="./datasets",
        help="Output directory for datasets",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=[
            "sst2",
            "qnli",
            "mnli",
            "mrpc",
            "qqp",
            "cola",
            "rte",
            "wnli",
            "pawsx",
            "boolq",
            "agnews",
            "xsum",
            "hotpot",
            "samsum",
            "coco_captions",
            "flickr30k",
            "vizwiz_vqa",
            "audiocaps",
            "clotho",
            "anli",
            "winogrande",
            "hellaswag",
            "openbookqa",
            "piqa",
            "strategyqa",
            "gsm8k",
            "arc_easy",
            "arc_challenge",
            "drop",
            "esc50",
            "prm800k",
            "triviaqa",
            "mmlu_stem",
            "math500",
            "gpqa_diamond",
            "aime",
            "aime_2024",
            "minerva_math",
            "omnimath",
            "livemathbench",
            "amc",
            "all",
        ],
        default=["all"],
        help="Tasks to download",
    )
    parser.add_argument(
        "--preprocess", action="store_true", help="Apply preprocessing after download"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-download even if datasets exist"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine tasks to download
    if "all" in args.tasks:
        tasks_to_download = [
            "sst2",
            "qnli",
            "mnli",
            "mrpc",
            "qqp",
            "cola",
            "rte",
            "wnli",
            "pawsx",
            "boolq",
            "agnews",
            "xsum",
            "hotpot",
            "samsum",
            "coco_captions",
            "flickr30k",
            "vizwiz_vqa",
            "audiocaps",
            "clotho",
            "anli",
            "winogrande",
            "hellaswag",
            "openbookqa",
            "piqa",
            "strategyqa",
            "gsm8k",
            "arc_easy",
            "arc_challenge",
            "drop",
            "esc50",
            "prm800k",
            "triviaqa",
            "mmlu_stem",
            "math500",
            "gpqa_diamond",
            "aime",
            "aime_2024",
            "minerva_math",
            "omnimath",
            "livemathbench",
            "amc",
        ]
    else:
        tasks_to_download = args.tasks

    logger.info(f"Downloading tasks: {tasks_to_download}")

    # Download datasets
    for task in tasks_to_download:
        task_dir = args.output_dir / task

        if task_dir.exists() and not args.force:
            logger.info(f"Dataset {task} already exists, skipping download")
            continue

        if task in ["sst2", "qnli", "mnli", "mrpc", "qqp", "cola", "rte", "wnli"]:
            download_glue_tasks(args.output_dir)
            break  # GLUE tasks are downloaded together
        elif task == "pawsx":
            download_pawsx(args.output_dir)
        elif task == "boolq":
            download_boolq(args.output_dir)
        elif task == "agnews":
            download_agnews(args.output_dir)
        elif task == "xsum":
            download_xsum(args.output_dir)
        elif task == "hotpot":
            download_hotpotqa(args.output_dir)
        elif task == "samsum":
            download_samsum(args.output_dir)
        elif task == "coco_captions":
            download_coco_captions(args.output_dir)
        elif task == "flickr30k":
            download_flickr30k(args.output_dir)
        elif task == "vizwiz_vqa":
            download_vizwiz_vqa(args.output_dir)
        elif task == "audiocaps":
            download_audiocaps(args.output_dir)
        elif task == "clotho":
            download_clotho(args.output_dir)
        elif task == "anli":
            download_anli(args.output_dir)
        elif task == "winogrande":
            download_winogrande(args.output_dir)
        elif task == "hellaswag":
            download_hellaswag(args.output_dir)
        elif task == "openbookqa":
            download_openbookqa(args.output_dir)
        elif task == "piqa":
            download_piqa(args.output_dir)
        elif task == "strategyqa":
            download_strategyqa(args.output_dir)
        elif task == "gsm8k":
            download_gsm8k(args.output_dir)
        elif task == "arc_easy":
            download_arc(args.output_dir)
        elif task == "arc_challenge":
            download_arc_challenge(args.output_dir)
        elif task == "drop":
            download_drop(args.output_dir)
        elif task == "prm800k":
            download_prm800k(args.output_dir)
        elif task == "esc50":
            download_esc50(args.output_dir)
        elif task == "triviaqa":
            download_triviaqa(args.output_dir)
        elif task == "mmlu_stem":
            download_mmlu_stem(args.output_dir)
        elif task == "math500":
            download_math500(args.output_dir)
        elif task == "gpqa_diamond":
            download_gpqa_diamond(args.output_dir)
        elif task == "aime":
            download_aime(args.output_dir)
        elif task == "aime_2024":
            download_aime_2024(args.output_dir)
        elif task == "minerva_math":
            download_minerva_math(args.output_dir)
        elif task == "omnimath":
            download_omnimath(args.output_dir)
        elif task == "livemathbench":
            download_livemathbench(args.output_dir)
        elif task == "amc":
            download_amc(args.output_dir)

    # Apply preprocessing if requested
    if args.preprocess:
        preprocess_datasets(args.output_dir)

    # Create dataset info
    create_dataset_info(args.output_dir)

    logger.info("Dataset download and preprocessing completed!")


if __name__ == "__main__":
    main()
