#!/usr/bin/env python3
"""
Extract ESM-2 embeddings for all protein structures.

ESM-2-650M produces 1280-dim per-residue embeddings that capture rich
sequence-structure-function relationships. These replace/augment our
current 28-dim handcrafted features.

Usage:
    python scripts/extract_esm_embeddings.py [--model esm2_t33_650M_UR50D]
"""

import argparse
import logging
from pathlib import Path
import torch
import esm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Amino acid mapping (same as in sug_module.py)
AA_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'UNK': 'X', 'MSE': 'M', 'SEC': 'C', 'PYL': 'K',
}


def residues_to_sequence(residues: list) -> str:
    """Convert list of 3-letter residue codes to 1-letter sequence."""
    seq = []
    for res in residues:
        res_upper = res.upper() if isinstance(res, str) else str(res).upper()
        aa = AA_MAP.get(res_upper, 'X')
        seq.append(aa)
    return ''.join(seq)


def extract_embeddings_for_structure(
    model,
    batch_converter,
    structure_path: Path,
    output_dir: Path,
    device: torch.device,
    repr_layer: int = 33,  # Last layer for ESM-2-650M
):
    """Extract ESM-2 embeddings for a single protein structure."""
    uniprot_id = structure_path.stem
    output_path = output_dir / f"{uniprot_id}_esm.pt"

    # Skip if already processed
    if output_path.exists():
        logger.debug(f"Skipping {uniprot_id} (already exists)")
        return True

    try:
        # Load processed structure
        processed = torch.load(structure_path, weights_only=False)
        residues = processed.get('residues', [])

        if len(residues) == 0:
            logger.warning(f"No residues found for {uniprot_id}")
            return False

        # Convert to sequence
        sequence = residues_to_sequence(residues)

        # ESM-2 has a max length limit (~1024 for 650M on 11GB)
        max_len = 1022  # Leave room for BOS/EOS tokens
        if len(sequence) > max_len:
            logger.warning(f"{uniprot_id}: Truncating sequence from {len(sequence)} to {max_len}")
            sequence = sequence[:max_len]

        # Prepare batch
        data = [(uniprot_id, sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        # Extract embeddings
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)

        # Get per-residue embeddings (exclude BOS/EOS tokens)
        embeddings = results["representations"][repr_layer][0, 1:-1, :]  # [seq_len, 1280]

        # Pad back to original length if truncated
        original_len = len(processed.get('residues', []))
        if embeddings.shape[0] < original_len:
            padding = torch.zeros(original_len - embeddings.shape[0], embeddings.shape[1], device=device)
            embeddings = torch.cat([embeddings, padding], dim=0)

        # Save embeddings
        torch.save({
            'embeddings': embeddings.cpu(),
            'sequence': sequence,
            'uniprot_id': uniprot_id,
            'model': 'esm2_t33_650M_UR50D',
            'repr_layer': repr_layer,
        }, output_path)

        return True

    except Exception as e:
        logger.error(f"Error processing {uniprot_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Extract ESM-2 embeddings for proteins")
    parser.add_argument("--model", type=str, default="esm2_t33_650M_UR50D",
                        help="ESM model name")
    parser.add_argument("--structure-dir", type=str, default="data/processed/structures",
                        help="Directory with processed .pt structure files")
    parser.add_argument("--output-dir", type=str, default="data/processed/esm_embeddings",
                        help="Output directory for embeddings")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (keep 1 for 11GB GPU)")
    args = parser.parse_args()

    # Setup paths
    structure_dir = Path(args.structure_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

    # Load ESM model
    logger.info(f"Loading ESM model: {args.model}")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    # Get representation layer (last layer)
    repr_layer = 33  # For ESM-2-650M

    # Find all structure files
    structure_files = sorted(structure_dir.glob("*.pt"))
    logger.info(f"Found {len(structure_files)} structure files")

    # Process each structure
    success_count = 0
    fail_count = 0

    for i, struct_path in enumerate(structure_files):
        if (i + 1) % 10 == 0 or i == 0:
            logger.info(f"Processing {i+1}/{len(structure_files)}: {struct_path.stem}")

        success = extract_embeddings_for_structure(
            model=model,
            batch_converter=batch_converter,
            structure_path=struct_path,
            output_dir=output_dir,
            device=device,
            repr_layer=repr_layer,
        )

        if success:
            success_count += 1
        else:
            fail_count += 1

        # Clear GPU cache periodically
        if device.type == "cuda" and (i + 1) % 50 == 0:
            torch.cuda.empty_cache()

    logger.info(f"Done! Success: {success_count}, Failed: {fail_count}")
    logger.info(f"Embeddings saved to: {output_dir}")


if __name__ == "__main__":
    main()
