"""Carregamento do dataset bruto junto ao manifesto de splits congelado."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rag_hatespeech_ptbr.splits import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_SOURCE_SHA256,
    calculate_sha256,
)

ID_COLUMN = "id"
LABEL_COLUMN = "offensive_label"
COMMENT_COLUMN = "comment"
SPLIT_COLUMN = "split"
KNOWN_SPLITS = ("train", "validation", "test")


class DatasetIntegrityError(RuntimeError):
    """Hash do dataset bruto ou do manifesto diverge da versão aprovada."""


def load_labeled_frame(
    dataset_path: Path,
    manifest_path: Path,
    *,
    verify_hashes: bool = True,
) -> pd.DataFrame:
    """Junta o HateBRXplain bruto ao manifesto local de splits, por `id`.

    Por padrão, recusa-se a prosseguir se o hash de qualquer um dos dois
    arquivos não corresponder à revisão congelada em `docs/12_decisoes.md`,
    para impedir que um dataset ou split diferente do auditado entre
    silenciosamente em um experimento. `verify_hashes=False` existe apenas
    para testes com dados sintéticos.
    """
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifesto de split não encontrado: {manifest_path}")

    if verify_hashes:
        source_sha256 = calculate_sha256(dataset_path)
        if source_sha256 != EXPECTED_SOURCE_SHA256:
            raise DatasetIntegrityError(
                "Hash do dataset bruto não corresponde à revisão aprovada em "
                "docs/12_decisoes.md; abortando para evitar treinar sobre dados não "
                "auditados."
            )
        manifest_sha256 = calculate_sha256(manifest_path)
        if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
            raise DatasetIntegrityError(
                "Hash do manifesto de split não corresponde à divisão congelada em "
                "docs/12_decisoes.md; abortando para evitar treinar sobre uma divisão "
                "não aprovada."
            )

    frame = pd.read_csv(dataset_path)
    manifest = pd.read_csv(manifest_path, usecols=[ID_COLUMN, SPLIT_COLUMN])
    joined = frame.merge(manifest, on=ID_COLUMN, how="inner", validate="one_to_one")
    if len(joined) != len(frame):
        raise DatasetIntegrityError(
            "Nem todo registro do dataset bruto possui uma entrada no manifesto de "
            "split; o manifesto pode estar desatualizado em relação ao dataset."
        )
    return joined


def get_split(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Retorna apenas as linhas do split pedido (`train`, `validation` ou `test`)."""
    if split_name not in KNOWN_SPLITS:
        raise ValueError(f"Split desconhecido: {split_name!r}. Use um de {KNOWN_SPLITS}.")
    return frame.loc[frame[SPLIT_COLUMN] == split_name].reset_index(drop=True)
