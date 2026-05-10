"""
loader.py — NSL-KDD Dataset Loader
=====================================
Downloads and preprocesses the NSL-KDD dataset.
Handles:
  - Downloading train/test splits from GitHub mirror
  - Column naming (41 features + label + difficulty)
  - Encoding categorical features (protocol_type, service, flag)
  - Binary label encoding  (normal=0, attack=1)
  - Multiclass label encoding (normal=0, DoS=1, Probe=2, R2L=3, U2R=4)
  - Train/test split management

NSL-KDD is the improved version of KDD Cup 1999.
It removes duplicate records and balances attack categories.
"""

from __future__ import annotations
import os
import urllib.request
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# GitHub mirror of NSL-KDD
TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
TEST_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"

COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty_level",
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# Attack families in NSL-KDD
DOS_ATTACKS = {
    "back",
    "land",
    "neptune",
    "pod",
    "smurf",
    "teardrop",
    "apache2",
    "udpstorm",
    "processtable",
    "worm",
}
PROBE_ATTACKS = {"ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"}
R2L_ATTACKS = {
    "ftp_write",
    "guess_passwd",
    "imap",
    "multihop",
    "phf",
    "spy",
    "warezclient",
    "warezmaster",
    "sendmail",
    "named",
    "snmpgetattack",
    "snmpguess",
    "httptunnel",
    "xlock",
    "xsnoop",
}
U2R_ATTACKS = {
    "buffer_overflow",
    "loadmodule",
    "perl",
    "rootkit",
    "sqlattack",
    "xterm",
    "ps",
}


class NSLKDDLoader:
    """
    Handles all data loading and preprocessing for NSL-KDD.
    Call load() to get ready-to-use numpy arrays.
    """

    def __init__(self, data_dir: str = "./data", sample_frac: float = 1.0):
        """
        Args:
            data_dir    : where to cache downloaded files
            sample_frac : fraction of data to use (0.0–1.0); use < 1.0 for
                          quick iteration during development
        """
        self.data_dir = data_dir
        self.sample_frac = sample_frac
        os.makedirs(data_dir, exist_ok=True)

    # ── Public ────────────────────────────────────────────────────────────────

    def load(self, label_mode: str = "binary") -> tuple:
        """
        Download (if needed) and return preprocessed arrays.

        Args:
            label_mode : "binary"     → 0=normal, 1=attack
                         "multiclass" → 0=normal,1=DoS,2=Probe,3=R2L,4=U2R

        Returns:
            (X_train, X_test, y_train, y_test, feature_names)
        """
        train_path = os.path.join(self.data_dir, "KDDTrain+.txt")
        test_path = os.path.join(self.data_dir, "KDDTest+.txt")

        self._download_if_missing(TRAIN_URL, train_path)
        self._download_if_missing(TEST_URL, test_path)

        df_train = self._read_csv(train_path)
        df_test = self._read_csv(test_path)

        # optional: subsample for faster experiments
        if self.sample_frac < 1.0:
            df_train = df_train.sample(frac=self.sample_frac, random_state=42)
            df_test = df_test.sample(frac=self.sample_frac, random_state=42)

        df_train, df_test = self._encode_categoricals(df_train, df_test)

        y_train = self._encode_labels(df_train["label"], label_mode)
        y_test = self._encode_labels(df_test["label"], label_mode)

        feature_cols = [
            c for c in df_train.columns if c not in ("label", "difficulty_level")
        ]
        X_train = df_train[feature_cols].values.astype(np.float32)
        X_test = df_test[feature_cols].values.astype(np.float32)

        self._print_summary(X_train, X_test, y_train, y_test, label_mode)
        return X_train, X_test, y_train, y_test, feature_cols

    # ── Private helpers ───────────────────────────────────────────────────────

    def _download_if_missing(self, url: str, path: str) -> None:
        if not os.path.exists(path):
            print(f"  [Loader] Downloading {os.path.basename(path)} …")
            urllib.request.urlretrieve(url, path)
            print(f"  [Loader] Saved → {path}")
        else:
            print(f"  [Loader] Using cached file → {path}")

    def _read_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, header=None, names=COLUMNS)
        df.drop(columns=["difficulty_level"], inplace=True)
        return df

    def _encode_categoricals(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit LabelEncoder on train, transform both train + test.
        Unknown test categories are mapped to a safe default (0).
        """
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            le.fit(train[col])
            classes = set(le.classes_)

            train[col] = le.transform(train[col])
            test[col] = test[col].apply(
                lambda v: le.transform([v])[0] if v in classes else 0
            )
        return train, test

    def _encode_labels(self, labels: pd.Series, mode: str) -> np.ndarray:
        if mode == "binary":
            return (labels != "normal").astype(int).values
        elif mode == "multiclass":
            return labels.apply(self._to_multiclass_label).values
        else:
            raise ValueError(
                f"Unknown label_mode: '{mode}'. Use 'binary' or 'multiclass'."
            )

    @staticmethod
    def _to_multiclass_label(label: str) -> int:
        if label == "normal":
            return 0
        if label in DOS_ATTACKS:
            return 1
        if label in PROBE_ATTACKS:
            return 2
        if label in R2L_ATTACKS:
            return 3
        if label in U2R_ATTACKS:
            return 4
        return 1  # unknown attacks → DoS (most common fallback)

    @staticmethod
    def _print_summary(X_train, X_test, y_train, y_test, mode: str) -> None:
        print(f"\n  {'─' * 50}")
        print(f"  Dataset     : NSL-KDD  |  Label mode: {mode}")
        print(
            f"  Train size  : {X_train.shape[0]:,} samples × {X_train.shape[1]} features"
        )
        print(
            f"  Test size   : {X_test.shape[0]:,} samples × {X_test.shape[1]} features"
        )

        labels, counts = np.unique(y_train, return_counts=True)
        label_names = {0: "normal", 1: "attack/DoS", 2: "Probe", 3: "R2L", 4: "U2R"}
        print("  Train class distribution:")
        for lbl, cnt in zip(labels, counts):
            pct = 100 * cnt / len(y_train)
            name = label_names.get(lbl, str(lbl))
            print(f"    [{lbl}] {name:<12} : {cnt:,}  ({pct:.1f}%)")
        print(f"  {'─' * 50}\n")
