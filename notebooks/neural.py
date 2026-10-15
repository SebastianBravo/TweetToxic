"""Redes neuronales pequeñas para TweetToxic, entrenadas desde cero con PyTorch.

Arquitecturas (de menos a más compleja), todas con la misma API que modeling.TweetToxicClassifier
(texto crudo -> predict_proba (n, 3) y predict [HS, TR, AG] con umbrales, jerarquía y tuits vacíos):

    nn_mlp       MLP sobre el TF-IDF de palabras + caracteres del 03
    nn_fasttext  estilo fastText: embeddings de palabras y de n-gramas de caracteres (hashing), promediados
    nn_cnn_word  CNN 1D sobre palabras (ventanas 2, 3, 4) + max-pooling
    nn_cnn_char  CNN 1D sobre caracteres (ventanas 3, 5, 7) + max-pooling
    nn_bigru     BiGRU con pooling por atención (los pesos de atención se pueden inspeccionar)

Todo lo que depende de datos (preprocesamiento, vocabularios, TF-IDF, embeddings Word2Vec/FastText de
inicialización y pesos de la red) se ajusta en fit(), así que en la CV se ajusta dentro de cada fold.

Protocolo de entrenamiento (fit):
- early stopping con una validación INTERNA (val_size del train, estratificada por combinación de
  etiquetas): nunca se mira el fold de validación externo;
- con fixed_epochs, se entrena con todo el train un número fijo de épocas, sin validación interna
  (modelo final);
- n_seeds redes por modelo (seed-ensembling): se promedian sus probabilidades.

Guardado: save_neural() escribe un .pt con state_dicts + configuración + vocabularios + versiones (sin
pickle de clases) y load_neural() lo reconstruye; modeling.load_model() despacha por extensión.
"""

from __future__ import annotations

import copy
import io
import platform
import random
import re
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.validation import check_is_fitted
from torch import nn

import modeling as md
import preprocessing as pp
from common import LABELS, SEED

FORMAT = "tweettoxic-neural-v1"
TOKEN_RX = re.compile(r"<\w+>|\b\w+\b")  # mismo patrón que el TF-IDF de palabras: conserva <USER>, <URL>, 't'
PAD, UNK = 0, 1
CHARS = [chr(i) for i in range(32, 127)]  # ASCII imprimible (el texto ya no tiene otros caracteres)
CHAR_INDEX = {c: i + 2 for i, c in enumerate(CHARS)}
MAX_WORDS, MAX_CHARS, MAX_FT_FEATURES = 64, 300, 512

ARCH_DEFAULTS = {
    "nn_mlp": {"hidden": 128, "dropout": 0.5, "lr": 1e-3},
    "nn_fasttext": {"emb_dim": 100, "dropout": 0.3, "lr": 5e-3, "buckets": 20000},
    "nn_cnn_word": {"emb_dim": 100, "filters": 64, "dropout": 0.5, "lr": 1e-3},
    "nn_cnn_char": {"emb_dim": 32, "filters": 64, "dropout": 0.5, "lr": 1e-3},
    "nn_bigru": {"emb_dim": 100, "hidden": 64, "dropout": 0.4, "lr": 1e-3},
}
WORD_ARCHS = {"nn_fasttext", "nn_cnn_word", "nn_bigru"}  # las que pueden inicializarse con Word2Vec/FastText


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tokenize(text: str) -> list[str]:
    return TOKEN_RX.findall(text)


def char_ngrams(word: str, nmin: int = 3, nmax: int = 5) -> list[str]:
    w = f"<{word}>"
    return [w[i:i + n] for n in range(nmin, nmax + 1) for i in range(len(w) - n + 1)]


def _pad(seqs: list[list[int]], max_len: int) -> torch.Tensor:
    out = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        s = s[:max_len] or [UNK]  # un texto vacío se representa con <unk>
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out


# ---------------------------------------------------------------------------
# Codificación de entradas (vocabularios y TF-IDF, ajustados con el train)
# ---------------------------------------------------------------------------
class Encoder:
    """Convierte las vistas del TweetPreprocessor en tensores para una arquitectura."""

    def __init__(self, arch: str, variant: pp.Variant, buckets: int = 20000, min_freq: int = 2):
        self.arch = arch
        self.variant = variant
        self.buckets = buckets
        self.min_freq = min_freq
        self.word_source = "lemmas" if variant.lemma else "text"

    def fit(self, views: pd.DataFrame) -> "Encoder":
        if self.arch == "nn_mlp":
            blocks = [("text", dict(ngram_range=(1, 2), lowercase=False, token_pattern=md.TOKEN_PATTERN, sublinear_tf=True)),
                      ("text", dict(analyzer="char_wb", ngram_range=(2, 5), lowercase=False, sublinear_tf=True, min_df=2))]
            if self.variant.lemma:
                blocks.append(("lemmas", dict(ngram_range=(1, 2), lowercase=False, token_pattern=md.TOKEN_PATTERN,
                                              sublinear_tf=True)))
            self.tfidf = [(col, TfidfVectorizer(**params).fit(views[col])) for col, params in blocks]
            self.n_inputs = sum(len(v.vocabulary_) for _, v in self.tfidf)
        elif self.arch != "nn_cnn_char":
            counts = Counter(w for t in views[self.word_source] for w in tokenize(t))
            words = sorted((w for w, c in counts.items() if c >= self.min_freq), key=lambda w: (-counts[w], w))
            self.itos = ["<pad>", "<unk>"] + words
            self.stoi = {w: i for i, w in enumerate(self.itos)}
        return self

    @property
    def n_vocab(self) -> int:
        if self.arch == "nn_cnn_char":
            return len(CHARS) + 2
        if self.arch == "nn_fasttext":
            return len(self.itos) + self.buckets
        return len(self.itos)

    def word_ids(self, text: str) -> list[int]:
        return [self.stoi.get(w, UNK) for w in tokenize(text)]

    def transform(self, views: pd.DataFrame):
        if self.arch == "nn_mlp":
            return sp.hstack([v.transform(views[col]) for col, v in self.tfidf]).tocsr().astype(np.float32)
        if self.arch == "nn_cnn_char":
            return _pad([[CHAR_INDEX.get(c, UNK) for c in t] for t in views["text"]], MAX_CHARS)
        if self.arch == "nn_fasttext":
            seqs = []
            for t in views[self.word_source]:
                words = tokenize(t)
                ids = [self.stoi.get(w, UNK) for w in words]
                ids += [len(self.itos) + zlib.crc32(g.encode()) % self.buckets for w in words for g in char_ngrams(w)]
                seqs.append(ids)
            return _pad(seqs, MAX_FT_FEATURES)
        return _pad([self.word_ids(t) for t in views[self.word_source]], MAX_WORDS)

    # --- serialización a tipos simples (para torch.save con weights_only) ---
    def to_dict(self) -> dict:
        d = {"arch": self.arch, "variant": self.variant.name, "buckets": self.buckets, "min_freq": self.min_freq}
        if self.arch == "nn_mlp":
            d["tfidf"] = [{"column": col, "params": {k: (list(v) if isinstance(v, tuple) else v)
                                                     for k, v in vec.get_params().items()
                                                     if k in ("ngram_range", "analyzer", "lowercase", "token_pattern",
                                                              "sublinear_tf", "min_df")},
                           "vocabulary": {k: int(i) for k, i in vec.vocabulary_.items()},
                           "idf": torch.tensor(vec.idf_, dtype=torch.float64)} for col, vec in self.tfidf]
        elif self.arch != "nn_cnn_char":
            d["itos"] = self.itos
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Encoder":
        enc = cls(d["arch"], pp.VARIANTS[d["variant"]], d["buckets"], d["min_freq"])
        if enc.arch == "nn_mlp":
            enc.tfidf = []
            for block in d["tfidf"]:
                params = {k: (tuple(v) if k == "ngram_range" else v) for k, v in block["params"].items()}
                vec = TfidfVectorizer(**params, vocabulary=block["vocabulary"])
                vec.idf_ = block["idf"].numpy()
                enc.tfidf.append((block["column"], vec))
            enc.n_inputs = sum(len(v.vocabulary) for _, v in enc.tfidf)
        elif enc.arch != "nn_cnn_char":
            enc.itos = list(d["itos"])
            enc.stoi = {w: i for i, w in enumerate(enc.itos)}
        return enc


# ---------------------------------------------------------------------------
# Redes
# ---------------------------------------------------------------------------
class MLPNet(nn.Module):
    def __init__(self, n_inputs, n_out, hidden=128, dropout=0.5, **_):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_inputs, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


class FastTextNet(nn.Module):
    def __init__(self, n_vocab, n_out, emb_dim=100, dropout=0.3, pretrained=None, **_):
        super().__init__()
        self.emb = nn.EmbeddingBag(n_vocab, emb_dim, mode="mean", padding_idx=PAD)
        nn.init.uniform_(self.emb.weight, -0.05, 0.05)
        if pretrained is not None:  # vectores propios para las palabras del vocabulario (no para los n-gramas)
            with torch.no_grad():
                self.emb.weight[:len(pretrained)] = pretrained
        with torch.no_grad():
            self.emb.weight[PAD].zero_()
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(emb_dim, n_out)

    def forward(self, x):
        return self.out(self.drop(self.emb(x)))


class CNNNet(nn.Module):
    def __init__(self, n_vocab, n_out, emb_dim=100, filters=64, kernels=(2, 3, 4), dropout=0.5, pretrained=None, **_):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, emb_dim, padding_idx=PAD)
        if pretrained is not None:
            with torch.no_grad():
                self.emb.weight.copy_(pretrained)
                self.emb.weight[PAD].zero_()
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, filters, k, padding=k // 2) for k in kernels])
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(filters * len(kernels), n_out)

    def forward(self, x):
        h = self.emb(x).transpose(1, 2)                       # (B, E, L)
        pooled = [torch.relu(conv(h)).amax(dim=2) for conv in self.convs]  # max-pooling en el tiempo
        return self.out(self.drop(torch.cat(pooled, dim=1)))


class BiGRUAttnNet(nn.Module):
    def __init__(self, n_vocab, n_out, emb_dim=100, hidden=64, dropout=0.4, pretrained=None, **_):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, emb_dim, padding_idx=PAD)
        if pretrained is not None:
            with torch.no_grad():
                self.emb.weight.copy_(pretrained)
                self.emb.weight[PAD].zero_()
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.att = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1, bias=False))
        self.out = nn.Linear(2 * hidden, n_out)

    def forward(self, x, return_attention=False):
        mask = x != PAD
        lengths = mask.sum(1).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(self.drop(self.emb(x)), lengths.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        h, _ = self.gru(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=x.shape[1])
        scores = self.att(h).squeeze(-1).masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).nan_to_num(0.0)  # pooling por atención
        context = (weights.unsqueeze(-1) * h).sum(1)
        logits = self.out(self.drop(context))
        return (logits, weights) if return_attention else logits


NETS = {"nn_mlp": MLPNet, "nn_fasttext": FastTextNet, "nn_cnn_word": CNNNet,
        "nn_cnn_char": lambda **kw: CNNNet(kernels=(3, 5, 7), **kw), "nn_bigru": BiGRUAttnNet}


# ---------------------------------------------------------------------------
# Embeddings propios para inicializar (gensim, entrenados con el train del fold)
# ---------------------------------------------------------------------------
def pretrain_vectors(sentences: list[list[str]], itos: list[str], kind: str, dim: int, seed: int) -> torch.Tensor:
    """Matriz (len(itos), dim) con Word2Vec o FastText (skip-gram) entrenados con `sentences`."""
    from gensim.models import FastText, Word2Vec

    import embeddings as emb

    common = dict(vector_size=dim, window=5, min_count=1, sg=1, negative=5, epochs=30, workers=1, seed=seed,
                  hashfxn=emb.stable_hash)
    model = FastText(sentences, min_n=3, max_n=5, bucket=20000, **common) if kind == "ft" else Word2Vec(sentences, **common)
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(-0.05, 0.05, size=(len(itos), dim)).astype(np.float32)
    for i, w in enumerate(itos[2:], start=2):
        if kind == "ft" or w in model.wv.key_to_index:
            matrix[i] = model.wv[w]
    matrix[PAD] = 0.0
    return torch.from_numpy(matrix)


# ---------------------------------------------------------------------------
# Estimador
# ---------------------------------------------------------------------------
class NeuralTweetClassifier(md.DecisionMixin, BaseEstimator, ClassifierMixin):
    """Red neuronal multietiqueta: texto crudo -> HS, TR, AG (misma API que TweetToxicClassifier).

    - arch: 'nn_mlp', 'nn_fasttext', 'nn_cnn_word', 'nn_cnn_char', 'nn_bigru'.
    - head: 'shared' (una red con 3 salidas sigmoide y pérdida BCE conjunta) o 'independent' (3 redes).
    - pos_weight: pondera los positivos en la BCE por neg/pos de cada etiqueta.
    - init: 'random', 'w2v' o 'ft' (embeddings propios de gensim, solo arquitecturas de palabras).
    - hidden, dropout, lr, emb_dim, filters: None = valores de ARCH_DEFAULTS.
    - early stopping con validación interna (val_size) y paciencia; fixed_epochs entrena sin validación.
    - n_seeds: redes con semillas seed, seed+1, ... cuyas probabilidades se promedian.
    """

    def __init__(self, variant="V1", arch="nn_fasttext", head="shared", pos_weight=False, init="random",
                 hidden=None, dropout=None, lr=None, emb_dim=None, filters=None, weight_decay=1e-2,
                 batch_size=64, max_epochs=30, patience=5, val_size=0.12, n_seeds=3, fixed_epochs=None,
                 seed=SEED, n_threads=2, thresholds=0.5, hierarchy=True, empty_as_negative=True):
        self.variant = variant
        self.arch = arch
        self.head = head
        self.pos_weight = pos_weight
        self.init = init
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.emb_dim = emb_dim
        self.filters = filters
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_size = val_size
        self.n_seeds = n_seeds
        self.fixed_epochs = fixed_epochs
        self.seed = seed
        self.n_threads = n_threads
        self.thresholds = thresholds
        self.hierarchy = hierarchy
        self.empty_as_negative = empty_as_negative

    # --- compatibilidad con el registro de modelos ---
    @property
    def classifier(self) -> str:
        return self.arch

    @property
    def features(self) -> str:
        return f"{self.head}|{'pos_weight' if self.pos_weight else 'bce'}|init={self.init}"

    @property
    def clf_params(self) -> dict:
        p = dict(ARCH_DEFAULTS[self.arch])
        p.update({k: v for k, v in dict(hidden=self.hidden, dropout=self.dropout, lr=self.lr, emb_dim=self.emb_dim,
                                        filters=self.filters).items() if v is not None and k in p})
        p.update(head=self.head, pos_weight=self.pos_weight, init=self.init, weight_decay=self.weight_decay,
                 batch_size=self.batch_size, n_seeds=self.n_seeds)
        return p

    def _hparams(self) -> dict:
        return {k: v for k, v in self.clf_params.items() if k in ARCH_DEFAULTS[self.arch]}

    # --- red y entrenamiento ---
    def _new_net(self, n_out: int, pretrained=None) -> nn.Module:
        hp = self._hparams()
        size = {"n_inputs": self.encoder_.n_inputs} if self.arch == "nn_mlp" else {"n_vocab": self.encoder_.n_vocab}
        return NETS[self.arch](n_out=n_out, pretrained=pretrained, **size, **{k: v for k, v in hp.items() if k != "lr"
                                                                               and k != "buckets"})

    def _batch(self, inputs, idx):
        if sp.issparse(inputs):
            return torch.from_numpy(inputs[idx].toarray())
        return inputs[torch.as_tensor(idx)]

    def _loss_fn(self, y: np.ndarray):
        if not self.pos_weight:
            return nn.BCEWithLogitsLoss()
        pos = y.sum(0)
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y) - pos) / np.maximum(pos, 1), dtype=torch.float32))

    def _train_net(self, net, inputs, y, val_inputs, val_y, seed):
        set_seed(seed)
        loss_fn = self._loss_fn(y)
        opt = torch.optim.AdamW(net.parameters(), lr=self._hparams()["lr"], weight_decay=self.weight_decay)
        yt = torch.from_numpy(y.astype(np.float32))
        epochs = self.fixed_epochs or self.max_epochs
        gen = torch.Generator().manual_seed(seed)
        curve = {"train": [], "val": []}
        best, best_loss, best_epoch, bad = None, np.inf, epochs, 0
        n = y.shape[0]
        for epoch in range(1, epochs + 1):
            net.train()
            order = torch.randperm(n, generator=gen).numpy()
            total = 0.0
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(net(self._batch(inputs, idx)), yt[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(idx)
            curve["train"].append(total / n)
            if val_inputs is None:
                continue
            net.eval()
            with torch.no_grad():
                vloss = loss_fn(self._logits(net, val_inputs), torch.from_numpy(val_y.astype(np.float32))).item()
            curve["val"].append(vloss)
            if vloss < best_loss - 1e-4:
                best, best_loss, best_epoch, bad = copy.deepcopy(net.state_dict()), vloss, epoch, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if val_inputs is not None:
            net.load_state_dict(best)
        return net, best_epoch, curve

    def _logits(self, net, inputs, batch=256):
        n = inputs.shape[0]
        return torch.cat([net(self._batch(inputs, np.arange(s, min(s + batch, n)))) for s in range(0, n, batch)])

    def fit(self, X, y):
        torch.set_num_threads(self.n_threads)
        torch.use_deterministic_algorithms(True, warn_only=True)
        start = time.perf_counter()
        texts = pp._as_text_series(X)
        Y = np.asarray(y).astype(np.float32)
        self.prep_ = pp.TweetPreprocessor(self.variant).fit(texts)
        views = self.prep_.transform(texts)
        self.encoder_ = Encoder(self.arch, self.prep_.variant_, ARCH_DEFAULTS[self.arch].get("buckets", 20000)).fit(views)
        inputs = self.encoder_.transform(views)

        if self.fixed_epochs is None:  # validación interna estratificada por combinación de etiquetas
            combo = ["".join(str(int(v)) for v in row) for row in Y]
            tr_idx, va_idx = next(StratifiedShuffleSplit(1, test_size=self.val_size, random_state=self.seed)
                                  .split(np.zeros(len(Y)), combo))
        else:
            tr_idx, va_idx = np.arange(len(Y)), None
        self.train_idx_, self.val_idx_ = tr_idx, va_idx  # posiciones dentro de los datos de fit()

        pretrained = None
        if self.init in ("w2v", "ft"):
            if self.arch not in WORD_ARCHS:
                raise ValueError(f"init={self.init} solo aplica a {sorted(WORD_ARCHS)}")
            sentences = [tokenize(t) for t in views[self.encoder_.word_source]]
            pretrained = pretrain_vectors(sentences, self.encoder_.itos, self.init, self._hparams()["emb_dim"], self.seed)

        def sub(a, idx):
            return None if idx is None else (a[idx] if sp.issparse(a) else a[torch.as_tensor(idx)])

        targets = [list(range(3))] if self.head == "shared" else [[j] for j in range(3)]
        self.states_, self.best_epochs_, self.curves_ = [], [], []
        for s in range(self.n_seeds):
            seed_states, seed_epochs, seed_curves = [], [], []
            for cols in targets:
                set_seed(self.seed + 1000 * s + cols[0])
                net = self._new_net(len(cols), pretrained)
                net, best_epoch, curve = self._train_net(net, sub(inputs, tr_idx), Y[tr_idx][:, cols], sub(inputs, va_idx),
                                                         None if va_idx is None else Y[va_idx][:, cols],
                                                         self.seed + 1000 * s + cols[0])
                seed_states.append({k: v.detach().clone() for k, v in net.state_dict().items()})
                seed_epochs.append(best_epoch)
                seed_curves.append(curve)
            self.states_.append(seed_states)
            self.best_epochs_.append(seed_epochs)
            self.curves_.append(seed_curves)
        self.train_seconds_ = time.perf_counter() - start
        self.labels_ = list(LABELS)
        self._nets = None
        return self

    def _networks(self):
        if getattr(self, "_nets", None) is None:
            targets = [3] if self.head == "shared" else [1, 1, 1]
            self._nets = []
            for seed_states in self.states_:
                nets = []
                for n_out, state in zip(targets, seed_states):
                    net = self._new_net(n_out)
                    net.load_state_dict(state)
                    nets.append(net.eval())
                self._nets.append(nets)
        return self._nets

    def predict_proba_per_seed(self, X) -> np.ndarray:
        """Probabilidades de cada semilla: (n_seeds, n, 3)."""
        check_is_fitted(self, "states_")
        inputs = self.encoder_.transform(self.prep_.transform(pp._as_text_series(X)))
        out = []
        with torch.no_grad():
            for nets in self._networks():
                out.append(torch.cat([torch.sigmoid(self._logits(net, inputs)) for net in nets], dim=1).numpy())
        return np.stack(out)

    def predict_proba(self, X) -> np.ndarray:
        """Media de las probabilidades de las n_seeds redes: (n, 3) en el orden HS, TR, AG."""
        return self.predict_proba_per_seed(X).mean(axis=0)

    def attention(self, X) -> list[list[tuple[str, float]]]:
        """Pesos de atención (media de las semillas, cabeza compartida) por palabra. Solo nn_bigru."""
        if self.arch != "nn_bigru" or self.head != "shared":
            raise ValueError("La atención solo está disponible en nn_bigru con cabeza compartida")
        views = self.prep_.transform(pp._as_text_series(X))
        inputs = self.encoder_.transform(views)
        with torch.no_grad():
            weights = np.mean([net(inputs, return_attention=True)[1].numpy() for (net,) in self._networks()], axis=0)
        out = []
        for i, t in enumerate(views[self.encoder_.word_source]):
            toks = tokenize(t)[:MAX_WORDS] or ["<unk>"]
            out.append([(w if w in self.encoder_.stoi else f"{w}→<unk>", float(weights[i, k])) for k, w in enumerate(toks)])
        return out


# ---------------------------------------------------------------------------
# Validación cruzada
# ---------------------------------------------------------------------------
def _fit_fold(model, texts, Y, trn, val, keep_model):
    fitted = md.clone(model).fit(texts[trn], Y[trn])
    per_seed = fitted.predict_proba_per_seed(texts[val])
    train_proba = fitted.predict_proba(texts[trn])  # dentro de la muestra (para medir sobreajuste)
    return (per_seed, fitted.best_epochs_, fitted.curves_, fitted.train_seconds_, (fitted if keep_model else None),
            train_proba, fitted.train_idx_, fitted.val_idx_)


def neural_cross_validate(model: NeuralTweetClassifier, texts: pd.Series, Y: np.ndarray, folds: pd.Series,
                          n_jobs: int = 5, keep_models: bool = False) -> dict:
    """CV con los folds compartidos. Cada fold se entrena en un proceso (n_threads hilos de torch).

    Devuelve oof (media de semillas), oof_seeds (n_seeds, n, 3), at_05/tuned (como md.cross_validate),
    tuned_per_seed (métricas de cada semilla por separado), best_epochs, curves (del fold 0), tiempo y,
    si keep_models, los modelos de cada fold. Para medir el sobreajuste, también:
    - train: métricas por fold sobre la parte del fold con la que se entrenaron los pesos (88%);
    - inner_val: métricas por fold sobre la validación interna (12%) que eligió la época;
    ambas con el mismo umbral y jerarquía que el fold de validación externo."""
    texts = pd.Series(texts).reset_index(drop=True)
    folds = pd.Series(folds).reset_index(drop=True)
    ks = sorted(folds.unique())
    start = time.perf_counter()
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_fold)(model, texts, Y, (folds != k).to_numpy(), (folds == k).to_numpy(), keep_models) for k in ks)
    return _assemble_cv(model, results, ks, folds, Y, start, keep_models)


def _assemble_cv(model, results, ks, folds, Y, start, keep_models) -> dict:
    """Junta los resultados por fold (tuplas de _fit_fold) en el diccionario de la CV."""
    oof_seeds = np.zeros((model.n_seeds,) + Y.shape)
    for k, (per_seed, *_rest) in zip(ks, results):
        oof_seeds[:, (folds == k).to_numpy()] = per_seed
    oof = oof_seeds.mean(axis=0)
    at_05, tuned = md._summarize_folds(oof, Y, folds)
    per_seed_tuned = [md._summarize_folds(oof_seeds[s], Y, folds)[1] for s in range(model.n_seeds)]
    train_rows, inner_rows = [], []
    for k, r in zip(ks, results):
        y_trn = Y[(folds != k).to_numpy()]
        thr = md.fold_thresholds(oof, Y, folds, k)
        train_rows.append(md.decision_metrics(y_trn[r[6]], r[5][r[6]], thr))
        inner_rows.append(md.decision_metrics(y_trn[r[7]], r[5][r[7]], thr))
    return {"oof": oof, "oof_seeds": oof_seeds, "at_05": at_05, "tuned": tuned, "tuned_per_seed": per_seed_tuned,
            "train": pd.DataFrame(train_rows), "inner_val": pd.DataFrame(inner_rows),
            "best_epochs": [r[1] for r in results], "curves": results[0][2], "fold_train_seconds": [r[3] for r in results],
            "fit_seconds": time.perf_counter() - start, "models": [r[4] for r in results] if keep_models else None}


def final_epochs(cv: dict) -> int:
    """Épocas del modelo final = mediana de las mejores épocas de todos los folds y semillas."""
    return max(1, int(np.median([e for fold in cv["best_epochs"] for seed in fold for e in seed])))


# ---------------------------------------------------------------------------
# Guardado y carga (.pt sin pickle de clases)
# ---------------------------------------------------------------------------
def _checkpoint(model: NeuralTweetClassifier) -> dict:
    import gensim
    import sklearn

    params = {k: v for k, v in model.get_params().items()}
    return {"format": FORMAT, "params": params,
            "restore_dict": model.prep_.restore_dict_, "encoder": model.encoder_.to_dict(),
            "states": model.states_, "best_epochs": model.best_epochs_,
            "versions": {"torch": str(torch.__version__), "python": platform.python_version(),
                         "scikit-learn": sklearn.__version__, "gensim": gensim.__version__}}


def save_neural(model: NeuralTweetClassifier, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint(model), path)
    return path


def neural_size_mb(model: NeuralTweetClassifier) -> float:
    buffer = io.BytesIO()
    torch.save(_checkpoint(model), buffer)
    return buffer.tell() / 1e6


def load_neural(path):
    """Carga un .pt de neural.py: NeuralTweetClassifier (05) o TransformerTweetClassifier (06), según su formato."""
    ck = torch.load(path, weights_only=True)
    if ck.get("format") == "tweettoxic-transformer-v1":
        return _load_transformer(ck)
    if ck.get("format") != FORMAT:
        raise ValueError(f"{path} no es un modelo {FORMAT}")
    model = NeuralTweetClassifier(**ck["params"])
    prep = pp.TweetPreprocessor(model.variant)
    prep.variant_ = prep._resolve_variant()
    prep.restore_dict_ = ck["restore_dict"]
    model.prep_ = prep
    model.encoder_ = Encoder.from_dict(ck["encoder"])
    model.states_ = ck["states"]
    model.best_epochs_ = ck["best_epochs"]
    model.labels_ = list(LABELS)
    model._nets = None
    return model


def save_neural_model(model: NeuralTweetClassifier, name: str, metadata: dict | None = None,
                      models_dir: Path = md.MODELS_DIR, notebook: str | None = "05") -> Path:
    """Guarda models/<carpeta del notebook>/<name>.pt y .json (metadatos legibles, como save_model del 04)."""
    import datetime as dt
    import json

    path = save_neural(model, md.model_folder(notebook or (metadata or {}).get("notebook"), models_dir) / f"{name}.pt")
    meta = {"name": name, "created": dt.datetime.now().isoformat(timespec="seconds"), "variant": model.variant,
            "variant_steps": pp.VARIANTS[model.variant].steps(), "architecture": model.arch,
            "family": "neuronal", "params": model.clf_params, "fixed_epochs": model.fixed_epochs,
            "thresholds": model.thresholds, "hierarchy": model.hierarchy, "empty_as_negative": model.empty_as_negative,
            "labels": LABELS, "environment": md.environment_info() | {"torch": str(torch.__version__)},
            **(metadata or {})}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    return path


# ===========================================================================
# 06 · Transformer (encoder) desde cero
# ===========================================================================
# Nada preentrenado: la arquitectura es torch.nn con pesos aleatorios; el tokenizador (caracteres o
# BPE) y el preentrenamiento MLM opcional se ajustan en fit(), con el texto del fold de entrenamiento.
TRF_FORMAT = "tweettoxic-transformer-v1"
SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[MASK]", "<USER>", "<URL>"]
PAD_ID, UNK_ID, CLS_ID, MASK_ID = 0, 1, 2, 3
SPECIAL_RX = re.compile(r"(<USER>|<URL>)")
MAX_LEN_CAP = 310


def char_noise(text: str, p: float, rng: np.random.Generator) -> str:
    """Aumentación: con probabilidad p por carácter, lo borra, lo intercambia con el siguiente o lo sustituye
    por otra letra. Nunca toca los tokens <USER> y <URL>."""
    out = []
    for seg in SPECIAL_RX.split(text):
        if SPECIAL_RX.fullmatch(seg) or not seg:
            out.append(seg)
            continue
        chars, i = list(seg), 0
        res = []
        while i < len(chars):
            c = chars[i]
            if rng.random() < p:
                op = rng.integers(3)
                if op == 0:                      # borrar
                    i += 1
                    continue
                if op == 1 and i + 1 < len(chars):  # intercambiar con el siguiente
                    res += [chars[i + 1], c]
                    i += 2
                    continue
                res.append("abcdefghijklmnopqrstuvwxyz"[rng.integers(26)])  # sustituir
                i += 1
                continue
            res.append(c)
            i += 1
        out.append("".join(res))
    return "".join(out)


class TextTokenizer:
    """Tokenizador del 06, ajustado con los textos del fold de entrenamiento.

    - kind='char': un token por carácter (vocabulario = caracteres vistos en el fold) + <USER>/<URL> atómicos.
    - kind='bpe':  BPE entrenado con la librería `tokenizers` (solo su entrenador; nada descargado), con
      vocab_size tokens y <USER>/<URL> como tokens especiales atómicos.
    La longitud máxima es el percentil 99 de las longitudes del fold (tope MAX_LEN_CAP), más el token [CLS].
    """

    def __init__(self, kind: str = "char", vocab_size: int = 800):
        self.kind = kind
        self.vocab_size = vocab_size

    def _raw_ids(self, texts: list[str]) -> list[list[int]]:
        if self.kind == "char":
            out = []
            for t in texts:
                ids = []
                for seg in SPECIAL_RX.split(t):
                    if SPECIAL_RX.fullmatch(seg):
                        ids.append(self.stoi[seg])
                    else:
                        ids += [self.stoi.get(c, UNK_ID) for c in seg]
                out.append(ids)
            return out
        return [e.ids for e in self.bpe_.encode_batch(list(texts))]

    def fit(self, texts) -> "TextTokenizer":
        texts = list(texts)
        if self.kind == "char":
            chars = sorted({c for t in texts for seg in SPECIAL_RX.split(t) if not SPECIAL_RX.fullmatch(seg) for c in seg})
            self.itos = SPECIALS + chars
            self.stoi = {w: i for i, w in enumerate(self.itos)}
        else:
            from tokenizers import Tokenizer, models, pre_tokenizers, trainers

            bpe = Tokenizer(models.BPE(unk_token="[UNK]"))
            bpe.pre_tokenizer = pre_tokenizers.Metaspace()
            trainer = trainers.BpeTrainer(vocab_size=self.vocab_size, special_tokens=SPECIALS, min_frequency=2,
                                          show_progress=False)
            bpe.train_from_iterator(texts, trainer)
            assert [bpe.token_to_id(t) for t in SPECIALS[:4]] == [PAD_ID, UNK_ID, CLS_ID, MASK_ID]
            self.bpe_ = bpe
        lengths = [len(ids) for ids in self._raw_ids(texts)]
        self.max_len = int(min(MAX_LEN_CAP, max(8, np.percentile(lengths, 99))))
        return self

    @property
    def n_vocab(self) -> int:
        return len(self.itos) if self.kind == "char" else self.bpe_.get_vocab_size()

    def id_to_token(self, i: int) -> str:
        return self.itos[i] if self.kind == "char" else (self.bpe_.id_to_token(int(i)) or "[UNK]")

    def encode(self, texts) -> torch.Tensor:
        """Textos -> tensor (n, max_len + 1) con [CLS] al principio y [PAD] al final."""
        seqs = self._raw_ids(list(texts))
        out = torch.full((len(seqs), self.max_len + 1), PAD_ID, dtype=torch.long)
        out[:, 0] = CLS_ID
        for i, ids in enumerate(seqs):
            ids = ids[:self.max_len]
            out[i, 1:len(ids) + 1] = torch.tensor(ids, dtype=torch.long) if ids else torch.empty(0, dtype=torch.long)
        return out

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "vocab_size": self.vocab_size, "max_len": self.max_len}
        if self.kind == "char":
            d["itos"] = self.itos
        else:
            d["bpe_json"] = self.bpe_.to_str()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TextTokenizer":
        tok = cls(d["kind"], d["vocab_size"])
        tok.max_len = d["max_len"]
        if tok.kind == "char":
            tok.itos = list(d["itos"])
            tok.stoi = {w: i for i, w in enumerate(tok.itos)}
        else:
            from tokenizers import Tokenizer

            tok.bpe_ = Tokenizer.from_str(d["bpe_json"])  # tokenizador propio guardado, no descargado
        return tok


class TransformerNet(nn.Module):
    """Encoder Transformer pequeño: embeddings de token + posiciones aprendidas, n_layers capas pre-LayerNorm
    con GELU, pooling (media con máscara o [CLS]) y cabeza lineal de 3 salidas. mlm_head solo se usa en el
    preentrenamiento MLM."""

    def __init__(self, n_vocab, max_len, n_out=3, d_model=128, n_layers=2, n_heads=4, dropout=0.1, pooling="mean"):
        super().__init__()
        self.pooling = pooling
        self.tok = nn.Embedding(n_vocab, d_model, padding_idx=PAD_ID)
        self.pos = nn.Embedding(max_len + 1, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2 * d_model, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_out)
        self.mlm_head = nn.Linear(d_model, n_vocab)

    def _embed(self, x):
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        return self.drop(self.tok(x) + self.pos(positions))

    def encode(self, x):
        mask = x == PAD_ID
        return self.norm(self.encoder(self._embed(x), src_key_padding_mask=mask)), mask

    def pool(self, h, mask):
        if self.pooling == "cls":
            return h[:, 0]
        keep = (~mask).unsqueeze(-1).float()
        return (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)

    def forward(self, x):
        h, mask = self.encode(x)
        return self.head(self.drop(self.pool(h, mask)))

    def mlm_logits(self, x):
        return self.mlm_head(self.encode(x)[0])

    def attention_weights(self, x) -> torch.Tensor:
        """Atención de la última capa (media de las cabezas): fila del [CLS] con pooling 'cls', o atención
        media recibida por cada token con pooling 'mean'. Devuelve (B, L)."""
        mask = x == PAD_ID
        h = self._embed(x)
        for layer in self.encoder.layers[:-1]:
            h = layer(h, src_key_padding_mask=mask)
        last = self.encoder.layers[-1]
        z = last.norm1(h)
        _, w = last.self_attn(z, z, z, key_padding_mask=mask, need_weights=True, average_attn_weights=True)
        if self.pooling == "cls":
            return w[:, 0]
        valid = (~mask).float()
        return (w * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp(min=1.0)


def _schedule(opt, total_steps: int, warmup: float):
    """Warmup lineal durante la fracción `warmup` de los pasos y decaimiento lineal hasta 0 al final."""
    warm = max(1, int(warmup * total_steps))

    def factor(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        return max(0.0, (total_steps - step) / max(1, total_steps - warm))

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


class TransformerTweetClassifier(md.DecisionMixin, BaseEstimator, ClassifierMixin):
    """Encoder Transformer desde cero: texto crudo -> HS, TR, AG (misma API que NeuralTweetClassifier).

    - tokenization: 'char', 'bpe800' o 'bpe2000' (BPE entrenado en el fold con ~800 o ~2000 tokens).
    - d_model, n_layers, n_heads, dropout (también el de atención), pooling ('mean' o 'cls').
    - augment: ruido de caracteres (aug_char_p) y token dropout a [UNK] (aug_token_p), solo en entrenamiento.
    - mlm_epochs: épocas de preentrenamiento MLM (15% de tokens enmascarados) con los textos del fold, sin
      etiquetas, antes del ajuste con etiquetas (0 = sin preentrenar).
    - AdamW con warmup (fracción `warmup`) y decaimiento lineal a lo largo de max_epochs; clipping 1.0.
    - early stopping con validación interna (val_size) o épocas fijas (fixed_epochs, con el mismo calendario
      de tasa de aprendizaje truncado); n_seeds redes promediadas.
    """

    def __init__(self, variant="V1", tokenization="char", d_model=128, n_layers=2, n_heads=4, dropout=0.1,
                 pooling="mean", augment=False, aug_char_p=0.05, aug_token_p=0.1, mlm_epochs=0, mlm_prob=0.15,
                 lr=5e-4, weight_decay=0.02, warmup=0.1, batch_size=32, max_epochs=40, patience=6, val_size=0.12,
                 n_seeds=3, fixed_epochs=None, seed=SEED, n_threads=2, log_path=None, thresholds=0.5,
                 hierarchy=True, empty_as_negative=True):
        self.variant = variant
        self.tokenization = tokenization
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.pooling = pooling
        self.augment = augment
        self.aug_char_p = aug_char_p
        self.aug_token_p = aug_token_p
        self.mlm_epochs = mlm_epochs
        self.mlm_prob = mlm_prob
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup = warmup
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_size = val_size
        self.n_seeds = n_seeds
        self.fixed_epochs = fixed_epochs
        self.seed = seed
        self.n_threads = n_threads
        self.log_path = log_path
        self.thresholds = thresholds
        self.hierarchy = hierarchy
        self.empty_as_negative = empty_as_negative

    # --- compatibilidad con el registro ---
    @property
    def classifier(self) -> str:
        return "nn_trf_char" if self.tokenization == "char" else "nn_trf_bpe"

    @property
    def arch(self) -> str:
        return self.classifier

    @property
    def features(self) -> str:
        return (f"{self.tokenization}|d{self.d_model}-l{self.n_layers}|{self.pooling}|drop{self.dropout}"
                f"|aug={self.augment}|mlm={self.mlm_epochs}")

    @property
    def clf_params(self) -> dict:
        return {k: getattr(self, k) for k in ("tokenization", "d_model", "n_layers", "n_heads", "dropout", "pooling",
                                              "augment", "mlm_epochs", "lr", "weight_decay", "warmup", "batch_size",
                                              "n_seeds")}

    def _log(self, msg: str) -> None:
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} [{self.variant} {self.features}] {msg}\n")

    def _new_net(self) -> TransformerNet:
        return TransformerNet(self.tokenizer_.n_vocab, self.tokenizer_.max_len, 3, self.d_model, self.n_layers,
                              self.n_heads, self.dropout, self.pooling)

    def _encode_train(self, texts: list[str], rng: np.random.Generator) -> torch.Tensor:
        if not self.augment:
            return self.tokenizer_.encode(texts)
        ids = self.tokenizer_.encode([char_noise(t, self.aug_char_p, rng) for t in texts])
        drop = (torch.from_numpy(rng.random(ids.shape)) < self.aug_token_p) & (ids >= len(SPECIALS))
        return ids.masked_fill(drop, UNK_ID)

    def _pretrain_mlm(self, net, texts: list[str], seed: int) -> list[float]:
        """MLM: enmascara el 15% de los tokens (80% [MASK], 10% token al azar, 10% sin cambio) y aprende a
        predecirlos. Sin etiquetas: solo los textos del fold de entrenamiento."""
        set_seed(seed)
        ids_all = self.tokenizer_.encode(texts)
        n, steps_per_epoch = len(texts), int(np.ceil(len(texts) / self.batch_size))
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = _schedule(opt, self.mlm_epochs * steps_per_epoch, self.warmup)
        gen = torch.Generator().manual_seed(seed)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        curve = []
        for epoch in range(self.mlm_epochs):
            net.train()
            order = torch.randperm(n, generator=gen)
            total, count = 0.0, 0
            for start in range(0, n, self.batch_size):
                x = ids_all[order[start:start + self.batch_size]].clone()
                candidates = x >= len(SPECIALS) - 2  # tokens normales y <USER>/<URL>; nunca [PAD]/[CLS]
                chosen = (torch.rand(x.shape, generator=gen) < self.mlm_prob) & candidates
                labels = torch.where(chosen, x, torch.full_like(x, -100))
                r = torch.rand(x.shape, generator=gen)
                x[chosen & (r < 0.8)] = MASK_ID
                rand_tok = torch.randint(len(SPECIALS), self.tokenizer_.n_vocab, x.shape, generator=gen)
                swap = chosen & (r >= 0.8) & (r < 0.9)
                x[swap] = rand_tok[swap]
                if not chosen.any():
                    continue
                opt.zero_grad()
                logits = net.mlm_logits(x)
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sched.step()
                total += loss.item() * int(chosen.sum())
                count += int(chosen.sum())
            curve.append(total / max(count, 1))
        return curve

    def _train_net(self, net, texts, y, val_ids, val_y, seed):
        set_seed(seed)
        rng = np.random.default_rng(seed)
        loss_fn = nn.BCEWithLogitsLoss()
        n, steps_per_epoch = len(texts), int(np.ceil(len(texts) / self.batch_size))
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = _schedule(opt, self.max_epochs * steps_per_epoch, self.warmup)  # mismo calendario con épocas fijas
        yt = torch.from_numpy(y.astype(np.float32))
        gen = torch.Generator().manual_seed(seed)
        epochs = self.fixed_epochs or self.max_epochs
        curve = {"train": [], "val": []}
        best, best_loss, best_epoch, bad = None, np.inf, epochs, 0
        for epoch in range(1, epochs + 1):
            net.train()
            ids = self._encode_train(texts, rng)
            order = torch.randperm(n, generator=gen)
            total = 0.0
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(net(ids[idx]), yt[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sched.step()
                total += loss.item() * len(idx)
            curve["train"].append(total / n)
            if val_ids is None:
                continue
            net.eval()
            with torch.no_grad():
                vloss = loss_fn(self._logits(net, val_ids), torch.from_numpy(val_y.astype(np.float32))).item()
            curve["val"].append(vloss)
            if vloss < best_loss - 1e-4:
                best, best_loss, best_epoch, bad = copy.deepcopy(net.state_dict()), vloss, epoch, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if val_ids is not None:
            net.load_state_dict(best)
        return net, best_epoch, curve

    @staticmethod
    def _logits(net, ids, batch=256):
        return torch.cat([net(ids[s:s + batch]) for s in range(0, ids.shape[0], batch)])

    def fit(self, X, y):
        torch.set_num_threads(self.n_threads)
        torch.use_deterministic_algorithms(True, warn_only=True)
        start = time.perf_counter()
        raw = pp._as_text_series(X)
        Y = np.asarray(y).astype(np.float32)
        self.prep_ = pp.TweetPreprocessor(self.variant).fit(raw)
        texts = self.prep_.transform(raw)["text"].tolist()
        kind, vocab = ("char", 0) if self.tokenization == "char" else ("bpe", int(self.tokenization[3:]))
        self.tokenizer_ = TextTokenizer(kind, vocab).fit(texts)

        if self.fixed_epochs is None:
            combo = ["".join(str(int(v)) for v in row) for row in Y]
            tr_idx, va_idx = next(StratifiedShuffleSplit(1, test_size=self.val_size, random_state=self.seed)
                                  .split(np.zeros(len(Y)), combo))
        else:
            tr_idx, va_idx = np.arange(len(Y)), None
        self.train_idx_, self.val_idx_ = tr_idx, va_idx
        tr_texts = [texts[i] for i in tr_idx]
        val_ids = None if va_idx is None else self.tokenizer_.encode([texts[i] for i in va_idx])
        val_y = None if va_idx is None else Y[va_idx]

        self.states_, self.best_epochs_, self.curves_ = [], [], []
        for s in range(self.n_seeds):
            seed = self.seed + 1000 * s
            set_seed(seed)
            net = self._new_net()
            mlm_curve = self._pretrain_mlm(net, texts, seed) if self.mlm_epochs else []
            t0 = time.perf_counter()
            net, best_epoch, curve = self._train_net(net, tr_texts, Y[tr_idx], val_ids, val_y, seed)
            curve["mlm"] = mlm_curve
            self.states_.append([{k: v.detach().clone() for k, v in net.state_dict().items()}])
            self.best_epochs_.append([best_epoch])
            self.curves_.append([curve])
            self._log(f"semilla {s}: {len(curve['train'])} épocas, mejor {best_epoch}, "
                      f"val {min(curve['val']) if curve['val'] else float('nan'):.4f}, {time.perf_counter() - t0:.0f} s"
                      + (f", MLM {len(mlm_curve)} épocas (pérdida final {mlm_curve[-1]:.3f})" if mlm_curve else ""))
        self.train_seconds_ = time.perf_counter() - start
        self.labels_ = list(LABELS)
        self._nets = None
        return self

    def _networks(self):
        if getattr(self, "_nets", None) is None:
            self._nets = []
            for (state,) in self.states_:
                net = self._new_net()
                net.load_state_dict(state)
                self._nets.append(net.eval())
        return self._nets

    def predict_proba_per_seed(self, X) -> np.ndarray:
        check_is_fitted(self, "states_")
        ids = self.tokenizer_.encode(self.prep_.transform(pp._as_text_series(X))["text"].tolist())
        with torch.no_grad():
            return np.stack([torch.sigmoid(self._logits(net, ids)).numpy() for net in self._networks()])

    def predict_proba(self, X) -> np.ndarray:
        return self.predict_proba_per_seed(X).mean(axis=0)

    def attention(self, X) -> list[list[tuple[str, float]]]:
        """Peso de atención por token (última capa, media de cabezas y de semillas). No es una explicación causal."""
        texts = self.prep_.transform(pp._as_text_series(X))["text"].tolist()
        ids = self.tokenizer_.encode(texts)
        with torch.no_grad():
            w = np.mean([net.attention_weights(ids).numpy() for net in self._networks()], axis=0)
        out = []
        for i in range(len(texts)):
            row = [(self.tokenizer_.id_to_token(int(t)), float(w[i, j])) for j, t in enumerate(ids[i]) if t != PAD_ID]
            out.append(row)
        return out


def _trf_checkpoint(model: TransformerTweetClassifier) -> dict:
    import tokenizers

    return {"format": TRF_FORMAT, "params": model.get_params(), "restore_dict": model.prep_.restore_dict_,
            "tokenizer": model.tokenizer_.to_dict(), "states": model.states_, "best_epochs": model.best_epochs_,
            "train_seconds": float(getattr(model, "train_seconds_", 0.0)),
            "versions": {"torch": str(torch.__version__), "python": platform.python_version(),
                         "tokenizers": tokenizers.__version__}}


def save_transformer(model: TransformerTweetClassifier, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_trf_checkpoint(model), path)
    return path


def transformer_size_mb(model: TransformerTweetClassifier) -> float:
    buffer = io.BytesIO()
    torch.save(_trf_checkpoint(model), buffer)
    return buffer.tell() / 1e6


def _load_transformer(ck: dict) -> TransformerTweetClassifier:
    model = TransformerTweetClassifier(**ck["params"])
    prep = pp.TweetPreprocessor(model.variant)
    prep.variant_ = prep._resolve_variant()
    prep.restore_dict_ = ck["restore_dict"]
    model.prep_ = prep
    model.tokenizer_ = TextTokenizer.from_dict(ck["tokenizer"])
    model.states_ = ck["states"]
    model.best_epochs_ = ck["best_epochs"]
    model.train_seconds_ = ck.get("train_seconds", 0.0)
    model.labels_ = list(LABELS)
    model._nets = None
    return model


def save_transformer_model(model: TransformerTweetClassifier, name: str, metadata: dict | None = None,
                           models_dir: Path = md.MODELS_DIR, notebook: str = "06") -> Path:
    """Guarda models/06_transformer/<name>.pt y .json."""
    import datetime as dt
    import json

    path = save_transformer(model, md.model_folder(notebook, models_dir) / f"{name}.pt")
    meta = {"name": name, "created": dt.datetime.now().isoformat(timespec="seconds"), "variant": model.variant,
            "variant_steps": pp.VARIANTS[model.variant].steps(), "architecture": md.CLASSIFIER_NAMES[model.classifier],
            "family": "transformer", "params": model.clf_params, "fixed_epochs": model.fixed_epochs,
            "max_len": model.tokenizer_.max_len, "vocab": model.tokenizer_.n_vocab,
            "thresholds": model.thresholds, "hierarchy": model.hierarchy, "empty_as_negative": model.empty_as_negative,
            "labels": LABELS, "environment": md.environment_info() | {"torch": str(torch.__version__)},
            **(metadata or {})}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    return path


# ---------------------------------------------------------------------------
# CV con caché por fold (para poder reanudar sin repetir trabajo)
# ---------------------------------------------------------------------------
def _fit_fold_cached(model, texts, Y, trn, val, path: Path, keep_model: bool):
    result = _fit_fold(model, texts, Y, trn, val, keep_model)
    per_seed, best_epochs, curves, seconds, fitted, train_proba, tr_idx, va_idx = result
    payload = {"per_seed": torch.from_numpy(per_seed), "best_epochs": best_epochs, "curves": curves,
               "seconds": float(seconds), "train_proba": torch.from_numpy(train_proba),
               "train_idx": torch.from_numpy(np.asarray(tr_idx)), "val_idx": torch.from_numpy(np.asarray(va_idx))}
    if keep_model and fitted is not None:
        payload["model"] = _trf_checkpoint(fitted) if isinstance(fitted, TransformerTweetClassifier) else _checkpoint(fitted)
    torch.save(payload, path)
    return result


def _load_fold(path: Path, keep_model: bool):
    d = torch.load(path, weights_only=True)
    fitted = None
    if keep_model and "model" in d:
        fitted = _load_transformer(d["model"]) if d["model"]["format"] == TRF_FORMAT else load_neural_dict(d["model"])
    return (d["per_seed"].numpy(), d["best_epochs"], d["curves"], d["seconds"], fitted, d["train_proba"].numpy(),
            d["train_idx"].numpy(), d["val_idx"].numpy())


def cached_cross_validate(model, texts: pd.Series, Y: np.ndarray, folds: pd.Series, cache_dir, n_jobs: int = 5,
                          recompute: bool = False, keep_models: bool = False) -> dict:
    """Como neural_cross_validate, pero guarda cada fold en cache_dir/fold_<k>.pt en cuanto termina.

    Si la caché existe y la configuración coincide, reutiliza los folds ya calculados y solo entrena los que
    falten (recompute=True borra la caché y reentrena todo). Devuelve el mismo diccionario, con
    'cached_folds' = número de folds leídos de disco."""
    import json

    cache_dir = Path(cache_dir)
    config = json.dumps({k: v for k, v in model.get_params().items() if k not in ("n_threads", "log_path")},
                        sort_keys=True, default=str)
    cfg_file = cache_dir / "config.json"
    if recompute or (cfg_file.exists() and cfg_file.read_text() != config):
        for f in cache_dir.glob("fold_*.pt"):
            f.unlink()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg_file.write_text(config)
    texts = pd.Series(texts).reset_index(drop=True)
    folds = pd.Series(folds).reset_index(drop=True)
    ks = sorted(folds.unique())
    paths = {k: cache_dir / f"fold_{k}.pt" for k in ks}
    missing = [k for k in ks if not paths[k].exists()]
    start = time.perf_counter()
    computed = dict(zip(missing, Parallel(n_jobs=n_jobs)(
        delayed(_fit_fold_cached)(model, texts, Y, (folds != k).to_numpy(), (folds == k).to_numpy(), paths[k],
                                  keep_models) for k in missing)))
    results = [computed[k] if k in computed else _load_fold(paths[k], keep_models) for k in ks]
    out = _assemble_cv(model, results, ks, folds, Y, start, keep_models)
    out["cached_folds"] = len(ks) - len(missing)
    out["fold_train_seconds"] = [r[3] for r in results]
    return out


def load_neural_dict(ck: dict) -> NeuralTweetClassifier:
    """load_neural a partir de un checkpoint ya leído."""
    model = NeuralTweetClassifier(**ck["params"])
    prep = pp.TweetPreprocessor(model.variant)
    prep.variant_ = prep._resolve_variant()
    prep.restore_dict_ = ck["restore_dict"]
    model.prep_ = prep
    model.encoder_ = Encoder.from_dict(ck["encoder"])
    model.states_ = ck["states"]
    model.best_epochs_ = ck["best_epochs"]
    model.labels_ = list(LABELS)
    model._nets = None
    return model
