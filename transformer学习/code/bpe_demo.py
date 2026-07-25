"""一个可读的 BPE 训练玩具示例：只依赖 Python 标准库。"""
from __future__ import annotations

from collections import Counter


def pair_stats(vocab: dict[tuple[str, ...], int]) -> Counter[tuple[str, str]]:
    """统计加权语料中相邻符号对的频次。"""
    counts: Counter[tuple[str, str]] = Counter()
    for symbols, frequency in vocab.items():
        for left, right in zip(symbols, symbols[1:]):
            counts[left, right] += frequency
    return counts


def merge_pair(pair: tuple[str, str], vocab: dict[tuple[str, ...], int]) -> dict[tuple[str, ...], int]:
    """把每个出现的 (left, right) 合成一个新符号。"""
    merged = {}
    for symbols, frequency in vocab.items():
        out, i = [], 0
        while i < len(symbols):
            if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
                out.append("".join(pair))
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        merged[tuple(out)] = frequency
    return merged


def main() -> None:
    corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
    vocab = {tuple(word) + ("</w>",): freq for word, freq in corpus.items()}
    print("初始语料：")
    for symbols, freq in vocab.items():
        print(f"  {' '.join(symbols):<22} × {freq}")
    for step in range(1, 9):
        stats = pair_stats(vocab)
        if not stats:
            break
        best_pair, frequency = stats.most_common(1)[0]
        print(f"\n第 {step} 轮：合并 {best_pair[0]!r} + {best_pair[1]!r} (加权频次 {frequency})")
        vocab = merge_pair(best_pair, vocab)
        for symbols, freq in vocab.items():
            print(f"  {' '.join(symbols):<22} × {freq}")


if __name__ == "__main__":
    main()
