"""Реализация трёх NIST-тестов согласно методичке (СГАУ, лаб. работа №2).

Тесты рассчитаны на последовательности фиксированной длины 128 бит
(блок длиной M=8 для теста на длинную серию единиц), как задано в
методических указаниях к лабораторной работе.
"""

import math
import sys
from scipy.special import erfc, gammaincc


def _validate_bits(bits: str) -> None:
    """Проверяет, что bits - непустая строка из '0' и '1'."""
    if not isinstance(bits, str):
        raise ValueError("bits должна быть строкой")
    if len(bits) == 0:
        raise ValueError("bits не может быть пустой строкой")
    if not all(c in ('0', '1') for c in bits):
        raise ValueError("bits должна содержать только символы '0' и '1'")


def monobit_test(bits: str) -> float:
    """
    Частотный побитовый тест.

    Каждый бит '1' интерпретируется как +1, '0' как -1. Вычисляется
    модуль отклонения суммы от нуля и через дополнительную функцию
    ошибок находится P-значение.

    Parameters
    ----------
    bits : str
        Бинарная строка из символов '0' и '1'.

    Returns
    -------
    float
        P-value. В случае ошибки возвращает -1.0.
    """
    try:
        _validate_bits(bits)
        n = len(bits)
        count_ones = bits.count('1')
        s_n = abs(2 * count_ones - n)
        p_value = erfc(s_n / math.sqrt(2 * n))
        return p_value
    except Exception as e:
        print(f"Ошибка в monobit_test: {e}", file=sys.stderr)
        return -1.0


def runs_test(bits: str) -> float:
    """
    Тест на одинаковые подряд идущие биты (тест прогонов).

    Сначала вычисляется доля единиц pi в последовательности.
    Если |pi - 0.5| >= tau, где tau = 2/sqrt(n), тест неприменим и
    P-значение считается равным нулю. Иначе подсчитывается число
    смен битов V (без добавления +1) и по нему вычисляется
    P-значение через дополнительную функцию ошибок; знаменатель
    формулы согласно методичке — 2*sqrt(2*n*pi*(1-pi)).

    Parameters
    ----------
    bits : str
        Бинарная строка.

    Returns
    -------
    float
        P-value. При ошибке возвращает -1.0; если условие
        применимости теста не выполнено — 0.0.
    """
    try:
        _validate_bits(bits)
        n = len(bits)
        count_ones = bits.count('1')
        pi = count_ones / n

        tau = 2 / math.sqrt(n)
        if abs(pi - 0.5) >= tau:
            return 0.0

        v = 0
        for i in range(n - 1):
            if bits[i] != bits[i + 1]:
                v += 1

        numerator = abs(v - 2 * n * pi * (1 - pi))
        denominator = 2 * math.sqrt(2 * n * pi * (1 - pi))
        if denominator == 0:
            return 0.0
        p_value = erfc(numerator / denominator)
        return p_value
    except Exception as e:
        print(f"Ошибка в runs_test: {e}", file=sys.stderr)
        return -1.0


def longest_run_ones_in_block(bits: str, m: int = 8) -> float:
    """
    Тест на самую длинную последовательность единиц в блоке.

    Согласно методичке, для последовательности длиной 128 бит
    последовательность разбивается на блоки длиной M=8, и по
    четырём категориям максимальной длины серии единиц в блоке
    (<=1, =2, =3, >=4) считается статистика хи-квадрат, а затем
    P-значение через неполную гамма-функцию.

    Parameters
    ----------
    bits : str
        Бинарная строка.
    m : int
        Длина блока (по умолчанию 8, как в методичке).

    Returns
    -------
    float
        P-value, или -1.0 если строка слишком короткая (меньше
        одного блока) или произошла ошибка.
    """
    try:
        _validate_bits(bits)
        n = len(bits)
        if n < m:
            return -1.0

        v = [1, 2, 3, 4]
        pi = [0.21484375, 0.36718750, 0.23046875, 0.18750000]

        num_blocks = n // m
        counts = [0] * len(pi)

        for block_idx in range(num_blocks):
            block = bits[block_idx * m:(block_idx + 1) * m]
            longest = 0
            cur = 0
            for bit in block:
                if bit == '1':
                    cur += 1
                    if cur > longest:
                        longest = cur
                else:
                    cur = 0

            if longest <= v[0]:
                counts[0] += 1
            elif longest >= v[-1]:
                counts[-1] += 1
            else:
                for i in range(len(v) - 1):
                    if v[i] < longest <= v[i + 1]:
                        counts[i + 1] += 1
                        break

        chi2 = 0.0
        for i in range(len(pi)):
            expected = num_blocks * pi[i]
            chi2 += (counts[i] - expected) ** 2 / expected

        degrees = len(pi) - 1
        p_value = gammaincc(degrees / 2.0, chi2 / 2.0)
        return p_value
    except Exception as e:
        print(f"Ошибка в longest_run_ones_in_block: {e}", file=sys.stderr)
        return -1.0
