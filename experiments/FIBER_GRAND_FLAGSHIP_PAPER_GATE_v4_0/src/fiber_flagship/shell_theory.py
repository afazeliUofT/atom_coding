from __future__ import annotations

import math
from dataclasses import dataclass

from .channels import FixedDeletionBSC
from .utils import h2


@dataclass(frozen=True)
class ShellCertificateBound:
    actual_error_weight: int
    extra_shells: int
    certificate_shell: int
    stream_count: int
    history_upper_bound: int
    normalized_log2_bound: float
    asymptotic_h2_p: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "actual_error_weight": self.actual_error_weight,
            "extra_shells": self.extra_shells,
            "certificate_shell": self.certificate_shell,
            "stream_count": self.stream_count,
            "history_upper_bound": self.history_upper_bound,
            "normalized_log2_bound": self.normalized_log2_bound,
            "asymptotic_h2_p": self.asymptotic_h2_p,
        }


def shell_certificate_bound(channel: FixedDeletionBSC, actual_error_weight: int) -> ShellCertificateBound:
    m = channel.m
    streams = channel.stream_count
    if channel.p == 0.0:
        extra = 0
    else:
        a = (1.0 - channel.p) / channel.p
        extra = int(math.floor(math.log(streams, a)))
    shell = min(m, int(actual_error_weight) + extra)
    cumulative = sum(math.comb(m, i) for i in range(shell + 1))
    histories = (1 << channel.deletions) * streams * cumulative
    return ShellCertificateBound(
        actual_error_weight=int(actual_error_weight),
        extra_shells=extra,
        certificate_shell=shell,
        stream_count=streams,
        history_upper_bound=histories,
        normalized_log2_bound=math.log2(max(1, histories)) / channel.n,
        asymptotic_h2_p=h2(channel.p),
    )


def certificate_inequality_holds(channel: FixedDeletionBSC, actual_error_weight: int) -> bool:
    bound = shell_certificate_bound(channel, actual_error_weight)
    if channel.p == 0.0:
        return True
    e = actual_error_weight
    w = bound.certificate_shell
    q = 1.0 / channel.stream_count
    incumbent_component = q * (channel.p**e) * ((1.0 - channel.p) ** (channel.m - e))
    residual = (channel.p ** min(channel.m, w + 1)) * (
        (1.0 - channel.p) ** max(0, channel.m - min(channel.m, w + 1))
    )
    if w >= channel.m:
        residual = 0.0
    return incumbent_component > residual - 1e-15
