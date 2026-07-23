"""
Low-level AlphaGenome API plumbing shared by the calibration and deletion
modules: grpc retry wrappers and splice-track lookup. alphagenome/grpc are
imported lazily so this module imports without them installed.
"""

import time

# retries before giving up on transient grpc RpcErrors
PREDICT_MAX_ATTEMPTS = 5


def _with_rpc_retry(call, what):
    """
    ``call()`` with exponential backoff on ``grpc.RpcError``; re-raises after
    ``PREDICT_MAX_ATTEMPTS``. ``what`` labels the retry log line.
    """
    import grpc

    for attempt in range(1, PREDICT_MAX_ATTEMPTS + 1):
        try:
            return call()
        except grpc.RpcError as e:
            if attempt == PREDICT_MAX_ATTEMPTS:
                raise
            print(
                f"  {what} RpcError (attempt {attempt}/{PREDICT_MAX_ATTEMPTS}): "
                f"{e.code() if hasattr(e, 'code') else e}; retrying"
            )
            time.sleep(2 ** (attempt - 1))
    raise AssertionError(f"{what} retry loop exited without returning")  # unreachable


def predict_variants_with_retry(model, **kwargs):
    """``model.predict_variants`` with grpc retry (see :func:`_with_rpc_retry`)."""
    return _with_rpc_retry(lambda: model.predict_variants(**kwargs), "predict_variants")


def predict_interval_with_retry(model, **kwargs):
    """``model.predict_interval`` with grpc retry (see :func:`_with_rpc_retry`)."""
    return _with_rpc_retry(lambda: model.predict_interval(**kwargs), "predict_interval")


def find_strand_track(ss, st_type, strand):
    """Index of the ``st_type`` ("donor"/"acceptor") track on ``strand`` in ``ss``."""
    track_names = list(ss.metadata["name"])
    track_strands = list(ss.metadata["strand"])
    for t, (tn, ts) in enumerate(zip(track_names, track_strands)):
        if ts == strand and st_type in tn.lower():
            return t
    raise ValueError(
        f"No {st_type} track found for strand {strand}; "
        f"tracks={list(zip(track_names, track_strands))}"
    )
