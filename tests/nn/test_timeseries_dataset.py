import time
import torch
from torch.utils.data import DataLoader

from bnode_core.nn.nn_utils.load_data import (
    TimeSeriesDataset,
    LegacyTimeSeriesDataset,
    timeseries_collate_fn,
)


def _build_dummy_datasets(N: int = 3, T: int = 10, seq_len: int = 4, stride: int = 1):
    """Create matching LegacyTimeSeriesDataset and TimeSeriesDataset on random data."""
    torch.manual_seed(0)

    # base time vector (shared across samples)
    time_base = torch.arange(T, dtype=torch.float32)
    time = time_base.unsqueeze(0).expand(N, -1).unsqueeze(1)  # (N, 1, T)

    states = torch.randn(N, 25, T)
    states_der = torch.randn(N, 25, T)
    controls = torch.randn(N, 12, T)
    outputs = torch.randn(N, 13, T)
    parameters = torch.randn(N, 7)

    legacy = LegacyTimeSeriesDataset(
        seq_len,
        stride,
        time=time,
        states=states,
        states_der=states_der,
        controls=controls,
        outputs=outputs,
        parameters=parameters,
    )

    modern = TimeSeriesDataset(
        seq_len,
        stride,
        time=time,
        states=states,
        states_der=states_der,
        controls=controls,
        outputs=outputs,
        parameters=parameters,
    )

    return legacy, modern


def _build_dummy_modern_dataset_with_max_samples(N: int, T: int, seq_len: int, stride: int, max_samples: int):
    """Create only the modern TimeSeriesDataset with a max_samples constraint."""
    torch.manual_seed(0)

    time_base = torch.arange(T, dtype=torch.float32)
    time = time_base.unsqueeze(0).expand(N, -1).unsqueeze(1)

    states = torch.randn(N, 2, T)
    states_der = torch.randn(N, 2, T)
    controls = torch.randn(N, 1, T)
    outputs = torch.randn(N, 1, T)
    parameters = torch.randn(N, 1)

    modern = TimeSeriesDataset(
        seq_len,
        stride,
        max_samples=max_samples,
        time=time,
        states=states,
        states_der=states_der,
        controls=controls,
        outputs=outputs,
        parameters=parameters,
    )

    return modern


def _assert_sample_equal(sample_legacy: dict, sample_modern: dict, atol: float = 1e-6):
    assert sample_legacy.keys() == sample_modern.keys()
    for key in sample_legacy.keys():
        v_legacy = sample_legacy[key]
        v_modern = sample_modern[key]
        assert v_legacy.shape == v_modern.shape
        diff = (v_legacy - v_modern).abs().max().item() if v_legacy.numel() > 0 else 0.0
        assert diff <= atol, f"Mismatch for key {key}: max diff {diff} > {atol}"


def test_timeseries_dataset_len_matches_legacy():
    legacy, modern = _build_dummy_datasets()
    assert len(legacy) == len(modern)


def test_timeseries_dataset_getitem_matches_legacy():
    legacy, modern = _build_dummy_datasets()

    n = len(legacy)
    # test a subset of indices including boundaries
    indices = [0, 1, n // 2, n - 2, n - 1]
    indices = sorted(set(i for i in indices if 0 <= i < n))

    for idx in indices:
        sample_legacy = legacy[idx]
        sample_modern = modern[idx]
        _assert_sample_equal(sample_legacy, sample_modern)


def test_timeseries_dataset_getitems_matches_legacy():
    legacy, modern = _build_dummy_datasets()

    n = len(legacy)
    indices = [0, 1, 3, n - 3, n - 1]
    indices = sorted(set(i for i in indices if 0 <= i < n))

    # Legacy returns list-of-dicts
    legacy_samples = legacy.__getitems__(indices)
    assert isinstance(legacy_samples, list)
    assert len(legacy_samples) == len(indices)

    # Modern returns dict-of-batched-tensors
    modern_batch = modern.__getitems__(indices)
    assert isinstance(modern_batch, dict)

    # Convert legacy list-of-dicts to dict-of-batches and compare
    keys = legacy_samples[0].keys()
    assert keys == modern_batch.keys()

    for key in keys:
        stacked_legacy = torch.stack([s[key] for s in legacy_samples], dim=0)
        v_modern = modern_batch[key]
        assert stacked_legacy.shape == v_modern.shape
        diff = (stacked_legacy - v_modern).abs().max().item() if stacked_legacy.numel() > 0 else 0.0
        assert diff <= 1e-6, f"Mismatch in __getitems__ for key {key}: max diff {diff} > 1e-6"


def test_timeseries_dataset_stride_len_matches_legacy():
    legacy, modern = _build_dummy_datasets(T=11, seq_len=4, stride=3)
    assert len(legacy) == len(modern)


def test_timeseries_dataset_stride_getitem_matches_legacy():
    legacy, modern = _build_dummy_datasets(T=11, seq_len=4, stride=3)

    n = len(legacy)
    indices = [0, 1, n // 2, n - 2, n - 1]
    indices = sorted(set(i for i in indices if 0 <= i < n))

    for idx in indices:
        sample_legacy = legacy[idx]
        sample_modern = modern[idx]
        _assert_sample_equal(sample_legacy, sample_modern)


def test_timeseries_dataset_stride_getitems_matches_legacy():
    legacy, modern = _build_dummy_datasets(T=11, seq_len=4, stride=3)

    n = len(legacy)
    indices = [0, 1, 3, n - 3, n - 1]
    indices = sorted(set(i for i in indices if 0 <= i < n))

    legacy_samples = legacy.__getitems__(indices)
    assert isinstance(legacy_samples, list)
    assert len(legacy_samples) == len(indices)

    modern_batch = modern.__getitems__(indices)
    assert isinstance(modern_batch, dict)

    keys = legacy_samples[0].keys()
    assert keys == modern_batch.keys()

    for key in keys:
        stacked_legacy = torch.stack([s[key] for s in legacy_samples], dim=0)
        v_modern = modern_batch[key]
        assert stacked_legacy.shape == v_modern.shape
        diff = (stacked_legacy - v_modern).abs().max().item() if stacked_legacy.numel() > 0 else 0.0
        assert diff <= 1e-6, f"Mismatch in __getitems__ for key {key}: max diff {diff} > 1e-6"


def test_timeseries_dataset_max_samples_effective_stride_and_length():
    """Validate effective stride computation with max_samples constraint.

    For N=10, T=100, seq_len=5, initial stride=1 and max_samples=40, the
    effective stride should be increased such that the resulting number of
    samples (windows) does not exceed max_samples.
    """

    N, T, seq_len, stride, max_samples = 10, 100, 5, 1, 40

    modern = _build_dummy_modern_dataset_with_max_samples(
        N=N,
        T=T,
        seq_len=seq_len,
        stride=stride,
        max_samples=max_samples,
    )
    # check that we respect the limits
    assert len(modern) <= max_samples
    assert modern.stride == 24

    # also cross-check via the helper
    eff_stride = TimeSeriesDataset.calculate_effective_stride(
        N=N,
        T=T,
        seq_len=seq_len,
        stride=stride,
        max_samples=max_samples,
    )
    assert eff_stride == modern.stride


def test_timeseries_dataset_timing_comparison():
    """Print simple timing comparison between Legacy and new TimeSeriesDataset.

    This is not a strict performance test, it just reports relative timings
    for iterating once over all windows via __getitem__.
    
    Run this test in console by 

    uv run pytest -s bnode_core/tests/nn/test_timeseries_dataset.py::test_timeseries_dataset_timing_comparison

    """
    # Use a slightly larger dummy dataset to make timing measurable
    legacy, modern = _build_dummy_datasets(N=2048, T=8000, seq_len=10)

    n = 1024  # number of windows to iterate over; can adjust for longer/shorter timing

    # Warm-up
    _ = legacy[0]
    _ = modern[0]

    t0 = time.perf_counter()
    for i in range(n):
        _ = legacy[i]
    t_legacy = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(n):
        _ = modern[i]
    t_modern = time.perf_counter() - t0

    print(f"LegacyTimeSeriesDataset iteration time: {t_legacy:.6f} s for {n} windows")
    print(f"TimeSeriesDataset iteration time:      {t_modern:.6f} s for {n} windows")

    print(f"TimeSeriesDataset is {t_legacy / t_modern:.2f}x faster than LegacyTimeSeriesDataset in __getitem__ iteration")

    # Keep the test always passing; this is diagnostic only
    assert n > 0


def test_timeseries_dataset_getitems_timing_comparison():
    """Print timing comparison for batched __getitems__ between Legacy and new dataset.

    Iterates over the dataset in batches of indices, calling __getitems__ each time,
    and reports total time for LegacyTimeSeriesDataset vs TimeSeriesDataset.

    Run in console with e.g.

    uv run pytest -s bnode_core/tests/nn/test_timeseries_dataset.py::test_timeseries_dataset_getitems_timing_comparison
    """
    legacy, modern = _build_dummy_datasets(N=2048, T=8000, seq_len=10)

    n = 100
    batch_size = 800

    # Warm-up
    _ = legacy.__getitems__([0, 1, 2])
    _ = modern.__getitems__([0, 1, 2])

    # Legacy timing
    t0 = time.perf_counter()
    for start in range(0, n, batch_size):
        indices = list(range(start, min(start + batch_size, n)))
        _ = legacy.__getitems__(indices)
    t_legacy = time.perf_counter() - t0

    # Modern timing
    t0 = time.perf_counter()
    for start in range(0, n, batch_size):
        indices = list(range(start, min(start + batch_size, n)))
        _ = modern.__getitems__(indices)
    t_modern = time.perf_counter() - t0

    print(
        f"LegacyTimeSeriesDataset __getitems__ time: {t_legacy:.6f} s "
        f"for {n} windows (batch_size={batch_size})"
    )
    print(
        f"TimeSeriesDataset __getitems__ time:      {t_modern:.6f} s "
        f"for {n} windows (batch_size={batch_size})"
    )

    if t_modern > 0:
        print(
            f"TimeSeriesDataset is {t_legacy / t_modern:.2f}x faster than "
            f"LegacyTimeSeriesDataset in __getitems__ iteration"
        )

    # Diagnostic-only test
    assert n > 0

def time_iteration(dataloader, n_items):
    """Iterate over an existing DataLoader iterator n_items times.

    Important: we create the iterator only once so that multi-worker
    DataLoader startup/prefetch happens once, as in a normal training
    epoch, instead of re-starting a new epoch on every iteration.
    """
    t0 = time.perf_counter()
    it = iter(dataloader)
    for _ in range(n_items):
        try:
            _ = next(it)
            time.sleep(0.01)  # simulate some processing time per batch
        except StopIteration:
            break
    return time.perf_counter() - t0

def test_timeseries_dataset_dataloader_overhead():
    """Measure overhead of DataLoader vs direct dataset access.

    This is a diagnostic test to understand where time is spent:

    - A) Direct iteration over TimeSeriesDataset via __getitem__.
    - B) DataLoader with num_workers=0.
    - C) DataLoader with a more realistic multi-worker setup.

    Run in console with:

        uv run pytest -s bnode_core/tests/nn/test_timeseries_dataset.py::test_timeseries_dataset_dataloader_overhead
    """

    # Use a moderately large dummy dataset to make timings measurable

    print("Starting Test: TimeSeriesDataset DataLoader Overhead Comparison")
    N, T, seq_len = 1024, 8000, 10
    bs = 128

    torch.manual_seed(0)
    time_base = torch.arange(T, dtype=torch.float32)
    time_tensor = time_base.unsqueeze(0).expand(N, -1).unsqueeze(1)

    states = torch.randn(N, 25, T)
    states_der = torch.randn(N, 25, T)
    controls = torch.randn(N, 12, T)
    outputs = torch.randn(N, 13, T)
    parameters = torch.randn(N, 7)

    legacy_dataset = LegacyTimeSeriesDataset(
        seq_len,
        time=time_tensor,
        states=states,
        states_der=states_der,
        controls=controls,
        outputs=outputs,
        parameters=parameters,
    )

    modern_dataset = TimeSeriesDataset(
        seq_len,
        time=time_tensor,
        states=states,
        states_der=states_der,
        controls=controls,
        outputs=outputs,
        parameters=parameters,
    )

    # Limit number of items for quicker diagnostics (same for both datasets)
    n_items = min(250, len(legacy_dataset), len(modern_dataset))  # number of windows to iterate over for timing
    n_items_warmup = 24

    print(f"Dataset size: {len(legacy_dataset)} windows, using n_items={n_items} for timing and n_items_warmup={n_items_warmup} for warm-up. \n")

    # B) DataLoader, single worker
    loader0_legacy = DataLoader(
        legacy_dataset,
        batch_size=bs,
        num_workers=0,
        collate_fn=timeseries_collate_fn,
        shuffle=False,
    )
    print("created Legacy DataLoader with num_workers=0, starting iteration...")
    t_legacy_dl_0_warmup = time_iteration(loader0_legacy, n_items_warmup)
    print(f"Warm-up iteration time (Legacy DataLoader, num_workers=0): {t_legacy_dl_0_warmup:.6f} s")
    t_legacy_dl_0 = time_iteration(loader0_legacy, n_items)
    print(f"Timing iteration time (Legacy DataLoader, num_workers=0): {t_legacy_dl_0:.6f} s \n")
    del loader0_legacy  # free memory

    loader0_modern = DataLoader(
        modern_dataset,
        batch_size=bs,
        num_workers=0,
        collate_fn=timeseries_collate_fn,
        shuffle=False,
    )
    print("created Modern DataLoader with num_workers=0, starting iteration...")
    t_modern_dl_0_warmup = time_iteration(loader0_modern, n_items_warmup)
    print(f"Warm-up iteration time (Modern DataLoader, num_workers=0): {t_modern_dl_0_warmup:.6f} s")
    t_modern_dl_0 = time_iteration(loader0_modern, n_items)
    print(f"Timing iteration time (Modern DataLoader, num_workers=0): {t_modern_dl_0:.6f} s\n")
    del loader0_modern  # free memory

    print(f"Legacy DataLoader (num_workers=0) time: {t_legacy_dl_0:.6f} s")
    print(f"Modern DataLoader (num_workers=0) time: {t_modern_dl_0:.6f} s\n")

    # C) DataLoader, multi-worker (roughly mirroring trainer settings)
    n_workers = 2
    prefetch_factor = 2

    loaderN_legacy = DataLoader(
        legacy_dataset,
        batch_size=bs,
        num_workers=n_workers,
        collate_fn=timeseries_collate_fn,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        persistent_workers=True,
        shuffle=True,
    )
    print(f"created Legacy DataLoader with num_workers={n_workers}, prefetch_factor={prefetch_factor}, starting iteration...")  
    # t0 = time.perf_counter()
    # for i in range(n_items):
    #     _ = next(iter(loaderN_legacy))  # force loading the batch
    # t_legacy_dl_N = time.perf_counter() - t0  
    t_legacy_dl_N_warmup = time_iteration(loaderN_legacy, n_items_warmup)
    print(f"Warm-up iteration time (Legacy DataLoader, num_workers={n_workers}): {t_legacy_dl_N_warmup:.6f} s")
    t_legacy_dl_N = time_iteration(loaderN_legacy, n_items)
    print(f"Legacy DataLoader with num_workers={n_workers} iteration time: {t_legacy_dl_N:.6f} s\n")
    del loaderN_legacy  # free memory

    loaderN_modern = DataLoader(
        modern_dataset,
        batch_size=bs,
        num_workers=n_workers,
        collate_fn=timeseries_collate_fn,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        persistent_workers=True,
        shuffle=True,
    )
    print(f"created Modern DataLoader with num_workers={n_workers}, prefetch_factor={prefetch_factor}, starting iteration...")
    t_modern_dl_N_warmup = time_iteration(loaderN_modern, n_items_warmup)
    print(f"Warm-up iteration time (Modern DataLoader, num_workers={n_workers}): {t_modern_dl_N_warmup:.6f} s")
    t_modern_dl_N = time_iteration(loaderN_modern, n_items)
    print(f"Modern DataLoader with num_workers={n_workers} iteration time: {t_modern_dl_N:.6f} s\n")
    del loaderN_modern  # free memory

    print(
        f"Legacy DataLoader (num_workers={n_workers}, prefetch_factor={prefetch_factor}) time: "
        f"{t_legacy_dl_N:.6f} s"
    )
    print(
        f"Modern DataLoader (num_workers={n_workers}, prefetch_factor={prefetch_factor}) time: "
        f"{t_modern_dl_N:.6f} s"
    )

    # Diagnostic-only test
    assert n_items > 0

if __name__ == "__main__":
    test_timeseries_dataset_dataloader_overhead()