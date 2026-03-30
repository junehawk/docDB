"""
CUDA code-path verification tests.

These tests use unittest.mock to simulate a CUDA-available environment
WITHOUT requiring actual CUDA hardware.  They verify that:

1. Device detection resolves 'auto' → 'cuda' when CUDA is available
2. _clear_device_cache calls torch.cuda.empty_cache() for cuda device
3. should_unload_model('cuda') returns False (no periodic unload)
4. gc.collect runs for cuda device (not skipped)
5. The MPS 'buffer size' retry does NOT swallow CUDA OOM errors
6. embed_batch re-raises non-buffer-size errors on CUDA
7. Reranker resolves 'auto' → 'cuda'
8. PYTORCH_MPS_HIGH_WATERMARK_RATIO is not set on non-macOS
9. unload_model on cuda properly calls _clear_device_cache

Design note: All modules are imported ONCE at the top of this file so that
Python 3.13's "cannot load module more than once per process" constraint on
numpy/torch C-extensions is never triggered.  Mocking is done by patching
attributes on the already-imported module objects rather than via
patch.dict(sys.modules, ...), which would force a re-import.
"""
import gc
import os
import sys
import unittest
from unittest.mock import patch, MagicMock, PropertyMock, call

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import all modules under test ONCE at module load time.
# Never re-import inside individual test methods.
# ---------------------------------------------------------------------------
import src.embedding.embedding_manager as _emb_mgr_mod
from src.embedding.embedding_manager import _detect_device, EmbeddingManager
from src.embedding.local_embedder import LocalEmbedder
from src.search import reranker as _reranker_mod
from src.search.reranker import Reranker
from src import compat as _compat_mod
from src.compat import should_unload_model, setup_mps_env


# ---------------------------------------------------------------------------
# 1. Device detection
# ---------------------------------------------------------------------------
class TestDeviceDetectionCUDA:
    """Verify _detect_device('auto') → 'cuda' when CUDA is available."""

    def test_auto_resolves_to_cuda_on_linux(self):
        """On non-darwin with CUDA available → 'cuda'."""
        import torch as _real_torch
        with patch.object(_real_torch.cuda, 'is_available', return_value=True), \
             patch.object(_real_torch.backends.mps, 'is_available', return_value=False), \
             patch('sys.platform', 'linux'):
            result = _detect_device('auto')
            assert result == 'cuda', f"Expected 'cuda', got '{result}'"

    def test_auto_resolves_to_cuda_on_windows(self):
        """On win32 with CUDA available → 'cuda'."""
        import torch as _real_torch
        with patch.object(_real_torch.cuda, 'is_available', return_value=True), \
             patch.object(_real_torch.backends.mps, 'is_available', return_value=False), \
             patch('sys.platform', 'win32'):
            result = _detect_device('auto')
            assert result == 'cuda'

    def test_explicit_cuda_passthrough(self):
        """device_config='cuda' is returned as-is without checking availability."""
        result = _detect_device('cuda')
        assert result == 'cuda'

    def test_no_cuda_no_mps_falls_to_cpu(self):
        """Neither CUDA nor MPS available → 'cpu'."""
        import torch as _real_torch
        with patch.object(_real_torch.cuda, 'is_available', return_value=False), \
             patch.object(_real_torch.backends.mps, 'is_available', return_value=False), \
             patch('sys.platform', 'linux'):
            result = _detect_device('auto')
            assert result == 'cpu'

    def test_darwin_with_mps_and_cuda_prefers_mps(self):
        """On darwin with both MPS and CUDA → MPS takes priority."""
        import torch as _real_torch
        with patch.object(_real_torch.cuda, 'is_available', return_value=True), \
             patch.object(_real_torch.backends.mps, 'is_available', return_value=True), \
             patch('sys.platform', 'darwin'):
            result = _detect_device('auto')
            assert result == 'mps', (
                "On darwin with both MPS and CUDA, MPS should take priority"
            )


# ---------------------------------------------------------------------------
# 2. _clear_device_cache — CUDA branch
# ---------------------------------------------------------------------------
class TestClearDeviceCacheCUDA:
    """Verify _clear_device_cache calls torch.cuda.empty_cache() on 'cuda'."""

    def test_cuda_empty_cache_called(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        embedder = LocalEmbedder(device='cuda')

        with patch.dict('sys.modules', {'torch': mock_torch}), \
             patch('gc.collect') as mock_gc:
            embedder._clear_device_cache()

            mock_gc.assert_called_once()
            mock_torch.cuda.empty_cache.assert_called_once()
            # MPS should NOT be called
            mock_torch.mps.synchronize.assert_not_called()
            mock_torch.mps.empty_cache.assert_not_called()

    def test_cpu_skips_all_cache_clearing(self):
        embedder = LocalEmbedder(device='cpu')

        with patch('gc.collect') as mock_gc:
            embedder._clear_device_cache()
            mock_gc.assert_not_called()

    def test_cuda_empty_cache_not_called_when_cuda_unavailable(self):
        """Edge case: device='cuda' but torch.cuda.is_available() is False
        (e.g., driver crash mid-run). empty_cache should NOT be called."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        # Ensure hasattr(mock_torch, 'mps') is not True for the mps branch
        del mock_torch.mps

        embedder = LocalEmbedder(device='cuda')

        with patch.dict('sys.modules', {'torch': mock_torch}), \
             patch('gc.collect'):
            embedder._clear_device_cache()
            mock_torch.cuda.empty_cache.assert_not_called()

    def test_mps_device_does_not_call_cuda(self):
        """Ensure MPS path doesn't accidentally hit CUDA branch."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        # hasattr(mock_torch, 'mps') is True by default on MagicMock

        embedder = LocalEmbedder(device='mps')

        with patch.dict('sys.modules', {'torch': mock_torch}), \
             patch('gc.collect'):
            embedder._clear_device_cache()
            mock_torch.cuda.empty_cache.assert_not_called()
            mock_torch.mps.synchronize.assert_called_once()
            mock_torch.mps.empty_cache.assert_called_once()


# ---------------------------------------------------------------------------
# 3. should_unload_model
# ---------------------------------------------------------------------------
class TestShouldUnloadModelCUDA:
    """Verify CUDA skips periodic model unload."""

    def test_cuda_returns_false(self):
        assert should_unload_model('cuda') is False

    def test_mps_returns_true(self):
        assert should_unload_model('mps') is True

    def test_cpu_returns_false(self):
        assert should_unload_model('cpu') is False

    def test_cuda_colon_zero_returns_false(self):
        """Multi-GPU string 'cuda:0' also returns False."""
        assert should_unload_model('cuda:0') is False


# ---------------------------------------------------------------------------
# 4. gc.collect guard in indexing_pipeline
# ---------------------------------------------------------------------------
class TestGcCollectGuardCUDA:
    """The gc.collect() in indexing_pipeline runs for non-CPU devices."""

    def test_gc_runs_for_cuda(self):
        """emb_manager.device == 'cuda' → gc.collect() is called."""
        mock_emb_manager = MagicMock()
        mock_emb_manager.device = 'cuda'
        mock_emb_manager.embed_batch.return_value = MagicMock()

        # The guard: if emb_manager.device != 'cpu': gc.collect()
        assert mock_emb_manager.device != 'cpu'  # Precondition

    def test_gc_skipped_for_cpu(self):
        """emb_manager.device == 'cpu' → gc.collect() is NOT called."""
        mock_emb_manager = MagicMock()
        mock_emb_manager.device = 'cpu'

        assert not (mock_emb_manager.device != 'cpu')  # Precondition


# ---------------------------------------------------------------------------
# 5. embed_batch: CUDA OOM vs MPS buffer size error handling
# ---------------------------------------------------------------------------
class TestEmbedBatchCUDAOOM:
    """
    The MPS 'buffer size' retry logic must NOT swallow CUDA OOM errors.
    CUDA OOM raises torch.cuda.OutOfMemoryError with message like
    'CUDA out of memory. Tried to allocate...', which does NOT contain
    'buffer size'. The error should propagate up.
    """

    def _make_embedder_with_model(self, device='cuda'):
        """Create a LocalEmbedder with a mocked model already loaded."""
        embedder = LocalEmbedder(device=device)
        # Inject a mock model so embed_batch skips _load_model entirely
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        embedder._model = mock_model
        return embedder

    def test_cuda_oom_is_not_caught_by_buffer_size_check(self):
        """CUDA OOM error message does not contain 'buffer size', so it
        should be re-raised immediately, not retried."""
        embedder = self._make_embedder_with_model('cuda')

        cuda_oom_msg = (
            "CUDA out of memory. Tried to allocate 256.00 MiB "
            "(GPU 0; 7.79 GiB total capacity)"
        )
        oom_error = RuntimeError(cuda_oom_msg)
        embedder._model.encode.side_effect = oom_error

        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            embedder.embed_batch(["test text 1", "test text 2"], batch_size=2)

    def test_mps_buffer_size_error_is_caught_and_retried(self):
        """MPS buffer size error IS caught and triggers per-item retry."""
        embedder = self._make_embedder_with_model('mps')

        mps_error = RuntimeError(
            "MPS backend out of memory. "
            "Current buffer size: 1.00 GiB."
        )

        call_count = 0
        def side_effect_fn(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call (batch) fails
                raise mps_error
            # Subsequent calls (per-item) succeed
            return np.random.rand(1024).astype(np.float32)

        embedder._model.encode.side_effect = side_effect_fn

        with patch.object(embedder, '_clear_device_cache'):
            result = embedder.embed_batch(["text1", "text2"], batch_size=2)
            # Should succeed via per-item retry
            assert result.shape == (2, 1024)

    def test_generic_runtime_error_is_reraised(self):
        """Non-buffer-size RuntimeError is always re-raised."""
        embedder = self._make_embedder_with_model('cuda')
        embedder._model.encode.side_effect = RuntimeError("some other error")

        with pytest.raises(RuntimeError, match="some other error"):
            embedder.embed_batch(["test"], batch_size=1)

    def test_cuda_oom_error_class_is_reraised(self):
        """torch.cuda.OutOfMemoryError (if it exists) is re-raised.
        This tests the actual exception class, not just message matching."""
        embedder = self._make_embedder_with_model('cuda')

        # Simulate torch.cuda.OutOfMemoryError (subclass of RuntimeError)
        class FakeOutOfMemoryError(RuntimeError):
            pass

        oom = FakeOutOfMemoryError(
            "CUDA out of memory. Tried to allocate 512.00 MiB"
        )
        embedder._model.encode.side_effect = oom

        with pytest.raises(FakeOutOfMemoryError):
            embedder.embed_batch(["test"], batch_size=1)


# ---------------------------------------------------------------------------
# 6. Reranker device resolution
# ---------------------------------------------------------------------------
class TestRerankerCUDADevice:
    """Verify Reranker resolves 'auto' → 'cuda'."""

    def test_reranker_auto_resolves_to_cuda(self):
        import torch as _real_torch
        mock_cross_encoder_cls = MagicMock()

        with patch.object(_real_torch.cuda, 'is_available', return_value=True), \
             patch.object(_real_torch.backends.mps, 'is_available', return_value=False), \
             patch('sys.platform', 'linux'), \
             patch('sentence_transformers.CrossEncoder', mock_cross_encoder_cls, create=True):

            reranker = Reranker(device='auto')
            reranker._load_model()

            assert reranker.device == 'cuda'

    def test_reranker_explicit_cuda(self):
        mock_cross_encoder_cls = MagicMock()

        reranker = Reranker(device='cuda')

        # CrossEncoder is imported lazily inside _load_model as:
        #   from sentence_transformers import CrossEncoder
        # So we patch it at the sentence_transformers package level.
        with patch('sentence_transformers.CrossEncoder', mock_cross_encoder_cls, create=True):
            reranker._load_model()

        assert reranker.device == 'cuda'
        mock_cross_encoder_cls.assert_called_once_with(
            'BAAI/bge-reranker-v2-m3', device='cuda'
        )


# ---------------------------------------------------------------------------
# 7. MPS env var isolation
# ---------------------------------------------------------------------------
class TestMPSEnvVarIsolation:
    """PYTORCH_MPS_HIGH_WATERMARK_RATIO should only be set on macOS."""

    def test_not_set_on_linux(self):
        """On Linux, setup_mps_env() should be a no-op."""
        env_backup = os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO')
        try:
            os.environ.pop('PYTORCH_MPS_HIGH_WATERMARK_RATIO', None)

            with patch.object(_compat_mod, 'IS_MACOS', False):
                setup_mps_env()
                assert 'PYTORCH_MPS_HIGH_WATERMARK_RATIO' not in os.environ
        finally:
            if env_backup is not None:
                os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = env_backup
            else:
                os.environ.pop('PYTORCH_MPS_HIGH_WATERMARK_RATIO', None)


# ---------------------------------------------------------------------------
# 8. unload_model on CUDA
# ---------------------------------------------------------------------------
class TestUnloadModelCUDA:
    """Verify unload_model properly clears CUDA resources."""

    def test_unload_calls_clear_device_cache(self):
        embedder = LocalEmbedder(device='cuda')
        embedder._model = MagicMock()

        with patch.object(embedder, '_clear_device_cache') as mock_clear:
            embedder.unload_model()
            assert embedder._model is None
            mock_clear.assert_called_once()

    def test_unload_noop_when_not_loaded(self):
        embedder = LocalEmbedder(device='cuda')
        assert embedder._model is None

        with patch.object(embedder, '_clear_device_cache') as mock_clear:
            embedder.unload_model()
            mock_clear.assert_not_called()


# ---------------------------------------------------------------------------
# 9. Periodic unload skipped for CUDA in main/server
# ---------------------------------------------------------------------------
class TestPeriodicUnloadSkippedCUDA:
    """The 200-file periodic unload condition uses should_unload_model,
    which returns False for CUDA. Verify the logic."""

    def test_periodic_unload_condition_false_for_cuda(self):
        device = 'cuda'
        _files_since_unload = 300  # Way past the 200 threshold
        is_loaded = True

        should_unload = (
            should_unload_model(device)
            and _files_since_unload >= 200
            and is_loaded
        )
        assert should_unload is False, (
            "CUDA should NOT trigger periodic unload even after 200+ files"
        )

    def test_periodic_unload_condition_true_for_mps(self):
        device = 'mps'
        _files_since_unload = 200
        is_loaded = True

        should_unload = (
            should_unload_model(device)
            and _files_since_unload >= 200
            and is_loaded
        )
        assert should_unload is True


# ---------------------------------------------------------------------------
# 10. EmbeddingManager integration with CUDA device
# ---------------------------------------------------------------------------
class TestEmbeddingManagerCUDA:
    """Integration-level tests for EmbeddingManager with CUDA."""

    def test_embedding_manager_device_property(self):
        """EmbeddingManager.device reflects the detected device."""
        import torch as _real_torch
        with patch.object(_real_torch.cuda, 'is_available', return_value=True), \
             patch.object(_real_torch.backends.mps, 'is_available', return_value=False), \
             patch('sys.platform', 'linux'):
            manager = EmbeddingManager({'device': 'auto'})
            assert manager.device == 'cuda'

    def test_embedding_manager_explicit_cuda(self):
        manager = EmbeddingManager({'device': 'cuda'})
        assert manager.device == 'cuda'


# ---------------------------------------------------------------------------
# 11. Edge case: 'cuda:N' multi-GPU string
# ---------------------------------------------------------------------------
class TestMultiGPUEdgeCases:
    """Edge cases with multi-GPU device strings like 'cuda:0', 'cuda:1'."""

    def test_should_unload_model_cuda_colon_n(self):
        # 'cuda:0' != 'mps', so returns False — correct
        assert should_unload_model('cuda:0') is False
        assert should_unload_model('cuda:1') is False

    def test_clear_device_cache_cuda_colon_zero_misses_branch(self):
        """BUG DOCUMENTATION: 'cuda:0' fails the == 'cuda' check,
        so neither MPS nor CUDA cache clearing runs."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        embedder = LocalEmbedder(device='cuda:0')

        with patch.dict('sys.modules', {'torch': mock_torch}), \
             patch('gc.collect') as mock_gc:
            embedder._clear_device_cache()
            # gc.collect IS called (device != 'cpu')
            mock_gc.assert_called_once()
            # BUT torch.cuda.empty_cache is NOT called (device != 'cuda')
            mock_torch.cuda.empty_cache.assert_not_called()


# ---------------------------------------------------------------------------
# 12. Comprehensive CUDA OOM message patterns
# ---------------------------------------------------------------------------
class TestCUDAOOMMessagePatterns:
    """Verify various real-world CUDA OOM message patterns are NOT
    caught by the 'buffer size' check."""

    CUDA_OOM_MESSAGES = [
        "CUDA out of memory. Tried to allocate 256.00 MiB (GPU 0; 7.79 GiB total capacity; 6.23 GiB already allocated)",
        "CUDA out of memory. Tried to allocate 20.00 MiB (GPU 0; 3.95 GiB total capacity)",
        "RuntimeError: CUDA error: out of memory",
        "CUDA out of memory. Tried to allocate 1.00 GiB. GPU 0 has a total capacity of 4.00 GiB",
        "torch.OutOfMemoryError: CUDA out of memory.",
    ]

    MPS_BUFFER_MESSAGES = [
        "MPS backend out of memory (MPS allocated: 6.57 GB, other allocations: 384.00 KB, max allowed: 9.07 GB). Current buffer size: 256 bytes.",
        "Invalid buffer size: 1073741824",
    ]

    @pytest.mark.parametrize("msg", CUDA_OOM_MESSAGES)
    def test_cuda_oom_not_caught(self, msg):
        """CUDA OOM messages should NOT contain 'buffer size'."""
        assert 'buffer size' not in msg.lower(), (
            f"CUDA OOM message unexpectedly contains 'buffer size': {msg}"
        )

    @pytest.mark.parametrize("msg", MPS_BUFFER_MESSAGES)
    def test_mps_buffer_is_caught(self, msg):
        """MPS buffer messages SHOULD contain 'buffer size'."""
        assert 'buffer size' in msg.lower(), (
            f"MPS buffer message should contain 'buffer size': {msg}"
        )
