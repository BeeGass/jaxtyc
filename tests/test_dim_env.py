"""Tests for jaxtyc.analyzer.dim_env — prime-based dimension environment."""

from __future__ import annotations

from jaxtyc.analyzer.dim_env import DimEnv
from jaxtyc.analyzer.dim_env import _prime_sieve
from jaxtyc.types import DimSpec
from jaxtyc.types import ShapeSpec


class TestPrimeSieve:
    def test_small_limit(self):
        assert _prime_sieve(10) == [2, 3, 5, 7]

    def test_limit_zero(self):
        assert _prime_sieve(0) == []

    def test_limit_one(self):
        assert _prime_sieve(1) == []

    def test_limit_two(self):
        assert _prime_sieve(2) == [2]

    def test_primes_up_to_30(self):
        assert _prime_sieve(30) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    def test_all_primes_are_prime(self):
        primes = _prime_sieve(100)
        for p in primes:
            assert p >= 2
            for d in range(2, int(p**0.5) + 1):
                assert p % d != 0, f"{p} is not prime"


class TestDimEnv:
    def test_assigns_unique_primes(self):
        env = DimEnv()
        batch = env.get_size("batch")
        seq = env.get_size("seq")
        d_model = env.get_size("d_model")
        assert batch != seq != d_model
        # All should be prime
        for size in (batch, seq, d_model):
            assert size >= 2

    def test_same_name_same_size(self):
        env = DimEnv()
        s1 = env.get_size("batch")
        s2 = env.get_size("batch")
        assert s1 == s2

    def test_reverse_mapping(self):
        env = DimEnv()
        env.get_size("batch")
        env.get_size("seq")
        assert env.resolve_name(env.get_size("batch")) == "batch"
        assert env.resolve_name(env.get_size("seq")) == "seq"
        assert env.resolve_name(999999) is None

    def test_shape_to_names(self):
        env = DimEnv()
        batch = env.get_size("batch")
        seq = env.get_size("seq")
        names = env.shape_to_names((batch, seq))
        assert names == ("batch", "seq")

    def test_shape_to_names_with_unknown(self):
        env = DimEnv()
        batch = env.get_size("batch")
        names = env.shape_to_names((batch, 999999))
        assert names == ("batch", None)

    def test_make_shape_named(self):
        env = DimEnv()
        spec = ShapeSpec(
            dims=(
                DimSpec(kind="named", name="batch"),
                DimSpec(kind="named", name="seq"),
                DimSpec(kind="named", name="d_model"),
            ),
            dtype="float32",
        )
        shape = env.make_shape(spec)
        assert len(shape) == 3
        # All unique primes
        assert len(set(shape)) == 3
        # Same name -> same size
        assert env.get_size("batch") == shape[0]
        assert env.get_size("seq") == shape[1]
        assert env.get_size("d_model") == shape[2]

    def test_make_shape_fixed(self):
        env = DimEnv()
        spec = ShapeSpec(
            dims=(
                DimSpec(kind="named", name="batch"),
                DimSpec(kind="fixed", size=4),
                DimSpec(kind="named", name="dim"),
            ),
            dtype="float32",
        )
        shape = env.make_shape(spec)
        assert shape[1] == 4

    def test_make_shape_variadic(self):
        env = DimEnv()
        spec = ShapeSpec(
            dims=(
                DimSpec(kind="variadic", name="batch"),
                DimSpec(kind="named", name="dim"),
            ),
            dtype="float32",
        )
        shape = env.make_shape(spec)
        # Variadic expands to 2 dims + 1 named = 3 total
        assert len(shape) == 3

    def test_make_shape_ellipsis(self):
        env = DimEnv()
        spec = ShapeSpec(
            dims=(
                DimSpec(kind="ellipsis"),
                DimSpec(kind="named", name="dim"),
            ),
            dtype="float32",
        )
        shape = env.make_shape(spec)
        # Ellipsis expands to 2 dims + 1 named = 3 total
        assert len(shape) == 3

    def test_make_shape_anonymous(self):
        env = DimEnv()
        spec = ShapeSpec(
            dims=(
                DimSpec(kind="anonymous"),
                DimSpec(kind="named", name="dim"),
            ),
            dtype="float32",
        )
        shape = env.make_shape(spec)
        assert len(shape) == 2
        # Anonymous dim should be a prime, different from named dim
        assert shape[0] != shape[1]

    def test_reset(self):
        env = DimEnv()
        env.get_size("batch")
        env.reset()
        assert env.resolve_name(2) is None

    def test_sieve_auto_extends(self):
        env = DimEnv(initial_sieve_limit=10)
        # There are only 4 primes up to 10: 2, 3, 5, 7
        # Requesting a 5th name should trigger sieve extension
        names = [f"dim_{i}" for i in range(10)]
        sizes = [env.get_size(n) for n in names]
        assert len(set(sizes)) == 10  # All unique

    def test_consistent_across_make_shape_calls(self):
        env = DimEnv()
        spec_a = ShapeSpec(
            dims=(DimSpec(kind="named", name="batch"), DimSpec(kind="named", name="seq")),
            dtype="float32",
        )
        spec_b = ShapeSpec(
            dims=(DimSpec(kind="named", name="seq"), DimSpec(kind="named", name="d_model")),
            dtype="float32",
        )
        shape_a = env.make_shape(spec_a)
        shape_b = env.make_shape(spec_b)
        # "seq" should map to the same size in both
        assert shape_a[1] == shape_b[0]
