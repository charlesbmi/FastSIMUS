# Python Unit Testing: Write Tests. Not Too Many. Mostly Integration.

## When to Use

Use this skill when:

- Writing or reviewing unit tests in Python
- Deciding what to test and how to test it
- Refactoring test suites to improve maintainability
- Balancing test coverage with development velocity
- Testing scientific/numerical code with Array API compliance

## Philosophy

> "Write tests. Not too many. Mostly integration." — Guillermo Rauch

Tests should maximize **confidence** while minimizing **maintenance burden**. Focus on testing behavior, not
implementation details.

## Core Principles

### 1. Test for Confidence, Not Coverage

**Goal**: Catch bugs before production, not achieve 100% coverage.

```python
# BAD: Testing framework behavior (low confidence)
def test_pydantic_validates_positive():
    """Test that Pydantic enforces gt=0."""
    with pytest.raises(ValidationError):
        MyModel(value=0)  # Just testing Pydantic itself

# GOOD: Testing custom business logic (high confidence)
def test_physical_constraints():
    """Test custom constraint: element width cannot exceed pitch."""
    with pytest.raises(ValueError, match="Element width .* cannot exceed pitch"):
        TransducerParams(pitch=0.0003, width=0.0004)
```

**Guideline**: 70-80% coverage is often optimal. Beyond that, diminishing returns set in.

### 2. Don't Test the Framework

Avoid testing library/framework behavior that's already tested upstream.

#### What NOT to Test

```python
# DON'T test basic Pydantic validation
def test_required_fields():
    with pytest.raises(ValidationError):
        MyModel()  # Missing required field

# DON'T test default values
def test_defaults():
    obj = MyModel(required_field=1)
    assert obj.optional_field == "default"  # Pydantic handles this

# DON'T test type coercion
def test_string_to_int():
    obj = MyModel(number="42")
    assert obj.number == 42  # Pydantic handles this
```

#### What TO Test

```python
# DO test custom validators
def test_custom_validation():
    """Test custom validator: width and kerf are mutually exclusive."""
    with pytest.raises(ValueError, match="Cannot specify both"):
        MyModel(width=0.1, kerf=0.05)

# DO test computed fields
def test_computed_field():
    """Test computed field logic."""
    params = TransducerParams(pitch=0.0003, width=0.00025)
    assert params.kerf_width == pytest.approx(0.00005)  # pitch - width
```

### 3. Sociable Tests Over Solitary Tests

Prefer testing real collaborations over mocking (Martin Fowler's "sociable" tests).

```python
# GOOD: Sociable test - tests real integration
def test_all_presets_instantiate():
    """Test that all presets create valid TransducerParams instances."""
    presets = [P4_2v(), L11_5v(), L12_3v(), C5_2v()]

    for params in presets:
        # Tests real object creation and validation
        assert params.freq_center > 0
        assert params.element_width + params.kerf_width == pytest.approx(params.pitch)

# AVOID: Solitary test with excessive mocking
def test_preset_with_mocks():
    with patch('module.TransducerParams') as mock:
        P4_2v()
        mock.assert_called_once()  # Low confidence - doesn't test actual behavior
```

**When to use test doubles**:

- External resources (databases, APIs, filesystems) when slow/non-deterministic
- Non-deterministic behavior (random, time, network)
- Expensive operations (GPU kernels, large computations)

### 4. High Cohesion, Low Coupling

**Cohesion**: Test related functionality together. **Coupling**: Minimize dependencies between test cases.

```python
# GOOD: High cohesion - related tests grouped
class TestWidthKerfRelationship:
    """Tests for width/kerf mutual exclusivity and computation."""

    def test_missing_both(self):
        """Either width or kerf must be provided."""
        with pytest.raises(ValueError, match="Either width or kerf"):
            TransducerParams(pitch=0.0003)

    def test_both_provided(self):
        """Cannot specify both width and kerf."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            TransducerParams(pitch=0.0003, width=0.00025, kerf=0.00005)

    def test_computation(self):
        """Width and kerf are computed from each other."""
        params = TransducerParams(pitch=0.0003, width=0.00025)
        assert params.kerf_width == pytest.approx(0.00005)

# GOOD: Low coupling - tests are independent
def test_basic_instantiation():
    """Test basic instantiation (no dependencies on other tests)."""
    params = TransducerParams(freq_center=2.5e6, pitch=0.0003, n_elements=64, width=0.00025)
    assert params.freq_center == 2.5e6
```

### 5. One Assertion Per Concept, Not Per Test

Group related assertions that test the same concept.

```python
# GOOD: Multiple assertions testing one concept
def test_width_kerf_computation():
    """Test that width and kerf are computed correctly from each other."""
    # Concept: Provide width, compute kerf
    params = TransducerParams(pitch=0.0003, width=0.00025)
    assert params.element_width == 0.00025
    assert params.kerf_width == pytest.approx(0.00005)

    # Concept: Provide kerf, compute width
    params = TransducerParams(pitch=0.0003, kerf=0.00005)
    assert params.element_width == pytest.approx(0.00025)
    assert params.kerf_width == 0.00005

# AVOID: Splitting into too many tiny tests
def test_width_provided_returns_width():
    params = TransducerParams(pitch=0.0003, width=0.00025)
    assert params.element_width == 0.00025

def test_width_provided_computes_kerf():
    params = TransducerParams(pitch=0.0003, width=0.00025)
    assert params.kerf_width == pytest.approx(0.00005)
# ... (too granular, harder to maintain)
```

## Testing Patterns for Scientific Python

### Pattern 1: Parametrized Backend Tests

For Array API compliance, test across multiple backends.

```python
@pytest.fixture(
    params=[
        pytest.param(np, id="numpy", marks=pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")),
        pytest.param(jnp, id="jax", marks=pytest.mark.skipif(not HAS_JAX, reason="JAX not available")),
        pytest.param(cp, id="cupy", marks=pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")),
    ]
)
def xp(request):
    """Fixture providing different array API backends."""
    return request.param

def test_array_operation(xp):
    """Test that operation works across all backends."""
    arr = xp.asarray([1, 2, 3])
    result = my_function(arr)
    assert result.shape == (3,)
```

### Pattern 2: Numerical Tolerance

Use `pytest.approx()` for floating-point comparisons.

```python
def test_numerical_computation():
    """Test numerical computation with appropriate tolerance."""
    params = TransducerParams(pitch=0.0003, width=0.00025)
    # Floating point: use approx
    assert params.kerf_width == pytest.approx(0.00005, rel=1e-6)

    # Exact values: direct comparison
    assert params.n_elements == 64
```

### Pattern 3: Consolidate Preset/Configuration Tests

Test collections of presets together rather than individually.

```python
# GOOD: Consolidated preset test
def test_all_presets_instantiate():
    """Test that all presets create valid TransducerParams instances."""
    presets = [P4_2v(), L11_5v(), L12_3v(), C5_2v()]

    for params in presets:
        # Common invariants
        assert params.freq_center > 0
        assert params.pitch > 0
        assert params.element_width + params.kerf_width == pytest.approx(params.pitch)
        assert params.fs == pytest.approx(4.0 * params.freq_center)

# GOOD: Spot-check key differences
def test_p4_2v_key_values():
    """Test P4-2v preset key values."""
    params = P4_2v()
    assert params.freq_center == 2.72e6
    assert params.radius == inf  # Linear array

def test_c5_2v_key_values():
    """Test C5-2v preset key values (convex array)."""
    params = C5_2v()
    assert params.freq_center == 3.57e6
    assert params.radius == pytest.approx(0.04957)  # Convex array

# AVOID: Testing every field of every preset
def test_p4_2v_freq_center():
    assert P4_2v().freq_center == 2.72e6

def test_p4_2v_pitch():
    assert P4_2v().pitch == 0.0003

def test_p4_2v_n_elements():
    assert P4_2v().n_elements == 64
# ... (too verbose, low value)
```

### Pattern 4: Basic Instantiation Test

One test to verify the class works and defaults are set.

```python
def test_basic_instantiation():
    """Test that class can be instantiated with valid parameters."""
    # Basic instantiation
    params = TransducerParams(freq_center=2.5e6, pitch=0.0003, n_elements=64, width=0.00025)

    # Spot-check required fields
    assert params.freq_center == 2.5e6
    assert params.pitch == 0.0003

    # Verify defaults are set (documents expected behavior)
    assert params.height == inf
    assert params.bandwidth == 0.75
    assert params.speed_of_sound == 1540.0
```

## Test Organization

### Structure by Domain, Not by Type

```python
# GOOD: Organized by domain/feature
class TestWidthKerfRelationship:
    """Tests for width/kerf computation and validation."""
    def test_missing_both(self): ...
    def test_exclusivity(self): ...
    def test_computation(self): ...

class TestPhysicalConstraints:
    """Tests for physical constraint validation."""
    def test_width_exceeds_pitch(self): ...
    def test_kerf_exceeds_pitch(self): ...

class TestPresets:
    """Tests for preset transducer configurations."""
    def test_all_presets_instantiate(self): ...
    def test_p4_2v_key_values(self): ...

# AVOID: Organized by test type
class TestValidation:
    def test_width_kerf_missing(self): ...
    def test_width_exceeds_pitch(self): ...
    def test_baffle_validation(self): ...
    # (unrelated validations grouped together)
```

### File Naming

```
tests/
├── test_transducers.py          # Tests for transducer module
├── test_pfield.py               # Tests for pressure field
├── test_txdelay.py              # Tests for transmit delay
├── conftest.py                  # Shared fixtures
└── utils/
    └── test_array_api.py        # Tests for array API utilities
```

## What to Test: Decision Tree

```
Is this custom business logic?
├─ Yes → TEST IT
│   ├─ Custom validators
│   ├─ Computed fields
│   ├─ Complex calculations
│   └─ Domain-specific constraints
│
└─ No → Is it framework/library behavior?
    ├─ Yes → DON'T TEST IT
    │   ├─ Pydantic validation (gt, ge, le)
    │   ├─ Default values
    │   ├─ Type coercion and covered by static-type
    │   └─ Required fields
    │
    └─ Is it a critical integration point?
        ├─ Yes → TEST IT
        │   ├─ Preset configurations
        │   ├─ Multi-backend compatibility
        │   └─ Cross-module interactions
        │
        └─ No → SKIP IT
```

## Refactoring Tests: Before & After

### Before: Testing Pydantic (282 lines, 21 tests)

```python
def test_required_fields():
    """Test that required fields are enforced."""
    with pytest.raises(ValidationError):
        TransducerParams(pitch=0.0003, n_elements=64, width=0.00025)  # Missing freq_center

def test_positive_value_constraints():
    """Test that numeric fields must be positive."""
    with pytest.raises(ValidationError):
        TransducerParams(freq_center=0, pitch=0.0003, n_elements=64, width=0.00025)

def test_default_values():
    """Test default values for optional fields."""
    params = TransducerParams(freq_center=2.5e6, pitch=0.0003, n_elements=64, width=0.00025)
    assert params.height == inf
    assert params.bandwidth == 0.75

def test_p4_2v_preset():
    """Test P4-2v preset values."""
    params = P4_2v()
    assert params.freq_center == 2.72e6
    assert params.pitch == 0.0003
    assert params.n_elements == 64
    # ... 10+ more assertions
```

### After: Testing Business Logic (84 lines, 10 tests)

```python
def test_basic_instantiation():
    """Test that TransducerParams can be instantiated with valid parameters."""
    params = TransducerParams(freq_center=2.5e6, pitch=0.0003, n_elements=64, width=0.00025)
    assert params.freq_center == 2.5e6
    # Verify defaults are set
    assert params.height == inf
    assert params.bandwidth == 0.75

def test_width_kerf_computation():
    """Test that width and kerf are computed correctly from each other."""
    params = TransducerParams(pitch=0.0003, width=0.00025)
    assert params.element_width == 0.00025
    assert params.kerf_width == pytest.approx(0.00005)

def test_all_presets_instantiate():
    """Test that all presets create valid TransducerParams instances."""
    presets = [P4_2v(), L11_5v(), L12_3v(), C5_2v()]
    for params in presets:
        assert params.freq_center > 0
        assert params.element_width + params.kerf_width == pytest.approx(params.pitch)
```

**Result**: 70% fewer lines, same confidence, easier maintenance.

## Speed Guidelines

Tests should be fast enough to run frequently.

| Suite Type        | Target Speed | When to Run             |
| ----------------- | ------------ | ----------------------- |
| **Compile suite** | < 1 second   | After every code change |
| **Commit suite**  | < 10 seconds | Before every commit     |
| **CI suite**      | < 5 minutes  | On every push           |

```python
# Fast: Pure Python logic
def test_computation():
    params = TransducerParams(pitch=0.0003, width=0.00025)
    assert params.kerf_width == pytest.approx(0.00005)  # ~0.1ms

# Slow: GPU operations, large arrays
@pytest.mark.slow
def test_large_simulation():
    result = simulate_rf(n_elements=1024, n_samples=100000)  # ~2 seconds
    assert result.shape == (1024, 100000)
```

## Common Anti-Patterns to Avoid

### ❌ Testing Implementation Details

```python
# BAD: Coupled to internal structure
def test_internal_method():
    obj = MyClass()
    assert obj._internal_helper() == 42  # Breaks on refactoring

# GOOD: Test public API
def test_public_behavior():
    obj = MyClass()
    assert obj.compute() == 42  # Stable across refactoring
```

### ❌ Excessive Mocking

```python
# BAD: Mocking everything
def test_with_mocks():
    with patch('module.ClassA'), patch('module.ClassB'), patch('module.ClassC'):
        result = my_function()
        # Low confidence - not testing real integration

# GOOD: Test real collaborations
def test_real_integration():
    result = my_function()  # Uses real ClassA, ClassB, ClassC
    assert result.is_valid()
```

### ❌ Brittle Assertions

```python
# BAD: Over-specified
def test_error_message():
    with pytest.raises(ValueError, match="Element width (0.0004) cannot exceed pitch (0.0003) for element spacing"):
        TransducerParams(pitch=0.0003, width=0.0004)

# GOOD: Flexible regex
def test_error_message():
    with pytest.raises(ValueError, match=r"Element width .* cannot exceed pitch"):
        TransducerParams(pitch=0.0003, width=0.0004)
```

## References

- Kent C. Dodds: [Write tests. Not too many. Mostly integration.](https://kentcdodds.com/blog/write-tests)
- Martin Fowler: [Unit Test](https://martinfowler.com/bliki/UnitTest.html) (Sociable vs Solitary)
- Testing Trophy: Focus on integration tests for best ROI
- High Cohesion, Low Coupling: [Medium article](https://medium.com/clarityhub/low-coupling-high-cohesion-3610e35ac4a6)

## Quick Checklist

Before writing a test, ask:

- [ ] Does this test custom business logic (not framework behavior)?
- [ ] Will this test catch real bugs?
- [ ] Can I refactor the code without changing this test?
- [ ] Is this test fast enough to run frequently?
- [ ] Does this test provide confidence, not just coverage?

If you answer "no" to any of these, reconsider whether the test is needed.
