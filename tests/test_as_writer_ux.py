"""
Tests for AUTOSTRUCTURE writer UX enhancements.

Tests Phase 6 features:
- Helper classes (CoreSpecification, SymmetryRestriction, etc.)
- High-level presets (for_structure_calculation, etc.)
- Fluent interface (method chaining)
- Validation methods
"""

import pytest
from pathlib import Path
from src.atomkit.autostructure.as_writer import (
    ASWriter,
    CoreSpecification,
    SymmetryRestriction,
    EnergyShifts,
    CollisionParams,
    OptimizationParams,
    RydbergSeries,
)


class TestHelperClasses:
    """Test helper dataclasses for simplified configuration."""

    def test_core_specification_helium_like(self):
        """Test He-like core specification."""
        core = CoreSpecification.helium_like()
        assert core.kcor1 == 1
        assert core.kcor2 == 1

    def test_core_specification_neon_like(self):
        """Test Ne-like core specification."""
        core = CoreSpecification.neon_like()
        assert core.kcor1 == 1
        assert core.kcor2 == 3

    def test_core_specification_argon_like(self):
        """Test Ar-like core specification."""
        core = CoreSpecification.argon_like()
        assert core.kcor1 == 1
        assert core.kcor2 == 6

    def test_core_specification_from_orbitals(self):
        """Test custom core from orbital indices."""
        core = CoreSpecification.from_orbitals(1, 5)
        assert core.kcor1 == 1
        assert core.kcor2 == 5

    def test_symmetry_restriction_single_term(self):
        """Test single term restriction."""
        sym = SymmetryRestriction.single_term(S=0, L=0, parity=1)
        assert sym.nast == 1
        assert sym.term_list == [(1, 0, 1)]  # 2S+1=1, L=0, π=1

    def test_symmetry_restriction_multiple_terms(self):
        """Test multiple terms restriction."""
        terms = [(1, 0, 1), (3, 1, -1)]  # 1S+ and 3P-
        sym = SymmetryRestriction.terms(terms)
        assert sym.nast == 2
        assert sym.term_list == terms

    def test_symmetry_restriction_levels(self):
        """Test level restriction for IC coupling."""
        levels = [(1, 0, 0, 1), (1, 0, 2, 1)]  # 1S0 and 1S1
        sym = SymmetryRestriction.levels(levels)
        assert sym.nastj == 2
        assert sym.level_list == levels

    def test_energy_shifts(self):
        """Test energy shift specification."""
        shifts = EnergyShifts(ls_shift=0.5, ic_shift=0.3)
        assert shifts.ls_shift == 0.5
        assert shifts.ic_shift == 0.3
        assert shifts.continuum_ls == 0.0
        assert shifts.continuum_ic == 0.0

    def test_collision_params_basic(self):
        """Test basic collision parameters."""
        coll = CollisionParams(min_L=0, max_L=10, min_J=0, max_J=20)
        assert coll.min_L == 0
        assert coll.max_L == 10
        assert coll.min_J == 0
        assert coll.max_J == 20
        assert not coll.include_orbit_orbit
        assert not coll.include_fine_structure

    def test_collision_params_advanced(self):
        """Test collision parameters with fine-structure."""
        coll = CollisionParams(
            min_L=0,
            max_L=10,
            max_exchange_L=8,
            include_orbit_orbit=True,
            include_fine_structure=True,
            max_J_fine_structure=15,
        )
        assert coll.max_exchange_L == 8
        assert coll.include_orbit_orbit
        assert coll.include_fine_structure
        assert coll.max_J_fine_structure == 15

    def test_optimization_params_basic(self):
        """Test basic optimization parameters."""
        opt = OptimizationParams(include_lowest=10, n_lambdas=5)
        assert opt.include_lowest == 10
        assert opt.n_lambdas == 5
        assert opt.get_iwght() == 1  # statistical weighting

    def test_optimization_params_weighting(self):
        """Test optimization weighting schemes."""
        opt_stat = OptimizationParams(weighting="statistical")
        assert opt_stat.get_iwght() == 1

        opt_equal = OptimizationParams(weighting="equal")
        assert opt_equal.get_iwght() == 0

        opt_custom = OptimizationParams(weighting="custom")
        assert opt_custom.get_iwght() == -1

    def test_rydberg_series(self):
        """Test Rydberg series specification."""
        ryd = RydbergSeries(n_min=3, n_max=15, l_max=7)
        assert ryd.n_min == 3
        assert ryd.n_max == 15
        assert ryd.l_min == 0
        assert ryd.l_max == 7
        assert ryd.use_internal_mesh

    def test_rydberg_series_with_limits(self):
        """Test Rydberg series with radiative limits."""
        ryd = RydbergSeries(n_min=3, n_max=20, l_max=5, limit_radiative=100)
        assert ryd.limit_radiative == 100


class TestPresetMethods:
    """Test high-level preset methods."""

    def test_for_structure_calculation_basic(self, tmp_path):
        """Test basic structure calculation preset."""
        filename = tmp_path / "structure.dat"
        asw = ASWriter.for_structure_calculation(
            str(filename), nzion=10, coupling="LS", radiation="E1"
        )

        content = asw.get_content()
        assert "A.S." in content
        assert "Structure calculation for Z=10" in content
        assert "&SALGEB" in content
        assert "CUP='LS'" in content
        assert "RAD='E1'" in content

    def test_for_structure_calculation_with_core(self, tmp_path):
        """Test structure calculation with core specification."""
        filename = tmp_path / "structure_core.dat"
        core = CoreSpecification.helium_like()
        asw = ASWriter.for_structure_calculation(
            str(filename), nzion=10, coupling="IC", core=core
        )

        content = asw.get_content()
        assert "KCOR1=1" in content
        assert "KCOR2=1" in content

    def test_for_structure_calculation_with_optimization(self, tmp_path):
        """Test structure calculation with optimization."""
        filename = tmp_path / "structure_opt.dat"
        opt = OptimizationParams(include_lowest=10, n_lambdas=5, n_variational=3)
        asw = ASWriter.for_structure_calculation(
            str(filename), nzion=10, optimization=opt
        )

        # Check that optimization params are stored
        assert hasattr(asw, "_pending_sminim")
        assert asw._pending_sminim["INCLUD"] == 10
        assert asw._pending_sminim["NLAM"] == 5
        assert asw._pending_sminim["NVAR"] == 3

    def test_for_photoionization(self, tmp_path):
        """Test photoionization preset."""
        filename = tmp_path / "pi.dat"
        asw = ASWriter.for_photoionization(
            str(filename), nzion=10, energy_min=0.0, energy_max=50.0, n_energies=20
        )

        content = asw.get_content()
        assert "Photoionization for Z=10" in content
        assert "RUN='PI'" in content
        assert hasattr(asw, "_pending_sradcon")
        assert asw._pending_sradcon["MENG"] == -20
        assert asw._pending_sradcon["EMIN"] == 0.0
        assert asw._pending_sradcon["EMAX"] == 50.0

    def test_for_dielectronic_recombination(self, tmp_path):
        """Test DR preset."""
        filename = tmp_path / "dr.dat"
        ryd = RydbergSeries(n_min=3, n_max=15, l_max=7)
        asw = ASWriter.for_dielectronic_recombination(
            str(filename), nzion=10, rydberg=ryd
        )

        content = asw.get_content()
        assert "Dielectronic recombination for Z=10" in content
        assert "RUN='DR'" in content
        assert hasattr(asw, "_pending_drr")
        assert asw._pending_drr["NMIN"] == 3
        assert asw._pending_drr["NMAX"] == 15
        assert asw._pending_drr["LMAX"] == 7

    def test_for_collision(self, tmp_path):
        """Test collision preset."""
        filename = tmp_path / "collision.dat"
        coll = CollisionParams(min_L=0, max_L=10, min_J=0, max_J=20)
        asw = ASWriter.for_collision(str(filename), nzion=10, collision=coll)

        content = asw.get_content()
        assert "Electron impact excitation for Z=10" in content
        assert "RUN='DE'" in content
        assert "MINLT=0" in content
        assert "MAXLT=10" in content
        assert "MINJT=0" in content
        assert "MAXJT=20" in content

    def test_for_collision_with_exchange(self, tmp_path):
        """Test collision preset with exchange control."""
        filename = tmp_path / "collision_exchange.dat"
        coll = CollisionParams(
            min_L=0,
            max_L=10,
            max_exchange_L=8,
            max_exchange_multipole=5,
            include_orbit_orbit=True,
        )
        asw = ASWriter.for_collision(str(filename), nzion=10, collision=coll)

        content = asw.get_content()
        assert "MAXLX=8" in content
        assert "MXLAMX=5" in content
        assert "KUTOOX=1" in content


class TestFluentInterface:
    """Test fluent interface (method chaining)."""

    def test_with_core_returns_self(self, tmp_path):
        """Test that with_core returns self for chaining."""
        filename = tmp_path / "fluent.dat"
        asw = ASWriter(str(filename))
        result = asw.with_core(CoreSpecification.helium_like())
        assert result is asw

    def test_with_optimization_returns_self(self, tmp_path):
        """Test that with_optimization returns self for chaining."""
        filename = tmp_path / "fluent.dat"
        asw = ASWriter(str(filename))
        result = asw.with_optimization(OptimizationParams(include_lowest=10))
        assert result is asw

    def test_with_symmetry_returns_self(self, tmp_path):
        """Test that with_symmetry returns self for chaining."""
        filename = tmp_path / "fluent.dat"
        asw = ASWriter(str(filename))
        sym = SymmetryRestriction.single_term(S=0, L=0, parity=1)
        result = asw.with_symmetry(sym)
        assert result is asw

    def test_with_energy_shifts_returns_self(self, tmp_path):
        """Test that with_energy_shifts returns self for chaining."""
        filename = tmp_path / "fluent.dat"
        asw = ASWriter(str(filename))
        shifts = EnergyShifts(ls_shift=0.5)
        result = asw.with_energy_shifts(shifts)
        assert result is asw

    def test_method_chaining(self, tmp_path):
        """Test full method chaining."""
        filename = tmp_path / "chained.dat"
        core = CoreSpecification.neon_like()
        opt = OptimizationParams(include_lowest=5)
        shifts = EnergyShifts(ls_shift=0.3)

        asw = (
            ASWriter(str(filename))
            .with_core(core)
            .with_optimization(opt)
            .with_energy_shifts(shifts)
        )

        assert hasattr(asw, "_pending_core")
        assert hasattr(asw, "_pending_optimization")
        assert hasattr(asw, "_pending_shifts")


class TestValidationMethods:
    """Test validation methods."""

    def test_validate_empty_writer(self, tmp_path):
        """Test validation of empty writer."""
        filename = tmp_path / "empty.dat"
        asw = ASWriter(str(filename))
        warnings = asw.validate()

        assert len(warnings) > 0
        assert any("header" in w.lower() for w in warnings)
        assert any("salgeb" in w.lower() for w in warnings)
        assert any("sminim" in w.lower() for w in warnings)

    def test_validate_with_header_only(self, tmp_path):
        """Test validation with only header."""
        filename = tmp_path / "header_only.dat"
        asw = ASWriter(str(filename))
        asw.write_header("Test")
        warnings = asw.validate()

        assert len(warnings) > 0
        assert not any("header" in w.lower() for w in warnings)
        assert any("salgeb" in w.lower() for w in warnings)

    def test_validate_complete_structure(self, tmp_path):
        """Test validation of complete structure calculation."""
        filename = tmp_path / "complete.dat"
        asw = ASWriter(str(filename))
        asw.write_header("Test")
        asw.add_salgeb(CUP="LS", RAD="E1")
        asw.add_orbitals([(1, 0), (2, 0)])
        asw.add_configurations([[2, 0], [1, 1]])
        asw.add_sminim(NZION=10)

        warnings = asw.validate()
        assert len(warnings) == 0

    def test_validate_pi_without_sradcon(self, tmp_path):
        """Test validation catches missing SRADCON for PI."""
        filename = tmp_path / "pi_incomplete.dat"
        asw = ASWriter(str(filename))
        asw.write_header("Test")
        asw.add_salgeb(CUP="IC", RUN="PI")
        asw.add_sminim(NZION=10)

        warnings = asw.validate()
        assert any("SRADCON" in w and "PI" in w for w in warnings)

    def test_validate_dr_without_drr(self, tmp_path):
        """Test validation catches missing DRR for DR."""
        filename = tmp_path / "dr_incomplete.dat"
        asw = ASWriter(str(filename))
        asw.write_header("Test")
        asw.add_salgeb(CUP="IC", RUN="DR")
        asw.add_sminim(NZION=10)

        warnings = asw.validate()
        assert any("DRR" in w and "DR" in w for w in warnings)

    def test_validate_ls_with_j_quantum_numbers(self, tmp_path):
        """Test validation catches J quantum numbers in LS coupling."""
        filename = tmp_path / "ls_j_error.dat"
        asw = ASWriter(str(filename))
        asw.write_header("Test")
        asw.add_salgeb(CUP="LS", NASTJ=5)  # NASTJ shouldn't be used with LS
        asw.add_sminim(NZION=10)

        warnings = asw.validate()
        assert any("LS" in w and "J" in w for w in warnings)

    def test_validate_and_raise_with_errors(self, tmp_path):
        """Test validate_and_raise throws exception."""
        filename = tmp_path / "error.dat"
        asw = ASWriter(str(filename))

        with pytest.raises(ValueError) as exc_info:
            asw.validate_and_raise()

        assert "validation failed" in str(exc_info.value).lower()

    def test_validate_and_raise_with_no_errors(self, tmp_path):
        """Test validate_and_raise succeeds when valid."""
        filename = tmp_path / "valid.dat"
        asw = ASWriter(str(filename))
        asw.write_header("Test")
        asw.add_salgeb(CUP="LS")
        asw.add_sminim(NZION=10)

        # Should not raise
        asw.validate_and_raise()

    def test_check_completeness_incomplete(self, tmp_path):
        """Test check_completeness returns False when incomplete."""
        filename = tmp_path / "incomplete.dat"
        asw = ASWriter(str(filename))
        asw.write_header("Test")

        assert not asw.check_completeness()

    def test_check_completeness_complete(self, tmp_path):
        """Test check_completeness returns True when complete."""
        filename = tmp_path / "complete.dat"
        asw = ASWriter(str(filename))
        asw.write_header("Test")
        asw.add_salgeb(CUP="LS")
        asw.add_sminim(NZION=10)

        assert asw.check_completeness()


class TestIntegrationWithExistingFeatures:
    """Test that new UX features integrate well with existing functionality."""

    def test_preset_with_configs_from_atomkit(self, tmp_path):
        """Test using preset methods with configs_from_atomkit."""
        pytest.importorskip("atomkit")
        from atomkit import Configuration

        filename = tmp_path / "integrated.dat"
        ground = Configuration.from_string("1s2.2s2.2p6")

        asw = ASWriter.for_structure_calculation(
            str(filename), nzion=10, coupling="IC", core=CoreSpecification.helium_like()
        )

        # Now add configurations
        asw.configs_from_atomkit([ground], last_core_orbital="1s")

        content = asw.get_content()
        # Check that configurations are present (in AS binary format)
        assert "2 0  2 1" in content  # Orbitals: 2s 2p
        assert " 2   6" in content  # Occupation: 2s2 2p6

    def test_fluent_interface_with_real_calculation(self, tmp_path):
        """Test fluent interface with complete calculation setup."""
        filename = tmp_path / "fluent_complete.dat"

        asw = (
            ASWriter(str(filename))
            .with_core(CoreSpecification.helium_like())
            .with_optimization(OptimizationParams(include_lowest=10))
        )

        asw.write_header("Fluent test")
        asw.add_salgeb(CUP="IC", RAD="E1")
        asw.add_orbitals([(1, 0), (2, 0), (2, 1)])
        asw.add_configurations([[2, 0, 0], [2, 2, 1]])
        asw.add_sminim(NZION=10)

        # Validation should pass
        warnings = asw.validate()
        assert len(warnings) == 0
