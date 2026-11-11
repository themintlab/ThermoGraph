import os
import jax.numpy as jnp


def test_imports():
    import thermograph as tg
    import zgraph as zg
    from thermograph.config import LIBRARY_DIR, UNARY_TDB_PATH
    from thermograph.nodes.sgte import SGTENode
    from thermograph.nodes.einstein import EinsteinNode, GroundStateNode
    from thermograph.prediction import PhaseBoundaryPredictor
    from thermograph.io.tdb import parse_tdb
    from thermograph.io.library import load_library

    assert tg.SGTENode is SGTENode
    assert tg.EinsteinNode is EinsteinNode
    assert tg.GroundStateNode is GroundStateNode
    assert tg.PhaseBoundaryPredictor is PhaseBoundaryPredictor
    assert tg.parse_tdb is parse_tdb
    assert tg.load_library is load_library
    assert tg.LIBRARY_DIR == LIBRARY_DIR
    assert tg.UNARY_TDB_PATH == UNARY_TDB_PATH
    assert os.path.exists(UNARY_TDB_PATH)
    assert hasattr(tg, "SI_DIA")
    assert hasattr(tg, "GE_DIA")
    assert hasattr(zg, "FactorNode")
    assert hasattr(zg, "SignalNodes")


def test_bundled_library_and_nodes():
    import thermograph as tg
    import zgraph as zg

    # 1. Load bundled SGTE node and evaluate
    cu_fcc = tg.load_library("unary/GHSERCU")
    cu_liq = tg.load_library("unary/GLIQCU")
    val_fcc = cu_fcc(jnp.array([1000.0]))
    val_liq = cu_liq(jnp.array([1000.0]))
    assert jnp.isfinite(val_fcc)
    assert jnp.isfinite(val_liq)
    # At 1000 K (below Cu melting point 1357.77 K), solid has lower G than liquid
    assert float(val_fcc) < float(val_liq)

    # 2. Test EinsteinNode and GroundStateNode
    e0 = tg.GroundStateNode(tg.SI_DIA["E_0"])
    osc = tg.EinsteinNode(tg.SI_DIA["theta_acoustic"], T_index=0)
    si_phase = e0 + 3.0 * osc
    val_si = si_phase(jnp.array([1000.0]))
    assert jnp.isfinite(val_si)

    # 3. Test TDB parser directly
    parsed = tg.parse_tdb(phases=["GHSERCU", "GLIQCU"])
    assert "GHSERCU" in parsed and "GLIQCU" in parsed
    assert jnp.allclose(parsed["GHSERCU"](jnp.array([1000.0])), val_fcc, rtol=1e-5)

    # 4. Test PhaseBoundaryPredictor with ZGraph v0.1.1 tau API
    T, mu_Si, mu_Ge = zg.SignalNodes(0, 1, 2)
    RT = tg.KB_R * T
    ge_sol = tg.load_library("unary/GHSERGE")
    ge_liq = tg.load_library("unary/GLIQGE")
    si_liq = tg.load_library("unary/GLIQSI")

    phase_dia = zg.FactorNode(jnp.eye(2), [mu_Si - si_phase, mu_Ge - ge_sol], tau=RT)
    phase_liq = zg.FactorNode(jnp.eye(2), [mu_Si - si_liq, mu_Ge - ge_liq], tau=RT)
    system = zg.FactorNode(jnp.eye(2), [phase_dia, phase_liq], tau=0.0)

    predictor = tg.PhaseBoundaryPredictor(system)
    T_vals = jnp.array([1400.0, 1500.0])
    mu_diff = jnp.linspace(-150000.0, 150000.0, 50)
    T_grid, mu_grid = jnp.meshgrid(T_vals, mu_diff, indexing="ij")
    inputs = jnp.stack([T_grid, mu_grid / 2.0, -mu_grid / 2.0], axis=-1)
    bounds = predictor.predict_compositions(inputs, mu_index=2)
    assert bounds.shape == (2, 2)
    assert jnp.all((bounds >= 0.0) & (bounds <= 1.0))

