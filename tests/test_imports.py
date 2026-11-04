def test_imports():
    import thermograph as tg
    import zgraph as zg
    from thermograph.config import LIBRARY_DIR
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
    assert hasattr(zg, "FactorNode")
    assert hasattr(zg, "SignalNodes")
