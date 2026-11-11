import ast
import os
import re
from zgraph import save
from zgraph.core import TropicalPolynomialNode, PiecewiseNode
from thermograph.nodes.sgte import SGTESingleNode
from thermograph.config import LIBRARY_DIR, UNARY_TDB_PATH
from typing import Dict, Any, List, Optional
import jax.numpy as jnp

TERM_ORDER = ["1", "T", "T*LN(T)", "T**2", "T**-1", "T**3", "T**7", "T**-9"]


class TDBParser:
    """
    A minimal CALPHAD TDB parser that extracts FUNCTION definitions and 
    builds hierarchical ZGraph nodes, preserving all physical dependencies 
    (like GHSERSI) as exact subgraphs for perfect gradient flow.
    """
    def __init__(self, tdb_text):
        self.functions = self._parse_functions(tdb_text)
        self.node_registry = {}

    def _parse_functions(self, tdb_text):
        functions = {}
        blocks = re.split(r'\bFUNCTION\b', tdb_text, flags=re.IGNORECASE)[1:]
        
        for block in blocks:
            block = block.split("!")[0]
            segments = block.split(";")
            
            first_seg = segments[0].strip()
            parts = first_seg.split(None, 2)
            if len(parts) < 2: continue
            name = parts[0].upper()
            low_T = float(parts[1])
            
            expr = parts[2] if len(parts) > 2 else ""
            
            pieces = []
            for i in range(len(segments) - 1):
                if i > 0:
                    expr = segments[i].strip()
                
                next_seg = segments[i+1].strip()
                if not next_seg: continue
                
                ns_parts = next_seg.split(None, 2)
                t_max = float(ns_parts[0])
                
                if len(ns_parts) > 1 and ns_parts[1].upper() in ['Y', 'N']:
                    segments[i+1] = ns_parts[2] if len(ns_parts) > 2 else ""
                else:
                    segments[i+1] = ns_parts[1] if len(ns_parts) > 1 else ""
                    
                pieces.append({'t_max': t_max, 'expr': expr.strip()})
                
            functions[name] = {
                'low_T': low_T,
                'pieces': pieces
            }
        return functions

    def _extract_coeffs(self, expr_str):
        s = re.sub(r'\s+', '', expr_str.upper()).replace('LOG(T)', 'LN(T)')
        if not s:
            return [0.0] * 8, {}

        tree = ast.parse(s, mode='eval')

        def eval_node(node):
            if isinstance(node, ast.Expression):
                return eval_node(node.body)
            if isinstance(node, ast.Constant):
                return {"1": float(node.value)}
            if isinstance(node, ast.Name):
                return {node.id: 1.0}
            if isinstance(node, ast.UnaryOp):
                sub = eval_node(node.operand)
                if isinstance(node.op, ast.UAdd):
                    return sub
                if isinstance(node.op, ast.USub):
                    return {k: -v for k, v in sub.items()}
            if isinstance(node, ast.BinOp):
                left = eval_node(node.left)
                right = eval_node(node.right)
                if isinstance(node.op, (ast.Add, ast.Sub)):
                    sgn = 1.0 if isinstance(node.op, ast.Add) else -1.0
                    out = dict(left)
                    for k, v in right.items():
                        out[k] = out.get(k, 0.0) + sgn * v
                    return out
                if isinstance(node.op, ast.Mult):
                    out = {}
                    for k1, v1 in left.items():
                        for k2, v2 in right.items():
                            if k1 == "1":
                                k = k2
                            elif k2 == "1":
                                k = k1
                            else:
                                k = f"{k1}*{k2}"
                            out[k] = out.get(k, 0.0) + v1 * v2
                    return out
                if isinstance(node.op, ast.Pow):
                    exp = int(round(right["1"]))
                    return {f"T**{exp}": 1.0}
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "LN":
                    return {"LN(T)": 1.0}
            raise ValueError(f"Unsupported expression node in TDB: {expr_str}")

        terms_dict = eval_node(tree)
        coeffs = [float(terms_dict.get(term, 0.0)) for term in TERM_ORDER]
        deps = {}
        for term, c in terms_dict.items():
            if term not in TERM_ORDER and c != 0.0:
                if not term.isidentifier() or term == "T":
                    raise ValueError(f"Unsupported polynomial term '{term}' in SGTE 8-term kernel.")
                deps[term.upper()] = float(c)
        return coeffs, deps

    def get_node(self, name):
        name = name.upper()
        if name in self.node_registry:
            return self.node_registry[name]
            
        if name not in self.functions:
            raise KeyError(f"Phase/Function '{name}' not found in TDB.")
            
        func = self.functions[name]
        
        subgraphs = []
        bounds = []
        for piece in func['pieces']:
            coeffs, deps = self._extract_coeffs(piece['expr'])
            base = SGTESingleNode(coeffs, T_index=0)
            
            if not deps:
                sub = base
            else:
                subs = [base]
                weights = [1.0]
                for dep_name, mult in deps.items():
                    subs.append(self.get_node(dep_name))
                    weights.append(mult)
                sub = TropicalPolynomialNode(M_matrix=jnp.expand_dims(jnp.array(weights, dtype=jnp.float32), 0), subgraph_list=subs)
                
            subgraphs.append(sub)
            bounds.append(piece['t_max'])
            
        bounds = bounds[:-1] # PiecewiseNode takes N-1 bounds
        
        if len(bounds) == 0:
            node = subgraphs[0]
        else:
            node = PiecewiseNode(bounds, subgraphs, signal_index=0)
            
        self.node_registry[name] = node
        return node


def parse_tdb(tdb_path: Optional[str] = None, phases: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Reads a CALPHAD .tdb file and parses functional dependencies as exact ZGraph
    hierarchical subgraphs. Returns a dictionary of compiled nodes.
    """
    if tdb_path is None or (not os.path.exists(tdb_path) and os.path.basename(tdb_path) == "unary.tdb"):
        tdb_path = UNARY_TDB_PATH

    with open(tdb_path, 'r') as f:
        text = f.read()
        
    parser = TDBParser(text)
    
    if phases is None:
        phases = list(parser.functions.keys())
        
    nodes = {}
    for phase_name in phases:
        try:
            node = parser.get_node(phase_name)
            nodes[phase_name] = node
        except Exception as e:
            print(f"Warning: Failed to extract {phase_name} - {e}")
            
    return nodes


# Backwards compatibility wrapper
def extract_sgte_library(tdb_path: Optional[str] = None, output_dir: str = None, phases: list = None):
    if tdb_path is None or (not os.path.exists(tdb_path) and os.path.basename(tdb_path) == "unary.tdb"):
        tdb_path = UNARY_TDB_PATH
    nodes = parse_tdb(tdb_path, phases)
    
    if output_dir is None:
        tdb_name = os.path.splitext(os.path.basename(tdb_path))[0]
        output_dir = os.path.join(LIBRARY_DIR, tdb_name)
        
    save(nodes, output_dir)
    print(f"Successfully serialized {len(nodes)} nodes to {output_dir}")
    return list(nodes.keys())

