"""Symbolic parsing and solving using SymPy."""
from sympy import Eq, simplify, solve, sympify, latex as sympy_latex
from sympy.parsing.latex import parse_latex
import re
import logging

logger = logging.getLogger(__name__)


def parse_latex_to_sympy(latex_str: str):
    """Attempt to parse LaTeX string into a SymPy Eq or expression."""

    try:
        # Try parsing as equation using real LaTeX parser
        if "=" in latex_str:
            left_s, right_s = latex_str.split("=", 1)
            try:
                left = parse_latex(left_s.strip())
                right = parse_latex(right_s.strip())
                return Eq(left, right)
            except Exception:
                pass

        # Try parse single expression
        try:
            expr = parse_latex(latex_str)
            return expr
        except Exception:
            pass

    except Exception as e:
        logger.exception("LaTeX parsing failed: %s", e)

    # -----------------------------
    # Heuristic cleanup for OCR math
    # -----------------------------

    s = latex_str or ""

    # ------------------------------------------------
    # ADDED CODE (DO NOT REMOVE EXISTING CODE)
    # Fix OCR parenthesis imbalance like: (x+2)x-3)
    # ------------------------------------------------

    # remove spaces first
    s = s.replace(" ", "")

    # remove extra closing parentheses
    while s.count(")") > s.count("("):
        if s.endswith(")"):
            s = s[:-1]
        else:
            s = s.replace(")", "", 1)

    # remove extra opening parentheses
    while s.count("(") > s.count(")"):
        if s.startswith("("):
            s = s[1:]
        else:
            s = s.replace("(", "", 1)

    # ------------------------------------------------

    # Normalize common OCR issues
    s = s.replace("X", "x")
    s = s.replace("^", "**")
    s = s.replace("\u00bd", "1/2")  # ½

    # Fix missing multiplication like (x+2)(x-3)
    s = re.sub(r"\)\(", ")*(", s)

    # Fix number followed by bracket: 2(x+1)
    s = re.sub(r"(\d)\(", r"\1*(", s)

    # Fix variable followed by bracket: x(x+1)
    s = re.sub(r"([a-zA-Z])\(", r"\1*(", s)

    # Remove spaces
    s = s.replace(" ", "")

    # Insert multiplication signs
    s = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", s)      # 2x -> 2*x
    s = re.sub(r"([a-zA-Z])(\d)", r"\1*\2", s)      # x2 -> x*2
    s = re.sub(r"(\))([a-zA-Z0-9])", r"\1*\2", s)   # )(x) -> )*x
    s = re.sub(r"(\d)\(", r"\1*(", s)               # 2(x+1) -> 2*(x+1)
    s = re.sub(r"(\))\(", r"\1*(", s)               # )( -> )*(

    s = s.strip()

    if not s:
        raise ValueError("Empty expression")

    try:
        if "=" in s:
            left_s, right_s = s.split("=", 1)
            left = sympify(left_s)
            right = sympify(right_s)
            return Eq(left, right)

        expr = sympify(s)
        return expr

    except Exception as e:
        logger.exception("Heuristic sympify failed: %s", e)
        raise


def solve_equation(sym):
    """Solve an equation or expression."""

    try:
        if isinstance(sym, Eq):
            syms = list(sym.free_symbols)

            if not syms:
                return {"solutions": [], "error": "No variable found"}

            solutions = solve(sym, syms[0])
            return {"solutions": solutions}

        syms = list(sym.free_symbols)

        if syms:
            solutions = solve(sym, syms[0])
            return {"solutions": solutions}

        return {"solutions": []}

    except Exception as e:
        logger.exception("Solving failed: %s", e)
        return {"solutions": [], "error": str(e)}


def generate_steps(sym_input):
    """Generate minimal step-by-step solution."""

    steps = []

    try:
        if isinstance(sym_input, Eq):
            L = sym_input.lhs
            R = sym_input.rhs

            steps.append(("Original", sympy_latex(sym_input)))

            Ls = simplify(L)
            Rs = simplify(R)

            if Ls != L or Rs != R:
                steps.append(("Simplified", sympy_latex(Eq(Ls, Rs))))

            expr = simplify(L - R)
            steps.append(("Rearranged", sympy_latex(expr) + " = 0"))

            sol = solve(sym_input)
            steps.append(("Solve", str(sol)))

            return steps

        steps.append(("Original", sympy_latex(sym_input)))

        simp = simplify(sym_input)

        if simp != sym_input:
            steps.append(("Simplified", sympy_latex(simp)))

        syms = list(sym_input.free_symbols)

        if syms:
            sol = solve(sym_input, syms[0])
            steps.append(("Solve", str(sol)))

        return steps

    except Exception as e:
        logger.exception("Generating steps failed: %s", e)
        return [("Original", str(sym_input)), ("Error", str(e))]