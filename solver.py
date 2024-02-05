import numpy as np
from fractions import Fraction as Frac

DELIM = ' '

def import_mtx(filepath):
    with open(filepath, "r") as file:
        mtx_text = file.read()
    lines = [line for line in mtx_text.split('\n') if len(line) > 0]
    cols = len(lines[0].split(DELIM))
    mtx = np.zeros((len(lines), cols), dtype=Frac)

    for r, line in enumerate(lines):
        for c, val in enumerate(line.split(DELIM)):
            if "/" in val:
                v = Frac(*[int(x) for x in val.split("/")])
            else:
                v = Frac(int(val), 1)
            mtx[r][c] = v
        
    return mtx

def latex_frac(frac):
    if frac.denominator == 1:
        return str(frac.numerator)
    return "\\frac{%d}{%d}" % (frac.numerator, frac.denominator)

def latex_matrix(mtx):
    cols = mtx.shape[1]
    m = "&\\begin{amatrix}{%d}\n" % (cols-1)
    parsed_rows = []
    for row in mtx:
        parsed_rows.append(" & ".join(parse_entry(entry) for entry in row))
    m += " \\\\ ".join(parsed_rows) + "\n"
    m += "\\end{amatrix}\\\\\n"
    return m

def latex_swap_rows(from_, to):
    if from_ > to:
        return latex_swap_rows(to, from_)
    return f"&R_{from_+1}\\leftrightarrow R_{to+1}\\\\\n"

def latex_multiply_row(row, coeff):
    return "&" + latex_frac(coeff) + "R_%d\\rightarrow R_%d\\\\\n" % (row+1, row+1)

def latex_add_rows(orig_row, add_row, coeff):
    positive = coeff > 0

    parsed_coeff = "+" if positive else "-"
    if abs(coeff) != 1:
        parsed_coeff += latex_frac(coeff if positive else coeff * -1)

    return "&" + f"R_{orig_row+1}" + parsed_coeff + f"R_{add_row+1}\\rightarrow R_{orig_row+1}\\\\\n"

def perform_gaussian_elimination(mtx):
    rows, cols = mtx.shape
    pivot_col = 0
    iteration = 0

    latex = "\\begin{equation}\n\\begin{split}\n"
    latex += latex_matrix(mtx)

    # find pivot column
    while pivot_col < cols:
        pivot_row = -1
        for row in range(iteration, rows):
            if mtx[row][pivot_col] != 0:
                pivot_row = row
                break
        if pivot_row == -1:
            pivot_col += 1
            continue

        # swap to correct place
        if pivot_row != iteration:
            latex += latex_swap_rows(pivot_row, iteration)
            mtx[[pivot_row, iteration]] = mtx[[iteration, pivot_row]]
            latex += latex_matrix(mtx)
        # make this entry 1
        coeff = 1/mtx[iteration][pivot_col]
        if coeff != 1:
            latex += latex_multiply_row(iteration, coeff)
            mtx[iteration] *= coeff
            latex += latex_matrix(mtx)
        # zero the remaining entries
        for row in range(rows):
            if row == iteration:
                continue
            
            coeff = (-mtx[row][pivot_col])
            
            if coeff != 0:
                latex += latex_add_rows(row, iteration, coeff)
                mtx[row] += mtx[iteration]*coeff
                latex += latex_matrix(mtx)

        pivot_col += 1
        iteration += 1

    latex += "\\end{split}\n\\end{equation}"

    return latex

def parse_entry(entry: Frac):
    if entry.denominator == 1:
        return str(entry.numerator)
    return f"{entry.numerator}/{entry.denominator}"

def print_mtx(mtx):
    for row in mtx:
        print(" ".join(parse_entry(entry) for entry in row))


if __name__ == "__main__":
    mtx = import_mtx('mtx.txt')
    latex = perform_gaussian_elimination(mtx)

    with open("latex_out.tex", "w") as file:
        file.write(latex.strip())
