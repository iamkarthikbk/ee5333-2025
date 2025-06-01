# Detailed Router API Documentation

## Project Overview

This document describes the API for the detailed router implementation. The router takes a DEF file containing placement information, a LEF file with layer and macro information, and a GUIDE file with global routing solutions, and produces a DEF file with detailed routes for each net.

## Input/Output Specification

### Input
- A DEF file containing the placement boundary, nets, boundary pins, and cell instances' locations
- A GUIDE file containing the global routing solution
- A LEF file containing the layer and macro information

### Output
- DEF file containing the routes for each net

## Function Signature

```python
detailed_route(input_DEF, input_LEF, input_GUIDE, output_DEF)
```

## DEF Parser API

### Point Class

| Member | Type | Purpose |
|--------|------|--------|
| x | int | x coordinate |
| y | int | y coordinate |

### Rect Class

| Member | Type | Purpose |
|--------|------|--------|
| ll | Point | lower-left coordinate |
| ur | Point | upper-right coordinate |

### Pin Class

| Type | Methods | Purpose |
|------|---------|--------|
| str | name() | name of the pin |
| Point | origin() | location of the pin |
| str | print() | string containing the pin info |

### Net Class

| Type | Methods | Purpose |
|------|---------|--------|
| str | name() | Name of the net |
| (str, str) | pins() | list of pair of strings (cell name, pin name in the cell); for boundary pins cell name string is "PIN" |
| str | print() | string containing the pin info |
| (str, int, int, int, int) | addRect(layer, x1, y1, x2, y2) | add a solution shape to the net; e.g. addRect("metal1", 100, 200, 300, 400) |

### Grid Class

| Type | Methods | Purpose |
|------|---------|--------|
| str | orient() | orientation 'X' or 'Y' |
| int | x() | starting coordinate |
| int | num() | number of grids + 1 |
| int | step() | pitch of the grid |

### Component Class

| Type | Methods | Purpose |
|------|---------|--------|
| str | name() | instance name of the cell |
| Point | location() | lower left coordinate |
| str | orient() | orientation of the cell (N, S, FN, FS, ...) |
| str | macro() | type of the cell (defined in LEF file e.g. AND, INV) |
| | setLocation(int, int) | Update the current (x, y) location of the cell |
| | setOrient(str) | Update the orientation of the cell |

### DEFReader Class

| Type | Methods | Purpose |
|------|---------|--------|
| Rect | bbox() | Rectangular boundary |
| list<Pin> | pins() | list of boundary pins (Pin objects) |
| list<Net> | nets() | list of nets (Net objects) |
| list<Component> | components() | list of cells (Component objects) |
| list<Grid> | gcgrids() | list of Gcell grids (Grid objects) |
| dict<string, list<Grid>> | tracks() | dict of tracks (layer→list of Grid objects) |
| | readDEF(str) | Read the .def file specified as a string |
| | writeDEF(str) | write the updated results into the specified def file |

### Usage Example

```python
# Example usage of the detailed router
python router.py -l lef/sky130.lef -d def/c17.def -g gr/c17.guide -o output.def -v
```

Where:
- `-l`: Path to the LEF file
- `-d`: Path to the input DEF file
- `-g`: Path to the GUIDE file
- `-o`: Path to the output DEF file
- `-v`: Verbose mode (optional)
