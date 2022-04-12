
# Dependencies
- [direnv](https://direnv.net/)

# Set up
Set up `venv`:
```bash
❯ ./venv.sh setup
```

Configure `direnv`:
```bash
❯ direnv allow
```

# Steps to install TeX for matplotlib
On Mac:

```bash
❯ brew install basictex && sudo tlmgr update --self && sudo tlmgr install dvipng

# Add the installation directory of tlmgr to your PATH

❯ sudo tlmgr install type1cm

❯ sudo tlmgr install dvipng texlive-latex-extra texlive-fonts-recommended cm-super
```

On Linux, replace brew with `apt-get`.

# How to run
See [exp.py](exp.py) for example usage.

# Code overview
There are 4 variables that are essential to know in order to make sense of this repo:
- `k`: The number of initial objects
- `m`: The number of storage nodes
- `C`: Service capacity at a single node. All nodes are assumed to have the same capacity.
- `G`: `k x n` object composition matrix
- `obj_to_node_map`: Map from object id's to node id's


The storage system is designed as follows:
1. `k` initial objects are expanded to `n` objects according to a *linear redundancy scheme*.
That is, `n - k` redundant objects are created by taking a linear combination of the initial `k` objects,
and added into the set of objects.
The resulting set `S` of `n` object is ordered and we refer to them with their index.
Each `obj-i` for `i = 0, ..., k-1` refers to one of the initial objects.
Each `obj-i` for `i = k, ..., n-1` refers to one of the redundant objects.

Column-`i` of `G` shows the composition of `obj-i` for `i = 0, ..., n-1`.
First `k` columns represent the initial objects and the remaining columns represent
the redundant objects.
The initial objects have systematic composition, that is, `obj-i` for `i = 0, ..., k-1`
is represented by the systematic column vector in which the `i`th index is `1` while all other indices are `0`.
Redundant objects are composed by taking a linear combination of the initial objects.
Corresponding columns of `G` represent the coefficients of the linear combination, i.e.,
`G` is an encoding matrix.

E.g.,
A set of `k = 2` initial objects `{a, b}` get expanded to `n = 3` objects `S = {a, b, a+b}`.
Here, `obj-0` refers to `a` and `obj-1` refers to `b`, and `obj-2` refers to `a+b`.
Encoding matrix `G` in this case is given as
```
0 1 1
1 0 1
```

2. The set `S` of `n` objects are distributed across `m` nodes according to a *storage scheme*.
Storage scheme is expressed by `obj_to_node_map`, which maps object id's to node id's.

E.g., In the example above, the following map
```
obj_to_node_map = {
  0: 0,
  1: 1,
  2: 2
}
```
implies that
- `obj-0` (`a`) is stored on `node-0`
- `obj-1` (`b`) is stored on `node-1`
- `obj-2` (`a+b`) is stored on `node-2`

