
# Steps to install TeX for matplotlib
On Mac:

```bash
❯ brew install basictex && sudo tlmgr update --self && sudo tlmgr install dvipng

# Add the installation directory of tlmgr to your PATH

❯ sudo tlmgr install type1cm

❯ sudo tlmgr install dvipng texlive-latex-extra texlive-fonts-recommended cm-super
```

On Linux, replace brew with `apt-get`.
