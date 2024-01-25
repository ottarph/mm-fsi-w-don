FROM ghcr.io/scientificcomputing/fenics-gmsh:2023-08-16 as fenics


FROM fenics as fenics-pytorch

RUN python3 -m pip install torch


FROM fenics-pytorch as deeponet-learnext

# According to Dokken, as much as possible with pip.

# First install ARM-compiled VTK
# https://fenicsproject.discourse.group/t/pyvista-not-available-with-docker-due-to-lack-of-vtk/8813/16
RUN python3 -m pip install "https://github.com/finsberg/vtk-aarch64/releases/download/vtk-9.2.6-cp310/vtk-9.2.6.dev0-cp310-cp310-linux_aarch64.whl"

# Install pyvista
RUN python3 -m pip install pyvista

# Install everything else
RUN python3 -m pip install numpy matplotlib ipykernel tqdm pytest pyyaml

