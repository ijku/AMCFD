# property.py
# from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import taichi as ti

# Replace `your_types_file` with your actual module name (without .py)
from data_structures import PhysicsParams, SimulationParams, GridParams, State, MaterialProps


@dataclass
class PropertyCoeffs:
    """
    Coefficients required by the Fortran property model.

    Notes:
    - Solid thermal conductivity: k_s(T) = thconsa * T + thconsb
    - Powder thermal conductivity: k_p(T) = pthcona * T + pthconb
    - Liquid thermal conductivity: thconl (constant)
    - Solid Cp: Cp_s(T) = acpa * T + acpb (from PhysicsParams)
    - Powder Cp: Cp_p(T) = pcpa * T + pcpb
    """
    thconsa: float = 0.0
    thconsb: float = 25.0
    thconl: float = 25.0

    dens: float = 7800.0     # Solid density
    denl: float = 7800.0     # Liquid density
    viscos: float = 6.0e-3   # Liquid viscosity base

    layerheight: float = 0.0 # Powder layer height measured from the top surface
    pden: float = 3900.0

    pthcona: float = 0.0
    pthconb: float = 1.0
    pcpa: float = 0.0
    pcpb: float = 500.0

    @staticmethod
    def from_physics(
        phys: PhysicsParams,
        layerheight: float = 0.0,
        thconsa: float = 0.0,
        thconsb: Optional[float] = None,
        thconl: Optional[float] = None,
    ) -> "PropertyCoeffs":
        """
        Convenience constructor when you only have PhysicsParams.

        - If thconsb/thconl are not given, they fall back to phys.tcond / phys.tcondl.
        - dens/denl/viscos fall back to phys.rho / phys.rholiq / phys.vis0.
        """
        return PropertyCoeffs(
            thconsa=float(thconsa),
            thconsb=float(phys.tcond if thconsb is None else thconsb),
            thconl=float(phys.tcondl if thconl is None else thconl),
            dens=float(phys.rho),
            denl=float(phys.rholiq),
            viscos=float(phys.vis0),
            layerheight=float(layerheight),
        )


@ti.data_oriented
class PropertyModel:
    """
    Taichi translation of Fortran:
        module property
        subroutine properties

    Updates:
        props.vis(i,j,k), props.diff(i,j,k), props.den(i,j,k)

    Requires:
        state.temp, state.fracl
        grid.z (for powder layer check)
        solidfield(i,j,k) (1.0 for solid, 0.0 for liquid or similar; threshold 0.5 used)
    """

    def __init__(
        self,
        sim: SimulationParams,
        phys: PhysicsParams,
        grid: GridParams,
        state: State,
        props: MaterialProps,
        solidfield: ti.Field,
        coeffs: PropertyCoeffs,
    ):
        self.sim = sim
        self.phys = phys
        self.grid = grid
        self.state = state
        self.props = props
        self.solidfield = solidfield
        self.coeffs = coeffs

    @ti.kernel
    def _properties_kernel(
        self,
        ni: int, nj: int, nk: int,
        tsolid: ti.f64, tliquid: ti.f64,
        acpa: ti.f64, acpb: ti.f64, acpl: ti.f64,
        thconsa: ti.f64, thconsb: ti.f64, thconl: ti.f64,
        viscos: ti.f64,
        dens: ti.f64, denl: ti.f64,
        layerheight: ti.f64,
        pden: ti.f64,
        pthcona: ti.f64, pthconb: ti.f64,
        pcpa: ti.f64, pcpb: ti.f64,
    ):
        z_top = self.grid.z[nk - 1]

        for i, j, k in ti.ndrange(ni, nj, nk):
            T = self.state.temp[i, j, k]
            fL = self.state.fracl[i, j, k]

            # Turbulent corrections are disabled in the Fortran snippet (visT=0, diffT=0)
            visT = 0.0
            diffT = 0.0

            # Solid diffusivity: diffs = k_s(T) / Cp_s(T)
            cp_s = acpa * T + acpb
            cp_s = ti.max(cp_s, 1.0e-30)
            diffs = (thconsa * T + thconsb) / cp_s

            # Liquid diffusivity: diffl = k_l / Cp_l
            diffl = thconl / ti.max(acpl, 1.0e-30)

            # Default: liquid-like assignment (matches the Fortran pre-assign before conditionals)
            vis_val = viscos + visT
            diff_val = diffl + diffT
            den_val = denl

            if T < tliquid:
                # Solid-like (below liquidus) default
                vis_val = 1.0e10
                diff_val = diffs
                den_val = dens

                # Powder properties near top surface (only retained if T <= tsolid due to Fortran flow)
                if (z_top - self.grid.z[k] <= layerheight) and (self.solidfield[i, j, k] <= 0.5):
                    den_val = pden
                    vis_val = 1.0e10
                    cp_p = pcpa * T + pcpb
                    cp_p = ti.max(cp_p, 1.0e-30)
                    diff_val = (pthcona * T + pthconb) / cp_p

                # If temperature is between solidus and liquidus, overwrite with mixture (mushy) values
                if T > tsolid:
                    diff_val = fL * diffl + (1.0 - fL) * diffs
                    vis_val = viscos + visT
                    den_val = fL * denl + (1.0 - fL) * dens

            self.props.vis[i, j, k] = vis_val
            self.props.diff[i, j, k] = diff_val
            self.props.den[i, j, k] = den_val

    def update(self) -> None:
        """Public API to update material properties for the entire domain."""
        c = self.coeffs
        p = self.phys
        s = self.sim

        self._properties_kernel(
            s.ni, s.nj, s.nk,
            p.tsolid, p.tliquid,
            p.acpa, p.acpb, p.acpl,
            c.thconsa, c.thconsb, c.thconl,
            c.viscos,
            c.dens, c.denl,
            c.layerheight,
            c.pden,
            c.pthcona, c.pthconb,
            c.pcpa, c.pcpb,
        )


# ----------------------------
# Optional helper
# ----------------------------
def make_solidfield_placeholder(sim: SimulationParams) -> ti.Field:
    """
    Create a solidfield placeholder filled with zeros.
    This is useful if you have not ported solidfield yet.
    """
    solidfield = ti.field(dtype=ti.f64, shape=(sim.ni, sim.nj, sim.nk))

    @ti.kernel
    def _fill_zero():
        for i, j, k in ti.ndrange(sim.ni, sim.nj, sim.nk):
            solidfield[i, j, k] = 0.0

    _fill_zero()
    return solidfield


# ----------------------------
# Minimal example usage
# ----------------------------
def example_usage() -> None:
    ti.init(arch=ti.cpu)

    sim = SimulationParams()
    phys = PhysicsParams()

    grid = GridParams(sim.ni, sim.nj, sim.nk)
    state = State(sim.ni, sim.nj, sim.nk)
    props = MaterialProps(sim.ni, sim.nj, sim.nk)

    # nx/ny/nz mapping (Fortran dimensions)
    nx, ny, nz = sim.ni, sim.nj, sim.nk
    print("nx, ny, nz =", nx, ny, nz)

    solidfield = make_solidfield_placeholder(sim)

    # Build a simple z coordinate for the layerheight test (replace with your real geometry init)
    @ti.kernel
    def init_z():
        for k in range(sim.nk):
            grid.z[k] = ti.cast(k, ti.f64)

    init_z()

    # Example: fill temperature and liquid fraction (replace with solver results)
    @ti.kernel
    def init_state():
        for i, j, k in ti.ndrange(sim.ni, sim.nj, sim.nk):
            state.temp[i, j, k] = 300.0
            state.fracl[i, j, k] = 0.0

    init_state()

    coeffs = PropertyCoeffs.from_physics(phys, layerheight=2.0)  # example layerheight in z units
    model = PropertyModel(sim, phys, grid, state, props, solidfield, coeffs)
    model.update()

    print("vis(0,0,0) =", props.vis.to_numpy()[0, 0, 0])


if __name__ == "__main__":
    example_usage()
