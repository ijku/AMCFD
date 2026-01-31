# init_case.py
import taichi as ti

from data_structures import (
    PhysicsParams, SimulationParams,
    GridParams, State, StatePrev, MaterialProps, DiscretCoeffs, LaserState
)


def enthalpy_from_temp(T: float, phys: PhysicsParams) -> float:
    # Matches your Fortran-style polynomial enthalpy expression
    # H = 0.5*acpa*T^2 + acpb*T + hsmelt
    return 0.5 * phys.acpa * T * T + phys.acpb * T + phys.hsmelt


@ti.kernel
def initialize_fields(
    state: ti.template(),
    prev: ti.template(),
    props: ti.template(),
    coeffs: ti.template(),
    ni: int, nj: int, nk: int,
    viscos: float,
    denl: float,
    diff0: float,
    temp_preheat: float,
    enthalpy_preheat: float,
    enthalpy_west: float,
    enthalpy_east: float,
    enthalpy_north: float,
    enthalpy_bottom: float,
):
    # Initialize all volume fields
    for i, j, k in ti.ndrange(ni, nj, nk):
        props.vis[i, j, k] = viscos
        props.den[i, j, k] = denl
        props.diff[i, j, k] = diff0

        state.uVel[i, j, k] = 0.0
        state.vVel[i, j, k] = 0.0
        state.wVel[i, j, k] = 0.0

        prev.unot[i, j, k] = 0.0
        prev.vnot[i, j, k] = 0.0
        prev.wnot[i, j, k] = 0.0

        state.pressure[i, j, k] = 0.0
        state.pp[i, j, k] = 0.0

        state.temp[i, j, k] = temp_preheat
        state.enthalpy[i, j, k] = enthalpy_preheat
        prev.hnot[i, j, k] = enthalpy_preheat

        state.fracl[i, j, k] = 0.0

        # Discretization arrays (if you want them reset here)
        coeffs.ap[i, j, k] = 0.0
        coeffs.ae[i, j, k] = 0.0
        coeffs.aw[i, j, k] = 0.0
        coeffs.an[i, j, k] = 0.0
        coeffs.as_[i, j, k] = 0.0
        coeffs.at[i, j, k] = 0.0
        coeffs.ab[i, j, k] = 0.0
        coeffs.su[i, j, k] = 0.0
        coeffs.sp[i, j, k] = 0.0

        # Enthalpy boundary conditions (Fortran is 1-based; Python is 0-based)
        if i == 0:
            state.enthalpy[i, j, k] = enthalpy_west
        if i == ni - 1:
            state.enthalpy[i, j, k] = enthalpy_east
        if k == 0:
            state.enthalpy[i, j, k] = enthalpy_bottom
        if j == nj - 1:
            state.enthalpy[i, j, k] = enthalpy_north


def build_case():
    # You can choose arch=ti.gpu if available
    ti.init(arch=ti.gpu)

    phys = PhysicsParams()
    sim = SimulationParams()

    # Create Taichi field containers
    grid = GridParams(sim.ni, sim.nj, sim.nk)
    state = State(sim.ni, sim.nj, sim.nk)
    prev = StatePrev(sim.ni, sim.nj, sim.nk)
    props = MaterialProps(sim.ni, sim.nj, sim.nk)
    coeffs = DiscretCoeffs(sim.ni, sim.nj, sim.nk)
    laser = LaserState(sim.ni, sim.nj)

    # Derived scalars (mirrors the Fortran initialize logic)
    deltemp = phys.tliquid - phys.tsolid
    cpavg = 0.5 * ((phys.acpa * phys.tsolid + phys.acpb) + phys.acpl)
    phys.hlcal = phys.hsmelt + cpavg * deltemp  # store derived value if you want

    # Boundary temperatures (use your own config if needed)
    temp_preheat = phys.tpreheat
    temp_west = phys.tpreheat
    temp_east = phys.tpreheat
    temp_north = phys.tpreheat
    temp_bottom = phys.tpreheat

    enthalpy_preheat = enthalpy_from_temp(temp_preheat, phys)
    enthalpy_west = enthalpy_from_temp(temp_west, phys)
    enthalpy_east = enthalpy_from_temp(temp_east, phys)
    enthalpy_north = enthalpy_from_temp(temp_north, phys)
    enthalpy_bottom = enthalpy_from_temp(temp_bottom, phys)

    viscos = phys.vis0
    denl = phys.rholiq

    # Diffusivity: your Fortran used (k(T) / Cp(T)) style.
    # Here is a simple consistent version with your available parameters.
    cp_preheat = phys.acpa * temp_preheat * 0.5 + phys.acpb
    diff0 = phys.tcond / max(cp_preheat, 1e-12)

    # Fill Taichi fields
    initialize_fields(
        state, prev, props, coeffs,
        sim.ni, sim.nj, sim.nk,
        viscos, denl, diff0,
        temp_preheat, enthalpy_preheat,
        enthalpy_west, enthalpy_east, enthalpy_north, enthalpy_bottom,
    )

    # Laser state example (optional)
    laser.beam_x = 0.0
    laser.beam_y = 0.0
    laser.laser_on = False
    laser.current_segment = 0

    return phys, sim, grid, state, prev, props, coeffs, laser


if __name__ == "__main__":
    phys, sim, grid, state, prev, props, coeffs, laser = build_case()

    # Quick sanity check (reading a single cell back to Python)
    print("ni,nj,nk:", sim.ni, sim.nj, sim.nk)
    print("enthalpy(0,0,0):", state.enthalpy.to_numpy()[0, 0, 0])
    print("uVel(0,0,0):", state.uVel.to_numpy()[0, 0, 0])
