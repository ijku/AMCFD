# laserinput.py
# from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import taichi as ti

# Replace `your_types_file` with your actual module name (without .py)
from data_structures import SimulationParams, GridParams, LaserParams, ToolPath, LaserState


@dataclass
class PoolIndexBox:
    """Index box used as an initial guess for melt pool search bounds."""
    istart: int
    jstart: int
    imin: int
    imax: int
    jmin: int
    jmax: int
    kmin: int
    kmax: int


@ti.data_oriented
class LaserInput:
    """
    Python translation of Fortran module `laserinput`.

    Mapping notes:
    - Fortran nx,ny,nz are obtained from SimulationParams.ni,nj,nk.
    - toolmatrix(:,1:5) is mapped to ToolPath arrays:
        time -> toolpath.time
        x    -> toolpath.x
        y    -> toolpath.y
        z    -> toolpath.z
        on   -> toolpath.laser_on (0/1)
    - heatin(nx,ny) is LaserState.heatin (ti.field).
    - heatinLaser is stored in an internal scalar ti.field and also copied to LaserState.heat_total.
    """

    def __init__(
        self,
        sim: SimulationParams,
        grid: GridParams,
        laser_params: LaserParams,
        toolpath: ToolPath,
        laser_state: LaserState,
        gaussian_factor: float = 2.0,
    ):
        self.sim = sim
        self.grid = grid
        self.laser_params = laser_params
        self.toolpath = toolpath
        self.laser_state = laser_state

        # Gaussian exponent factor (Fortran uses alasfact in exp(-alasfact*dist2/rb2))
        # For a 1/e^2 radius model, a common choice is 2.0.
        self.gaussian_factor = float(gaussian_factor)

        # Scalar field to accumulate total heat input (heatinLaser)
        self._heatin_laser = ti.field(dtype=ti.f64, shape=())

        # Fortran-style counters
        self.path_num: int = 1  # Fortran PathNum (1-based concept)
        self.track_num: int = 0

        # Cache numpy copies for index search (CPU side)
        self._x_np: Optional[np.ndarray] = None
        self._y_np: Optional[np.ndarray] = None

    def _ensure_xy_cache(self) -> None:
        """Cache x and y coordinates as numpy arrays for fast nearest-index search."""
        if self._x_np is None:
            self._x_np = self.grid.x.to_numpy()
        if self._y_np is None:
            self._y_np = self.grid.y.to_numpy()

    def init_from_toolpath(self, start_segment: int = 0) -> None:
        """
        Initialize beam position and segment index from toolpath.
        This replaces Fortran initialization that sets beam_pos/toolmatrix(PathNum,2/3).
        """
        if self.toolpath.n_segments <= 0:
            raise ValueError("ToolPath has no segments.")

        start_segment = int(np.clip(start_segment, 0, self.toolpath.n_segments - 1))
        self.path_num = start_segment + 1  # keep a Fortran-like 1-based counter

        self.laser_state.current_segment = start_segment
        self.laser_state.beam_x = float(self.toolpath.x[start_segment])
        self.laser_state.beam_y = float(self.toolpath.y[start_segment])
        self.laser_state.beam_z = float(self.toolpath.z[start_segment]) if self.toolpath.z is not None else 0.0
        self.laser_state.laser_on = bool(self.toolpath.laser_on[start_segment] >= 0.5)

    def _maybe_advance_segment(self, timet: float) -> None:
        """
        Update PathNum and TrackNum using Fortran logic:

        if(timet .gt. toolmatrix(PathNum,1) .and. toolmatrix(PathNum+1,1) .ge. -0.5) then
            PathNum = PathNum + 1
            if(toolmatrix(PathNum,5) .ge. 0.5) TrackNum = TrackNum + 1
        endif
        """
        seg = int(self.laser_state.current_segment)
        nseg = int(self.toolpath.n_segments)
        if nseg <= 1:
            return

        # Fortran checks toolmatrix(PathNum+1,1) >= -0.5 as a valid next segment sentinel.
        if seg + 1 < nseg:
            next_time = float(self.toolpath.time[seg + 1])
            cur_time = float(self.toolpath.time[seg])
            if (timet > cur_time) and (next_time >= -0.5):
                seg += 1
                self.laser_state.current_segment = seg
                self.path_num = seg + 1

                if float(self.toolpath.laser_on[seg]) >= 0.5:
                    self.track_num += 1

    def _update_scan_velocity_and_beam(self) -> None:
        """
        Compute scanvelx/scanvely and advance beam position:
            scanvelx = (x(seg)-x(seg-1)) / (t(seg)-t(seg-1))
            beam_pos  += delt * scanvelx
        """
        seg = int(self.laser_state.current_segment)
        if seg <= 0:
            self.laser_state.scanvel_x = 0.0
            self.laser_state.scanvel_y = 0.0
            return

        t1 = float(self.toolpath.time[seg])
        t0 = float(self.toolpath.time[seg - 1])
        dt = t1 - t0
        if abs(dt) < 1e-30:
            vx = 0.0
            vy = 0.0
        else:
            vx = (float(self.toolpath.x[seg]) - float(self.toolpath.x[seg - 1])) / dt
            vy = (float(self.toolpath.y[seg]) - float(self.toolpath.y[seg - 1])) / dt

        self.laser_state.scanvel_x = vx
        self.laser_state.scanvel_y = vy

        self.laser_state.beam_x += self.sim.delt * vx
        self.laser_state.beam_y += self.sim.delt * vy

        # Optional: keep beam_z locked to the toolpath z of current segment
        if self.toolpath.z is not None:
            self.laser_state.beam_z = float(self.toolpath.z[seg])

    def _nearest_interior_index(self, coord: np.ndarray, value: float) -> int:
        """
        Find nearest index in the interior range [1, n-2].
        This matches the Fortran search loops i=2..nim1 (1-based).
        """
        n = coord.shape[0]
        if n < 3:
            return int(np.clip(int(round(value)), 0, max(n - 1, 0)))

        interior = coord[1:-1]
        local = int(np.argmin(np.abs(interior - value)))
        return local + 1  # shift back to full-array indexing

    @ti.kernel
    def _compute_heatin_kernel(
        self,
        beam_x: ti.f64,
        beam_y: ti.f64,
        radius: ti.f64,
        peak_flux: ti.f64,
        gaussian_factor: ti.f64,
        laser_on: ti.i32,
    ):
        """
        Compute heatin(i,j) and heatinLaser:
            heatin(i,j) = peak * exp(-factor * dist2 / r^2)
            heatinLaser = sum(areaij(i,j) * heatin(i,j))
        """
        self._heatin_laser[None] = 0.0
        rb2 = radius * radius

        for i, j in ti.ndrange(self.sim.ni, self.sim.nj):
            xdist = beam_x - self.grid.x[i]
            ydist = beam_y - self.grid.y[j]
            dist2 = xdist * xdist + ydist * ydist

            q = 0.0
            if laser_on == 1:
                q = peak_flux * ti.exp(-gaussian_factor * dist2 / rb2)

            self.laser_state.heatin[i, j] = q
            self._heatin_laser[None] += self.grid.areaij[i, j] * q

    def laser_beam(self, timet: float) -> Tuple[float, PoolIndexBox]:
        """
        Main entry: Python equivalent of Fortran subroutine laser_beam.

        Returns:
            heatin_laser (float): total heat input (heatinLaser)
            pool_box (PoolIndexBox): initial pool bounds (imin/imax/jmin/jmax/kmin/kmax)
        """
        timet = float(timet)

        # Step 1: update segment index and track counter
        self._maybe_advance_segment(timet)

        # Step 2: update scan velocity and advance beam location
        self._update_scan_velocity_and_beam()

        # Step 3: find nearest interior indices to beam center
        self._ensure_xy_cache()
        istart = self._nearest_interior_index(self._x_np, self.laser_state.beam_x)
        jstart = self._nearest_interior_index(self._y_np, self.laser_state.beam_y)

        # Step 4: compute heat input field
        seg = int(self.laser_state.current_segment)
        laser_on = 1 if float(self.toolpath.laser_on[seg]) >= 0.5 else 0

        # Effective peak flux:
        # LaserParams.peak_flux already computes: 2*power*absorptivity*efficiency/(pi*r^2)
        # This matches a common Gaussian model with gaussian_factor=2.
        self._compute_heatin_kernel(
            beam_x=float(self.laser_state.beam_x),
            beam_y=float(self.laser_state.beam_y),
            radius=float(self.laser_params.radius),
            peak_flux=float(self.laser_params.peak_flux) if laser_on == 1 else 0.0,
            gaussian_factor=float(self.gaussian_factor),
            laser_on=laser_on,
        )

        heatin_laser = float(self._heatin_laser.to_numpy()[()])

        # Copy to LaserState (Python scalar)
        self.laser_state.heat_total = heatin_laser
        self.laser_state.laser_on = bool(laser_on == 1)

        # Step 5: initial pool bounds (Fortran-style), clamped to valid range
        ni, nj, nk = self.sim.ni, self.sim.nj, self.sim.nk

        imin = int(max(istart - 2, 0))
        imax = int(min(istart + 2, ni - 1))

        jmin = int(max(jstart - 2, 0))
        jmax = int(min(jstart + 2, nj - 1))

        # Fortran: kmin = nk - 4 (1-based), kmax = nkm1 = nk - 1 (1-based)
        # Convert to 0-based approximately near the top interior:
        kmin = int(max((nk - 4) - 1, 0))     # nk-5
        kmax = int(min((nk - 1) - 1, nk - 1))  # nk-2

        pool_box = PoolIndexBox(
            istart=istart,
            jstart=jstart,
            imin=imin,
            imax=imax,
            jmin=jmin,
            jmax=jmax,
            kmin=kmin,
            kmax=kmax,
        )

        return heatin_laser, pool_box


# ----------------------------
# Minimal usage example
# ----------------------------
def example_usage() -> None:
    ti.init(arch=ti.cpu)

    sim = SimulationParams()
    phys = PhysicsParams()  # not required here, but usually exists in your code
    grid = GridParams(sim.ni, sim.nj, sim.nk)

    # nx/ny/nz mapping
    nx, ny, nz = sim.ni, sim.nj, sim.nk
    print("nx, ny, nz =", nx, ny, nz)

    # Dummy toolpath (replace with your real CRS loader)
    toolpath = ToolPath(
        time=np.array([0.0, 1e-4, 2e-4], dtype=np.float64),
        x=np.array([0.0, 1e-4, 2e-4], dtype=np.float64),
        y=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        z=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        laser_on=np.array([1, 1, 1], dtype=np.int32),
        n_segments=3,
    )

    laser_params = LaserParams()
    laser_state = LaserState(sim.ni, sim.nj)

    laser_input = LaserInput(sim, grid, laser_params, toolpath, laser_state)
    laser_input.init_from_toolpath(start_segment=0)

    # Build a trivial areaij for demonstration (replace with your real geometry setup)
    @ti.kernel
    def fill_areaij():
        for i, j in ti.ndrange(sim.ni, sim.nj):
            grid.areaij[i, j] = 1.0

    fill_areaij()

    heatin_laser, pool_box = laser_input.laser_beam(timet=0.0)
    print("heatinLaser =", heatin_laser)
    print("pool_box =", pool_box)


if __name__ == "__main__":
    example_usage()
