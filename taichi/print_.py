# printing.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence, TextIO, Dict, Tuple

import numpy as np

# Your existing type definitions (as you showed)
from data_structures import SimulationParams, PhysicsParams, GridParams, State, MaterialProps


@dataclass
class PrintFiles:
    """Holds output file handles similar to Fortran unit numbers."""
    unit9: Optional[TextIO] = None  # Fortran unit 9 (output.txt)


class Printing:
    """
    Python translation of Fortran module `printing` (based on the exact f90 you pasted).

    This implements:
      - OpenFiles  -> open_files()
      - StartTime  -> start_time()
      - CalTime    -> cal_time()
      - EndTime    -> end_time()
      - outputres  -> output_res()
      - Cust_Out   -> cust_out()

    Notes:
    - Fortran is 1-based indexing; Python is 0-based.
    - Fortran writes to unit 6 (stdout) and unit 9 (output.txt). We do the same.
    - VTK legacy "BINARY" is written as big-endian float32 payload.
    """

    def __init__(
        self,
        sim: SimulationParams,
        phys: PhysicsParams,
        output_dir: str = "./result",
        stdout: bool = True,
    ):
        self.sim = sim
        self.phys = phys
        self.output_dir = output_dir
        self.stdout = stdout

        # Fortran module variables
        self.itertot: int = 0
        self.niter: int = 0
        self.aAveSec: float = 0.0

        # Start/end timestamps (Python replacement of iTimeStart/iTimeEnd arrays)
        self._time_start: Optional[datetime] = None
        self._time_end: Optional[datetime] = None

        self.files = PrintFiles()
        os.makedirs(self.output_dir, exist_ok=True)

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _write_both(self, text: str) -> None:
        """Write to stdout and unit9 (if open)."""
        if self.stdout:
            print(text, end="")
        if self.files.unit9 is not None:
            self.files.unit9.write(text)
            self.files.unit9.flush()

    @staticmethod
    def _fmt_start_end(dt: datetime) -> str:
        """
        Matches Fortran:
          format(2x,'Date: ',I4,'-',I2,'-',I2,2x,'time: ',2(I2,' :'),I2,/)
        """
        return (
            f"  Date: {dt.year:4d}-{dt.month:02d}-{dt.day:02d}  "
            f"time: {dt.hour:02d} :{dt.minute:02d} :{dt.second:02d}\n"
        )

    @staticmethod
    def _default_bounds(ni: int, nj: int, nk: int) -> Tuple[slice, slice, slice]:
        """
        A safe default for interior region consistent with many CFD codes:
          i = 1..ni-2, j = 1..nj-2, k = 1..nk-2 (0-based)
        This approximates Fortran's (istatp1:iendm1, jstat:jend, kstat:nkm1).
        """
        return slice(1, max(ni - 1, 1)), slice(1, max(nj - 1, 1)), slice(1, max(nk - 1, 1))

    # ----------------------------
    # Fortran: OpenFiles
    # ----------------------------
    def open_files(self) -> None:
        """Open output file (unit 9) and write the Tecplot title placeholder."""
        os.makedirs(self.output_dir, exist_ok=True)
        path9 = os.path.join(self.output_dir, "output.txt")
        self.files.unit9 = open(path9, "w", encoding="utf-8")

        # Fortran also writes to unit 41, but that file is not shown opened in your snippet.
        # We print it to both outputs as a harmless placeholder.
        self._write_both('TITLE = "Thermo-Capillary Flow in Laser-Generated Melt Pool"\n')

    # ----------------------------
    # Fortran: StartTime
    # ----------------------------
    def start_time(self) -> None:
        """Record and print start time to stdout and unit9."""
        self._time_start = datetime.now()
        self._write_both(self._fmt_start_end(self._time_start))

    # ----------------------------
    # Fortran: CalTime
    # ----------------------------
    def cal_time(self) -> None:
        """
        Compute average seconds per iteration.
        Fortran uses date_and_time arrays and manual arithmetic; Python uses datetime delta.
        """
        if self._time_start is None:
            return
        self._time_end = datetime.now()
        sec_used = (self._time_end - self._time_start).total_seconds()
        self.aAveSec = float(sec_used) / float(self.itertot) if self.itertot > 0 else 0.0

    # ----------------------------
    # Fortran: EndTime
    # ----------------------------
    def end_time(self) -> None:
        """Print end time, total used time, and close unit9."""
        self._time_end = datetime.now()
        self._write_both(self._fmt_start_end(self._time_end))

        if self._time_start is not None:
            total_sec = int((self._time_end - self._time_start).total_seconds())
            hr = total_sec // 3600
            mn = (total_sec % 3600) // 60
            sc = total_sec % 60
            self._write_both(f"  Total time used:{hr:6d}  hr{mn:6d}  m{sc:6d}  s\n")

        if self.files.unit9 is not None:
            self.files.unit9.close()
            self.files.unit9 = None

    # ----------------------------
    # Fortran: outputres
    # ----------------------------
    def output_res(
        self,
        timet: float,
        niter: int,
        itertot: int,
        tpeak: float,
        state: State,
        alen: float,
        depth: float,
        width: float,
        residuals: Dict[str, float],
        fluxes: Dict[str, float],
        heat_terms: Dict[str, float],
        ratio: float,
        coordhistory_row0: Optional[Sequence[float]] = None,
        bounds: Optional[Tuple[slice, slice, slice]] = None,
    ) -> None:
        """
        Equivalent of Fortran subroutine outputres.

        Parameters that come from other Fortran modules are passed as dictionaries:
          residuals: resorh, resorm, resoru, resorv, resorw
          fluxes: flux_north, flux_south, flux_bottom, flux_west, flux_east
          heat_terms: ahtoploss, heatout, accul, heatinLaser, heatvol

        bounds optionally controls the region used for umax/vmax/wmax.
        """
        self.niter = int(niter)
        self.itertot = int(itertot)

        # Compute umax/vmax/wmax only if liquid exists (tpeak > tsolid)
        if tpeak > self.phys.tsolid:
            u = state.uVel.to_numpy()
            v = state.vVel.to_numpy()
            w = state.wVel.to_numpy()

            if bounds is None:
                si, sj, sk = self._default_bounds(u.shape[0], u.shape[1], u.shape[2])
            else:
                si, sj, sk = bounds

            umax = float(np.max(np.abs(u[si, sj, sk])))
            vmax = float(np.max(np.abs(v[si, sj, sk])))
            wmax = float(np.max(np.abs(w[si, sj, sk])))
        else:
            umax = vmax = wmax = 0.0

        resorh = float(residuals.get("resorh", 0.0))
        resorm = float(residuals.get("resorm", 0.0))
        resoru = float(residuals.get("resoru", 0.0))
        resorv = float(residuals.get("resorv", 0.0))
        resorw = float(residuals.get("resorw", 0.0))

        flux_north = float(fluxes.get("flux_north", 0.0))
        flux_south = float(fluxes.get("flux_south", 0.0))
        flux_bottom = float(fluxes.get("flux_bottom", 0.0))
        flux_west = float(fluxes.get("flux_west", 0.0))
        flux_east = float(fluxes.get("flux_east", 0.0))

        ahtoploss = float(heat_terms.get("ahtoploss", 0.0))
        heatout = float(heat_terms.get("heatout", 0.0))
        accul = float(heat_terms.get("accul", 0.0))
        heatinLaser = float(heat_terms.get("heatinLaser", 0.0))
        heatvol = float(heat_terms.get("heatvol", 0.0))

        # Format blocks (match the Fortran intent)
        self._write_both(
            "  time  iter  time/iter  tot_iter  res_enth  res_mass     res_u   res_v   res_w\n"
        )
        self._write_both(
            f"{timet:9.2e} {self.niter:4d}  {self.aAveSec:7.3f}   {self.itertot:7d}  "
            f"{resorh:8.1e}  {resorm:8.1e} {resoru:8.1e} {resorv:8.1e} {resorw:8.1e}\n"
        )

        self._write_both(
            "  Tmax        umax       vmax         wmax       length       depth     width\n"
        )
        self._write_both(
            f"{tpeak:9.2f}  {umax:9.2e}   {vmax:9.2e}   {wmax:9.2e}   "
            f"{alen:9.2e}   {depth:9.2e}   {width:9.2e}\n"
        )

        self._write_both(
            "  north  south  top  toploss  bottom  west  east  hout  accu  hin heatvol ratio\n"
        )
        # Fortran prints `ahtoploss` twice (top and toploss). We follow that.
        self._write_both(
            f"{flux_north:7.1f}{flux_south:7.1f}{ahtoploss:7.1f}{ahtoploss:7.1f}"
            f"{flux_bottom:7.1f}{flux_west:7.1f}{flux_east:7.1f}"
            f"{heatout:7.1f}{accul:7.1f}{heatinLaser:6.1f}{heatvol:6.1f}{ratio:7.2f}\n"
        )

        # coordhistory(1,1..8) in Fortran -> first row in Python
        if coordhistory_row0 is not None and len(coordhistory_row0) >= 8:
            ch = coordhistory_row0[:8]
            self._write_both(
                "  time    beam_posx  beam_posy  beam_posz  power  scanspeed   speedx   speedy\n"
            )
            self._write_both(
                f"{ch[0]:9.2e}{ch[1]:11.3e}{ch[2]:11.3e}{ch[3]:11.3e}"
                f"{ch[4]:7.1f}{ch[5]:9.3f}{ch[6]:9.3f}{ch[7]:9.3f}\n"
            )

    # ----------------------------
    # Fortran: Cust_Out (VTK legacy)
    # ----------------------------
    def cust_out(
        self,
        timet: float,
        delt: float,
        grid: GridParams,
        state: State,
        props: MaterialProps,
        solidfield: Optional[np.ndarray] = None,
        output_interval: int = 5,
        filename_prefix: str = "vtkmov",
    ) -> None:
        """
        Equivalent of Fortran subroutine Cust_Out.

        - Writes every `output_interval` steps:
            if (mod(int(timet/delt), output_interval) != 0) return
        - Uses interior nodes only:
            i = 1..ni-2, j = 1..nj-2, k = 1..nk-2  (0-based)
        - Outputs STRUCTURED_GRID with POINTS and POINT_DATA (Velocity, T, vis, diff, den, solidID).
        """
        step = int(timet / delt)
        if output_interval <= 0:
            return
        if (step % output_interval) != 0:
            return

        outputnum = step // output_interval
        vtk_path = os.path.join(self.output_dir, f"{filename_prefix}{outputnum}.vtk")

        ni, nj, nk = self.sim.ni, self.sim.nj, self.sim.nk
        if ni < 3 or nj < 3 or nk < 3:
            raise ValueError("Grid is too small for interior output (need at least 3 cells per dimension).")

        gridx, gridy, gridz = ni - 2, nj - 2, nk - 2
        npts = gridx * gridy * gridz

        # Pull Taichi fields to numpy
        x = grid.x.to_numpy()
        y = grid.y.to_numpy()
        z = grid.z.to_numpy()

        u = state.uVel.to_numpy()
        v = state.vVel.to_numpy()
        w = state.wVel.to_numpy()
        T = state.temp.to_numpy()

        vis = props.vis.to_numpy()
        diff = props.diff.to_numpy()
        den = props.den.to_numpy()

        if solidfield is None:
            solidfield = np.zeros((ni, nj, nk), dtype=np.float64)

        # Interior slices (Fortran: 2..nim1 => Python: 1..-2)
        si = slice(1, ni - 1)
        sj = slice(1, nj - 1)
        sk = slice(1, nk - 1)

        # Fortran:
        #   auvl(i,j,k)=(uVel(i,j,k)+uVel(i+1,j,k))*0.5
        # Similar for v and w
        au = 0.5 * (u[si, sj, sk] + u[2:, 1:-1, 1:-1])
        av = 0.5 * (v[si, sj, sk] + v[1:-1, 2:, 1:-1])
        aw = 0.5 * (w[si, sj, sk] + w[1:-1, 1:-1, 2:])

        # Zero velocities where solid (T <= tsolid)
        mask_solid = (T[si, sj, sk] <= self.phys.tsolid)
        au = np.where(mask_solid, 0.0, au)
        av = np.where(mask_solid, 0.0, av)
        aw = np.where(mask_solid, 0.0, aw)

        # Build point coordinates in the same nested loop order as Fortran:
        # do k=2..nkm1; do j=2..njm1; do i=2..nim1
        xs = x[1:-1]
        ys = y[1:-1]
        zs = z[1:-1]
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")  # shape (gridx,gridy,gridz)

        # Flatten with Fortran order so i varies fastest, then j, then k
        coords = np.empty((npts, 3), dtype=">f4")
        coords[:, 0] = X.reshape(-1, order="F").astype(">f4")
        coords[:, 1] = Y.reshape(-1, order="F").astype(">f4")
        coords[:, 2] = Z.reshape(-1, order="F").astype(">f4")

        vec = np.empty((npts, 3), dtype=">f4")
        vec[:, 0] = au.reshape(-1, order="F").astype(">f4")
        vec[:, 1] = av.reshape(-1, order="F").astype(">f4")
        vec[:, 2] = aw.reshape(-1, order="F").astype(">f4")

        T_out = T[si, sj, sk].reshape(-1, order="F").astype(">f4")
        vis_out = vis[si, sj, sk].reshape(-1, order="F").astype(">f4")
        diff_out = diff[si, sj, sk].reshape(-1, order="F").astype(">f4")
        den_out = den[si, sj, sk].reshape(-1, order="F").astype(">f4")
        solid_out = solidfield[si, sj, sk].reshape(-1, order="F").astype(">f4")

        os.makedirs(self.output_dir, exist_ok=True)

        with open(vtk_path, "wb") as f:
            # ASCII header
            f.write(b"# vtk DataFile Version 3.0\n")
            f.write(b"AMCFD Simulation Output\n")
            f.write(b"BINARY\n")
            f.write(b"DATASET STRUCTURED_GRID\n")
            f.write(f"DIMENSIONS {gridx} {gridy} {gridz}\n".encode("ascii"))
            f.write(f"POINTS {npts} float\n".encode("ascii"))

            # Binary coordinates
            f.write(coords.tobytes(order="C"))

            # POINT_DATA header
            f.write(f"\nPOINT_DATA {npts}\n".encode("ascii"))

            # Velocity vectors
            f.write(b"\nVECTORS Velocity float\n")
            f.write(vec.tobytes(order="C"))

            # Scalars
            f.write(b"\nSCALARS T float 1\nLOOKUP_TABLE default\n")
            f.write(T_out.tobytes(order="C"))

            f.write(b"\nSCALARS vis float 1\nLOOKUP_TABLE default\n")
            f.write(vis_out.tobytes(order="C"))

            f.write(b"\nSCALARS diff float 1\nLOOKUP_TABLE default\n")
            f.write(diff_out.tobytes(order="C"))

            f.write(b"\nSCALARS den float 1\nLOOKUP_TABLE default\n")
            f.write(den_out.tobytes(order="C"))

            f.write(b"\nSCALARS solidID float 1\nLOOKUP_TABLE default\n")
            f.write(solid_out.tobytes(order="C"))

# # printing.py
# from __future__ import annotations

# import os
# import time
# from dataclasses import dataclass
# from datetime import datetime
# from typing import Optional, Sequence, TextIO, Dict, Any

# import numpy as np
# import taichi as ti

# # Import your existing type definitions
# from data_structures import SimulationParams, PhysicsParams, GridParams, State, MaterialProps, LaserState


# @dataclass
# class PrintFiles:
#     """Holds output file handles similar to Fortran unit numbers."""
#     unit9: Optional[TextIO] = None  # Fortran unit 9 (output.txt)


# class Printing:
#     """
#     Python translation of Fortran module `printing`.

#     Notes:
#     - Fortran arrays are 1-based; Python is 0-based.
#     - The Fortran code writes both to stdout (unit 6) and to a log file (unit 9).
#     - VTK legacy binary requires big-endian float32.
#     """

#     def __init__(
#         self,
#         sim: SimulationParams,
#         phys: PhysicsParams,
#         output_dir: str = "./result",
#         stdout: bool = True,
#     ):
#         self.sim = sim
#         self.phys = phys
#         self.output_dir = output_dir
#         self.stdout = stdout

#         # Fortran variables: itertot, niter, aAveSec
#         self.itertot: int = 0
#         self.niter: int = 0
#         self.aAveSec: float = 0.0

#         # Fortran time arrays iTimeStart(8), iTimeEnd(8)
#         self._time_start: Optional[datetime] = None
#         self._time_end: Optional[datetime] = None

#         self.files = PrintFiles()

#         os.makedirs(self.output_dir, exist_ok=True)

#     # ----------------------------
#     # Utility / IO helpers
#     # ----------------------------
#     def _write_both(self, text: str) -> None:
#         """Write to stdout and unit9 if open."""
#         if self.stdout:
#             print(text, end="")
#         if self.files.unit9 is not None:
#             self.files.unit9.write(text)
#             self.files.unit9.flush()

#     @staticmethod
#     def _fmt_date_time(dt: datetime) -> str:
#         """Format similar to the Fortran format statement."""
#         # Fortran: 'Date: YYYY-MM-DD  time: HH :MM :SS'
#         return f"  Date: {dt.year:4d}-{dt.month:02d}-{dt.day:02d}  time: {dt.hour:02d} :{dt.minute:02d} :{dt.second:02d}\n"

#     @staticmethod
#     def _interior_slices() -> tuple[slice, slice, slice]:
#         """
#         Fortran interior loops: i=2..nim1, j=2..njm1, k=2..nkm1 (1-based).
#         Python 0-based interior becomes: 1..-2 => slice(1, -1).
#         """
#         return slice(1, -1), slice(1, -1), slice(1, -1)
    
#         # ----------------------------
#     # Fortran subroutine StartTime
#     # ----------------------------
#     def start_time(self) -> None:
#         """Record and print start time."""
#         self._time_start = datetime.now()
#         msg = self._fmt_date_time(self._time_start)
#         self._write_both(msg)

#     # ----------------------------
#     # Fortran subroutine CalTime
#     # ----------------------------
#     def cal_time(self) -> None:
#         """Compute average seconds per iteration."""
#         if self._time_start is None:
#             return
#         self._time_end = datetime.now()
#         sec_used = (self._time_end - self._time_start).total_seconds()
#         if self.itertot > 0:
#             self.aAveSec = float(sec_used) / float(self.itertot)
#         else:
#             self.aAveSec = 0.0

#     # ----------------------------
#     # Fortran subroutine EndTime
#     # ----------------------------
#     def end_time(self) -> None:
#         """Print end time, total used time, and close unit9."""
#         self._time_end = datetime.now()
#         msg = self._fmt_date_time(self._time_end)
#         self._write_both(msg)

#         if self._time_start is not None:
#             dt = self._time_end - self._time_start
#             total_sec = int(dt.total_seconds())
#             hr = total_sec // 3600
#             mn = (total_sec % 3600) // 60
#             sc = total_sec % 60
#             self._write_both(f"  Total time used:{hr:6d}  hr{mn:6d}  m{sc:6d}  s\n")

#         if self.files.unit9 is not None:
#             self.files.unit9.close()
#             self.files.unit9 = None


#     # ----------------------------
#     # Fortran subroutine OpenFiles
#     # ----------------------------
#     def open_files(self) -> None:
#         """Open output log file (Fortran unit 9)."""
#         os.makedirs(self.output_dir, exist_ok=True)
#         path = os.path.join(self.output_dir, "output.txt")
#         self.files.unit9 = open(path, "w", encoding="utf-8")

#         # Fortran writes to unit 41 a Tecplot title; we kee
