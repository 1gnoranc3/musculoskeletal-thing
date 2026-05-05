import numpy as np
import casadi as ca
import time as time_module
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class CostFunction(Enum):
    SUM_SQUARES      = "sum_sq"
    SUM_CUBES        = "sum_cu"
    POLYNOMIAL_3     = "poly3"
    METABOLIC_PROXY  = "metabolic"
    SUM_CUBES_STRESS = "stress3"
    SUM_POW_25       = "pow25"
    SUM_CUBES_PENRES = "cu_penres"


@dataclass
class ModelParams:
    n_muscles: int
    n_dof: int
    max_isometric_force: np.ndarray
    moment_arms: np.ndarray
    moment_arms_precomputed: bool = False


@dataclass
class OptimizationConfig:
    cost_function: CostFunction = CostFunction.SUM_SQUARES
    reserve_actuator_weight: float = 20.0
    reserve_weights_per_dof: np.ndarray = None
    activation_lower: float = 0.0
    activation_upper: float = 1.0
    ipopt_max_iter: int = 300
    ipopt_tol: float = 1e-4
    verbose: bool = False
    subsample: int = 1
    warm_start_multipliers: bool = True
    activation_dynamics: bool = False
    tau_act: float = 0.01
    tau_deact: float = 0.04
    temporal_smoothing: float = 0.0


class DifferentiableStaticOptimizer:

    def __init__(self, model: ModelParams, config: OptimizationConfig):
        self.model = model
        self.config = config
        self._solver = None
        self._build_solver()

    def _build_solver(self):
        n_m = self.model.n_muscles
        n_d = self.model.n_dof
        w   = self.config.reserve_actuator_weight

        a     = ca.MX.sym("a",    n_m)
        t_res = ca.MX.sym("tres", n_d)
        self._t_res_sym = t_res

        # R_scaled = R * diag(F0) передаётся как единый параметр:
        # убирает F0*a из символьного графа, сокращает вектор параметров
        # с (n_d*n_m + n_m + n_d) до (n_d*n_m + n_d).
        Rs     = ca.MX.sym("Rs", n_d * n_m)
        tau_id = ca.MX.sym("ti", n_d)

        R_sc = ca.reshape(Rs, n_d, n_m)
        x    = ca.vertcat(a, t_res)
        cost = self._build_cost(a)
        eq   = ca.mtimes(R_sc, a) + t_res - tau_id

        per_dof = self.config.reserve_weights_per_dof
        if per_dof is not None:
            res_lb = ca.DM(-np.asarray(per_dof, dtype=float))
            res_ub = ca.DM( np.asarray(per_dof, dtype=float))
        else:
            res_lb = ca.DM.ones(n_d) * (-w)
            res_ub = ca.DM.ones(n_d) * w

        lbg = ca.DM.zeros(n_d)
        ubg = ca.DM.zeros(n_d)

        if self.config.temporal_smoothing > 0.0:
            a_prev_p = ca.MX.sym("ap", n_m)
            cost     = cost + self.config.temporal_smoothing * ca.sumsqr(a - a_prev_p)
            p        = ca.vertcat(Rs, tau_id, a_prev_p)
            self._smooth_p = True
        else:
            p              = ca.vertcat(Rs, tau_id)
            self._smooth_p = False

        self._lbg     = lbg
        self._ubg     = ubg
        self._act_dyn = self.config.activation_dynamics
        self._lb_x_np = np.concatenate([
            np.full(n_m, self.config.activation_lower),
            np.array(res_lb).flatten()
        ])
        self._ub_x_np = np.concatenate([
            np.full(n_m, self.config.activation_upper),
            np.array(res_ub).flatten()
        ])

        opts = {
            "ipopt.max_iter":    self.config.ipopt_max_iter,
            "ipopt.tol":         self.config.ipopt_tol,
            "ipopt.print_level": 5 if self.config.verbose else 0,
            "print_time":        self.config.verbose,
        }
        if self.config.warm_start_multipliers:
            opts["ipopt.warm_start_init_point"]      = "yes"
            opts["ipopt.warm_start_bound_push"]      = 1e-9
            opts["ipopt.warm_start_mult_bound_push"] = 1e-9

        self._solver = ca.nlpsol("static_opt", "ipopt",
                                 {"x": x, "f": cost, "g": eq, "p": p}, opts)
        self._n_m = n_m
        self._n_d = n_d

    def _build_cost(self, a: ca.MX) -> ca.MX:
        cf = self.config.cost_function
        if cf == CostFunction.SUM_SQUARES:
            return ca.sumsqr(a)
        elif cf == CostFunction.SUM_CUBES:
            return ca.sum1(a ** 3)
        elif cf == CostFunction.POLYNOMIAL_3:
            return ca.sum1(0.5 * a**2 + 0.33 * a**3)
        elif cf == CostFunction.METABOLIC_PROXY:
            return ca.sum1(0.25 * a + 1.5 * a**2)
        elif cf == CostFunction.SUM_CUBES_STRESS:
            F0 = self.model.max_isometric_force
            w  = ca.DM((F0.mean() / F0) ** 3)
            return ca.sum1(w * a ** 3)
        elif cf == CostFunction.SUM_POW_25:
            return ca.sum1(a ** 2.5)
        elif cf == CostFunction.SUM_CUBES_PENRES:
            return ca.sum1(a ** 3) + 0.1 * ca.sumsqr(self._t_res_sym / self.config.reserve_actuator_weight)
        else:
            raise ValueError(f"Unknown cost function: {cf}")

    def solve_frame(
        self,
        R_scaled: np.ndarray,
        tau_id: np.ndarray,
        x0: Optional[np.ndarray] = None,
        lam_g0: Optional[np.ndarray] = None,
        lam_x0: Optional[np.ndarray] = None,
        a_prev: Optional[np.ndarray] = None,
        dt: float = 0.01,
    ) -> dict:
        n_m, n_d = self._n_m, self._n_d
        if x0 is None:
            x0 = np.zeros(n_m + n_d)

        if self._smooth_p:
            ap    = a_prev if a_prev is not None else np.zeros(n_m)
            p_val = np.concatenate([R_scaled.flatten(), tau_id, ap])
        else:
            p_val = np.concatenate([R_scaled.flatten(), tau_id])

        if self._act_dyn and a_prev is not None:
            lb_x = self._lb_x_np.copy()
            ub_x = self._ub_x_np.copy()
            lb_x[:n_m] = np.maximum(self.config.activation_lower,
                                     a_prev - dt / self.config.tau_deact)
            ub_x[:n_m] = np.minimum(self.config.activation_upper,
                                     a_prev + dt / self.config.tau_act)
        else:
            lb_x = self._lb_x_np
            ub_x = self._ub_x_np

        kwargs = dict(x0=x0, lbx=lb_x, ubx=ub_x,
                      lbg=self._lbg, ubg=self._ubg, p=p_val)
        if self.config.warm_start_multipliers and lam_g0 is not None:
            kwargs["lam_g0"] = lam_g0
            kwargs["lam_x0"] = lam_x0

        sol   = self._solver(**kwargs)
        stats = self._solver.stats()
        x_opt = np.array(sol["x"]).flatten()

        return {
            "activations":     x_opt[:n_m],
            "reserve_torques": x_opt[n_m:],
            "cost":            float(sol["f"]),
            "success":         stats["success"],
            "iter_count":      stats.get("iter_count", -1),
            "lam_g":           sol["lam_g"],
            "lam_x":           sol["lam_x"],
        }

    def solve_trajectory(
        self,
        moment_arms: np.ndarray,
        max_forces: np.ndarray,
        id_torques: np.ndarray,
        time: Optional[np.ndarray] = None,
    ) -> dict:
        n_frames = moment_arms.shape[0]
        n_m = self._n_m
        n_d = self._n_d
        sub = self.config.subsample

        R_scaled = moment_arms * max_forces[np.newaxis, np.newaxis, :]

        frame_indices   = np.arange(0, n_frames, sub)
        activations_sub = np.zeros((len(frame_indices), n_m))
        res_torques_sub = np.zeros((len(frame_indices), n_d))
        costs_sub       = np.zeros(len(frame_indices))
        solve_times_sub = np.zeros(len(frame_indices))
        iter_counts     = np.zeros(len(frame_indices), dtype=int)

        if self._act_dyn and time is not None:
            dts = np.diff(time, prepend=time[0] - (time[1] - time[0]))
            dts = np.clip(dts, 1e-4, 0.1)
        else:
            dts = np.full(n_frames, 0.01)

        t_total_start = time_module.perf_counter()
        x0     = np.zeros(n_m + n_d)
        lam_g  = None
        lam_x  = None
        a_prev = np.zeros(n_m)

        for i, t in enumerate(frame_indices):
            t_start = time_module.perf_counter()
            result  = self.solve_frame(
                R_scaled=R_scaled[t], tau_id=id_torques[t],
                x0=x0, lam_g0=lam_g, lam_x0=lam_x,
                a_prev=a_prev, dt=float(dts[t]),
            )
            solve_times_sub[i]  = time_module.perf_counter() - t_start
            activations_sub[i]  = result["activations"]
            res_torques_sub[i]  = result["reserve_torques"]
            costs_sub[i]        = result["cost"]
            iter_counts[i]      = result["iter_count"]
            x0     = np.concatenate([result["activations"], result["reserve_torques"]])
            lam_g  = result["lam_g"]
            lam_x  = result["lam_x"]
            a_prev = result["activations"]
            if not result["success"]:
                print(f"  [!] Frame {t}: IPOPT did not converge (iter={result['iter_count']})")

        total_time = time_module.perf_counter() - t_total_start

        if sub > 1:
            t_sub  = frame_indices
            t_full = np.arange(n_frames)
            activations     = np.column_stack([np.interp(t_full, t_sub, activations_sub[:, m]) for m in range(n_m)])
            reserve_torques = np.column_stack([np.interp(t_full, t_sub, res_torques_sub[:, d]) for d in range(n_d)])
            costs           = np.interp(t_full, t_sub, costs_sub)
            solve_times     = np.interp(t_full, t_sub, solve_times_sub)
        else:
            activations     = activations_sub
            reserve_torques = res_torques_sub
            costs           = costs_sub
            solve_times     = solve_times_sub

        return {
            "activations":     activations,
            "reserve_torques": reserve_torques,
            "costs":           costs,
            "solve_times":     solve_times,
            "total_time":      total_time,
            "mean_frame_time": np.mean(solve_times_sub),
            "mean_iter_count": float(np.mean(iter_counts)),
            "frames_solved":   len(frame_indices),
        }
