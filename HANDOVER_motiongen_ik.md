# Handover — Add QP differential-IK control mode (avatar `teleop-simulator`)

## Objective
Add a second, selectable low-level control mode for the Franka FR3 arms. The existing
**Cartesian impedance** controller stays untouched and remains the default. The new mode,
**JOINT_IK**, converts the incoming Cartesian *position* target into a joint reference via a
resolved-rate (differential) IK solved as a **box-constrained least squares** (pure Eigen,
no new dependency), then tracks that joint reference with the existing **joint impedance**
controller. Goal: enforce joint position/velocity limits as hard constraints and gain
explicit control of the arm's configuration (redundancy), which the current Cartesian-impedance
nullspace does only weakly.

Keep commands position-based: if the operator command stream stops, the goal is frozen, the
IK velocity decays to zero, and the arm holds — same natural stop as today.

## Scope of changes
- **Rename** `include/interpolator.hpp` → `include/MotionGenerator.hpp`, class `Interpolator`
  → `MotionGenerator`. Update all call sites (member `interpolator_` → `motion_gen_` in
  `arm_control.{hpp,cpp}`). Keep ALL existing methods (`planJoint`, `planCartesian`, `step`,
  `getCurrentJoint`, `getCurrentCartesian`, `isDone`, `reset`) working exactly as before —
  HOMING/RECOVERY still use `planJoint`, and CARTESIAN_IMPEDANCE mode still uses
  `planCartesian` + `getCurrentCartesian()`. This is a mechanical rename plus new additions.
- **Add** the resolved-rate IK to `MotionGenerator` (most of the new code lives here).
- **`arm_control`**: add a `ControlMode` enum (config-selected), seed/goal handling on ENGAGED
  entry, call the IK in the ENGAGED branch of the state thread, and switch the control-thread
  ENGAGED torque between `cartesianImpedanceControl` (existing) and `jointImpedanceControl`
  (existing) based on mode. Both code paths remain in the codebase.
- **Config**: add `control_mode` and an `ik:` block per arm.

## Current architecture (for reference)
- `runStateHandler` (~500 Hz): on new UDP cmd builds `T_cmd`, then
  `transformCommandToBase()` → `applySelfCollisionFilter()` → `validateTargetPose()` →
  `interpolator_.planCartesian(getCurrentCartesian(), T_target, LINEAR)`. `T_target` is in the
  **arm base frame**.
- `runControlHandler` (1 kHz franka callback): `interpolator_.step()`, then a `switch(state_)`:
  HOMING/RECOVERING → `jointImpedanceControl` (tracks `getCurrentJoint()`); AWAITING/ENGAGED →
  `cartesianImpedanceControl` (tracks `getCurrentCartesian()`).
- `jointImpedanceControl(rs)` already exists: `tau = Kp⊙(q_target − q) + Kd⊙(−dq) + coriolis`,
  with `q_target = interpolator_.getCurrentJoint()`.
- Kinematics available in `arm_control`: `model->zeroJacobian(franka::Frame::kEndEffector, rs)`
  (6×7, base-frame twist) and `rs.O_T_EE` (base→EE). `T_target` is base-frame. All consistent —
  no new kinematics code needed.
- `kMaxDq` (per-joint |dq| limits) already defined in `cartesianImpedanceControl`; reuse it.

## New design — resolved-rate IK (per IK tick, dt = 1/state_rate)
Inputs: current `q` (7), current EE pose `x = O_T_EE` (base frame), goal `X_d` (= validated
`T_target`, base frame), base-frame Jacobian `J` (6×7), previous solution `u_prev`.

```
e_p   = p_d - p
e_o   = vec( q_d ⊗ q_cur^{-1} )            // base-frame orientation error (same convention as
                                            // cartesianImpedanceControl: q_err = q_d*q_cur.inverse())
v_des = [ Kp_p * e_p ; Kp_o * e_o ]         // 6-vector
clamp ||v_des.head(3)|| <= v_lin_max ; ||v_des.tail(3)|| <= v_ang_max   // Cartesian speed cap
```

Solve for joint velocity `u ∈ R^7`:

```
minimize   || J u - v_des ||^2_Wtask  +  λ ||u||^2  +  μ || u - u_post ||^2
   u
subject to L_i <= u_i <= U_i                (per-joint box)
```

- `u_post = Kp_posture * (q0 - q)`  → posture pull (explicit, weighted redundancy resolution).
- `λ` = Levenberg–Marquardt damping (singularity robustness; same idea as the existing damped pinv).
- Per-joint box bounds fold ALL safety into per-joint limits (so it stays box-constrained):
```
U_i = min( +qd_max_i, (q_max_i - γ - q_i)/T_brake, u_prev_i + a_max*dt )
L_i = max( -qd_max_i, (q_min_i + γ - q_i)/T_brake, u_prev_i - a_max*dt )
```
  term 1 = velocity limit (use `kMaxDq` / FR3 datasheet), term 2 = position-limit braking
  (don't command a velocity that breaches the limit within horizon `T_brake`), term 3 = optional
  acceleration limit. γ = small safety buffer (rad).

Integrate and output the joint reference:
```
u = clamp_to_box( solve );   q_ref += u * dt;   u_prev = u;
```
`q_ref` is what the control loop tracks with `jointImpedanceControl`. Optionally expose `u` as a
velocity feedforward.

### Box-constrained LS solver (pure Eigen, deterministic)
Because every constraint is a per-joint bound, use a small active-set clamp loop — no QP lib:
```
Build normal equations for the unconstrained weighted LS:
   A = J^T Wtask J + (λ + μ) I            (7x7, SPD)
   b = J^T Wtask v_des + μ u_post
free = {0..6}; clamped = {}
loop (<= 7 iters):
   solve A_free u_free = b_free  (LDLT), with clamped joints fixed at their bound and moved to RHS
   if any free u_i violates [L_i,U_i]: clamp the worst one to its bound, move to `clamped`, repeat
   else break
return u
```
7×7 SPD solves, ≤7 iterations → single-digit microseconds, bounded iteration count (RT-safe).
(If coupled constraints are ever needed — e.g. dual-arm self-collision — swap this for
**proxsuite** or **qpOASES**. Not needed now; self-collision is still handled by
`applySelfCollisionFilter` on the goal before IK.)

## `MotionGenerator` additions (new API)
Keep existing members/methods. Add:
```cpp
struct IkConfig {
    Eigen::Vector3d Kp_p; double Kp_o;           // task P-gains (pos / ori)
    double v_lin_max, v_ang_max;                 // Cartesian speed caps
    double lambda;                               // LM damping
    Eigen::Matrix<double,7,1> Kp_posture;        // posture pull gains
    Eigen::Matrix<double,7,1> q0;                // posture target (use arm q0_)
    Eigen::Matrix<double,7,1> qd_max;            // joint vel limits (reuse kMaxDq)
    Eigen::Matrix<double,7,1> q_min, q_max;      // from config
    double T_brake, a_max, gamma;                // limit-braking horizon, accel cap, buffer
    Eigen::Matrix<double,6,1> Wtask;             // task weights (diag)
};

void   setIkConfig(const IkConfig& c);
void   seedJointReference(const Eigen::Matrix<double,7,1>& q);  // q_ref_ = q; u_prev_ = 0
void   setCartesianGoal(const Eigen::Isometry3d& X_d);          // base frame
// One resolved-rate step. Returns updated q_ref. Caller supplies fresh q, J, x.
Eigen::Matrix<double,7,1> stepIk(const Eigen::Matrix<double,7,1>& q,
                                 const Eigen::Matrix<double,6,7>& J,
                                 const Eigen::Isometry3d& x,
                                 double dt);
Eigen::Matrix<double,7,1> getJointReference() const;            // returns q_ref_
```
Internal state: `q_ref_`, `u_prev_`, `X_goal_`, `IkConfig`, mutex. `stepIk` implements the math
above. Keep it model-agnostic — the caller passes `J` and `x`.

## `arm_control` wiring
1. Add enum + member:
```cpp
enum class ControlMode { CARTESIAN_IMPEDANCE, JOINT_IK };
ControlMode control_mode_ = ControlMode::CARTESIAN_IMPEDANCE;  // from config["control_mode"]
```
2. Build `IkConfig` in the constructor (from the new `ik:` config block; `q0`/`q_min`/`q_max`
   already loaded; `qd_max` = the existing `kMaxDq` values) and call `motion_gen_.setIkConfig(...)`.
3. On ENGAGED entry (where state transitions to ENGAGED): seed to avoid a jump —
   `motion_gen_.seedJointReference(current q)` and `motion_gen_.setCartesianGoal(current O_T_EE)`.
4. `runStateHandler`, ENGAGED branch:
   - On new cmd: same `transformCommandToBase` → `applySelfCollisionFilter` → `validateTargetPose`
     as today. Then, if `JOINT_IK`: `motion_gen_.setCartesianGoal(T_target)` (instead of
     `planCartesian`). If `CARTESIAN_IMPEDANCE`: keep `planCartesian(...)` exactly as now.
   - Every iteration in `JOINT_IK` (whether or not a new cmd arrived): read `q` from
     `current_state`, compute `J = model->zeroJacobian(kEndEffector, current_state)` and
     `x = O_T_EE`, call `motion_gen_.stepIk(q, J, x, dt_state)`. (Goal frozen when no new cmd →
     converges and stops.)
5. `runControlHandler`, ENGAGED/AWAITING case:
   - `CARTESIAN_IMPEDANCE` → `cartesianImpedanceControl(rs)` (unchanged).
   - `JOINT_IK` → `jointImpedanceControl(rs)`. Make `jointImpedanceControl`'s `q_target` come from
     `motion_gen_.getJointReference()` when in JOINT_IK (today it reads `getCurrentJoint()`); a
     clean way is to have `getJointReference()` and `getCurrentJoint()` both return `q_ref_`/joint
     plan respectively and pick per mode, or in JOINT_IK have `stepIk` also feed the joint plan.
   - HOMING/RECOVERING unchanged (joint plan via `planJoint`).
6. Rate note: IK runs at state-thread rate (~500 Hz); the 1 kHz loop tracks `q_ref`. Because
   `q_ref` is smooth and slow, joint impedance handles the rate gap; add a first-order filter on
   `q_ref` in the control loop only if stair-stepping appears.

## Config additions (per arm, e.g. robot_config_avatar.yaml / _local.yaml)
```yaml
    control_mode: cartesian_impedance   # or: joint_ik
    ik:
      kp_p:  [6.0, 6.0, 6.0]            # task pos gain (1/s)  -- tune
      kp_o:  4.0                        # task ori gain (1/s)  -- tune
      v_lin_max: 0.5                    # m/s  (match interpolator max_linear_vel)
      v_ang_max: 0.8                    # rad/s
      lambda: 0.01                      # LM damping
      kp_posture: [2,2,2,2,2,2,2]       # redundancy pull toward q0 -- raise to fight bad configs
      qd_max: [2.175,2.175,2.175,2.175,2.610,2.610,2.610]   # reuse kMaxDq / verify FR3 datasheet
      t_brake: 0.10                     # s, position-limit braking horizon (~5-10x dt)
      a_max:  10.0                      # rad/s^2 accel cap (optional smoothing)
      gamma:  0.05                      # rad, limit safety buffer
      wtask:  [1,1,1,1,1,1]             # task weights (pos x3, ori x3)
```

## Safety / behavior properties
- Joint velocity, position-limit, and acceleration limits are HARD box constraints — the IK
  slows/deviates from the Cartesian target rather than violate them. Stronger than the current
  joint-limit-avoidance torque (which competes with the task torque).
- Connection loss → goal frozen → `v_des → 0` → `u → 0` → `q_ref` holds → arm stops. Preserved.
- `validateTargetPose` (workspace/tilt/velocity clamps) and `applySelfCollisionFilter` still run
  on `T_target` before it becomes the IK goal — keep them as the outer guard.
- Trade-off vs Cartesian impedance: strict Cartesian passivity is lost; compliance becomes
  joint-space. Fine for free-space pick-and-place. Stability maintained via the vel/accel limits
  and reference smoothing.

## Rollout & testing
1. Land the rename + JOINT_IK code with `control_mode: cartesian_impedance` (default) → verify
   zero behavior change first.
2. (Optional, recommended) Shadow mode: in CARTESIAN_IMPEDANCE, also run `stepIk` and log `q_ref`
   vs actual `q` and the commanded torques — inspect joint trajectories/configuration before
   handing control over. Cheap, no behavior change.
3. Flip one arm to `joint_ik`. Test: slow Cartesian moves (tracking), drive toward a joint limit
   (must brake, not breach), reach poses that previously caused weird elbow configs (raise
   `kp_posture` until the configuration is what you want), fast setpoint jumps (must stay within
   velocity/accel limits), and command-stop (must hold).
4. Tune `Kp_p/Kp_o` for responsiveness, `lambda` for singularity behavior, `kp_posture` vs
   `wtask` for the configuration/tracking trade-off.

## Open knobs / verify
- Confirm FR3 joint velocity limits (`qd_max`) against the official datasheet (placeholder reuses
  Panda-style `kMaxDq`).
- Joint-impedance gains for tracking `q_ref` (start from existing homing `kp_joint/kd_joint`).
- `T_brake` vs `dt_state`: keep `T_brake` ≈ 5–10·dt for smooth deceleration near limits.

## Expected outcome
- "Arm runs into joint limits / velocities" → fixed (hard constraints).
- "Arm moves into strange configurations" → substantially improved (posture is an explicit,
  weighted objective solved jointly with the task within limits; bias/raise `kp_posture`).
  Caveat: a pose that genuinely requires an awkward configuration still requires it — kinematics.
