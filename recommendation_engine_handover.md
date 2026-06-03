# Recommendation Engine — Concept Handover

## What it is

A separate process (not embedded in UE or the avatar) that sits between the local digital twin, the real avatar, and the operator interface. It observes all available state, infers operator intent and risk, and issues subtle guidance back to the operator through multiple perceptual channels. It does not control the robot — it shapes what the operator perceives.

Working name: **recommendation engine** or **nudging system**.

## Inputs

- Twin state: predicted EE wrench, joint positions, contact flags (local, near-zero latency)
- Real avatar state: EE pose, joint positions, tau_ext (network-delayed)
- Interface state: controller pose/velocity, clutch state, gear, gripper
- Episode/scene knowledge: object positions, grasp targets, task phase
- System latency estimate (updated live)

## Outputs (operator perceptual channels)

- **Haptic**: vibration commands to the VR controller (onset buzz, intensity ramp)
- **Audio**: ambient tones, prerecorded cues, TTS
- **Visual**: ghost overlay color/opacity changes, HUD annotations

## Core use cases established so far

**1. Contact warning**
Twin predicts rising EE wrench → haptic onset buzz + audio cue. Value: the twin is local so this warning arrives before the real robot makes contact, unlike tau_ext feedback which is latency-delayed.

**2. Predictive grasp timing**
Arm moving toward grasp target + known system latency → compute when the real EE will arrive → issue "close now" cue (ghost turns green) so the operator's gripper command arrives at the robot at the correct moment. This moves the latency compensation problem from the control domain into the human-machine interface domain.

## Key design principles

- **Nudging, not overriding**: all outputs are informational, operator retains full control
- **Distinct perceptual channels**: haptic for onset events, audio for magnitude/urgency, visual for spatial/timing cues — these don't compete
- **Separate process**: decoupled from UE and avatar so inference logic can be updated independently and learned models can be swapped in later
- **Everything is logged**: every mediator output is a timestamped labeled event, which becomes training data for improving the engine itself

## Open questions for architecture discussion

- How is operator intent represented? (raw controller velocity is easy; inferred goal requires scene knowledge)
- Twin sync strategy: twin runs its own control loop on the same commands as the real robot, with a secondary correction term toward a forward-predicted real joint configuration (Smith predictor + observer hybrid). Not fully designed yet.
- Fallback for unknown contacts: twin only predicts contacts in the known scene model. Real tau_ext from the avatar serves as a secondary channel for unexpected contacts.
- How intrusive is too intrusive? Need to establish thresholds empirically.
