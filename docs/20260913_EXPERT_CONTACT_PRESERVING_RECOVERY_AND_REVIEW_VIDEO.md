# 2026-09-13 Expert contact-preserving recovery and review video

## Change

The pressure-loss path now distinguishes partial contact loss from complete
loss. Partial loss uses the configured 0.08 mm correction; when all three pads
are absent it uses a separately bounded 0.45 mm reacquisition stroke. This
avoids repeatedly applying a large correction to a pad that is already seated,
while allowing a full dropout to reacquire the surface.

The review candidate is based on the fast-contact configuration with a 4 N per
finger target, 0.40 N/policy-step pressure ramp, no pressure-stage balance
tilt, and four stable iterations.

## Evidence

- Nominal expert review video:
  `outputs/meeting_experiments_20260913/expert_speed_audit/规范双视角热力图视频_lowforce_s11/`
- `success=true`, signal quality accepted, dual-contact ratio 1.0,
  dominant frequency 1.1667 Hz, maximum forces `[6.06, 3.62, 2.03] N`,
  forbidden contacts 0.
- Initial first contact: 14.4 s; pressure stable: 16.05 s (1.65 s after
  first contact).
- Artifact validator: `valid=true`, composite 1920x1080, 20 FPS, 654 frames,
  duration 32.7 s; front and fingertip views are 1280x720 and nonblank.
- A controlled +2 mm offset run also succeeded with signal quality accepted,
  maximum force `[6.03, 3.62, 2.16] N`, and no forbidden contacts. Recovery
  events reacquired all three pads at 23.0 s and 33.25 s.

## Collection gate

The random collection smoke tests remain failing for the hardest −2.7 to
−3.1 mm scenarios, so no large-scale collection was started. The new video is
the review artifact; wait for user approval before collection.
