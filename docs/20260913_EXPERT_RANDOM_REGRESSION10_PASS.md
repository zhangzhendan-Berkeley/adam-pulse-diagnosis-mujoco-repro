# 2026-09-13 Expert random regression: 10/10 pass

## Root cause and fix

The pressure controller only checked diagnostic contact at the beginning of a
physics action. A three-finger contact could therefore be lost during the
integration step without setting `pressure_contact_lost`; the expert then kept
raising pressure until timeout. The backend now reports post-step contact loss,
and the expert also treats a pre-step partial contact mask as a recovery trigger.

The final candidate enables contact-first stabilization: once every diagnostic
finger is in the safe force window, pressure stability does not require the
nominal force command to have fully ramped. This prevents unnecessary force
overshoot while retaining the 1.25--10 N safety window and signal-quality gate.

## Regression evidence

Command:

```text
scripts/run_stage3_cuda.sh scripts/collect_mujoco_dataset.py \
  --output outputs/expert_collection_fast_contact_contactfirst_20260913_regression10 \
  --config configs/alignment_pro_custom_right_hand_multifinger_elbow_out_safe_v7_expert_fast_contact_contactfirst.yaml \
  --target-accepted 10 --maximum-attempts 10 --seed 20260912 \
  --capture-interval 2 --record-width 320 --record-height 240 \
  --jpeg-quality 85 --search-strategy signal_quality_early_stop
```

The manifest reports `accepted=10`, `failed=0`, `complete=true` across offsets
from −5.36 mm to +5.23 mm, including tactile noise, proximity dropout,
spikes, action scaling, and command drops. Every accepted episode had signal
quality accepted, dual-contact ratio 1.0, force-safe ratio 1.0, and zero
forbidden contact events.

The final review video is:

`outputs/meeting_experiments_20260913/expert_speed_audit/规范双视角热力图视频_contactfirst_s11/`

Its validator reports `valid=true`, 1920×1080 composite, 20 FPS, 684 frames,
and nonblank front/fingertip views. A recorded nominal pressure dropout was
recovered in 0.25 s (14.50 to 14.75 s) before pressure stabilization.
